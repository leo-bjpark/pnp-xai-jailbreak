"""
Utilities for loading lightweight LLMs and computing attention head scores.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


# Project root config (same as python/model_load.py)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Allow short aliases in config.yaml.
MODEL_ALIASES: Dict[str, str] = {
    "tiny-gpt2": "sshleifer/tiny-gpt2",
}


def _load_llms_from_config() -> Any:
    """Load llms section from config.yaml. Returns either a list (flat) or dict (group -> list)."""
    if not CONFIG_PATH.exists():
        return ["tiny-gpt2", "distilgpt2"]
    with open(CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("llms") or ["tiny-gpt2", "distilgpt2"]


def get_config_models() -> List[str]:
    """
    Return flat list of model names from config.yaml.
    Supports both flat (llms: [a, b]) and two-level (llms: group: [a, b]) format.
    """
    llms = _load_llms_from_config()
    if isinstance(llms, dict):
        return [m for models in llms.values() for m in (models or [])]
    if isinstance(llms, list):
        return [m for m in llms if isinstance(m, str)]
    return ["tiny-gpt2", "distilgpt2"]


def get_config_models_grouped() -> Dict[str, List[str]]:
    """
    Return llms as grouped dict (group_name -> list of model ids).
    If config uses flat list, returns {"": flat_list}.
    """
    llms = _load_llms_from_config()
    if isinstance(llms, dict):
        return {k: (v if isinstance(v, list) else []) for k, v in llms.items()}
    if isinstance(llms, list):
        flat = [m for m in llms if isinstance(m, str)]
        return {"": flat} if flat else {"": ["tiny-gpt2", "distilgpt2"]}
    return {"": ["tiny-gpt2", "distilgpt2"]}


def resolve_model_name(model_key: str) -> str:
    """
    Map a config-provided key (e.g. tiny-gpt2) to an HF repo id.
    """
    return MODEL_ALIASES.get(model_key, model_key)


def get_model_device(model) -> torch.device:
    """
    Safely get the device of a model, avoiding meta device issues.

    Args:
        model: PyTorch model

    Returns:
        torch.device: The device the model is on (defaults to CPU if detection fails)
    """
    try:
        # Try to get device from parameters (most reliable)
        device = next(model.parameters()).device
        # Ensure it's not meta device
        if device.type == "meta":
            return torch.device("cpu")
        return device
    except Exception:
        # Fallback to CPU if we can't determine device
        return torch.device("cpu")


def get_model_status(model_key: str) -> Dict[str, Any]:
    """
    Load model (or use cache), then return status: num_layers, num_heads, name,
    num_parameters, and per-device memory/placement.
    """
    tokenizer, model = load_llm(model_key)
    config = model.config

    num_layers = getattr(config, "num_hidden_layers", None) or getattr(
        config, "n_layer", None
    )
    num_heads = getattr(config, "num_attention_heads", None) or getattr(
        config, "n_head", None
    )
    name = getattr(config, "name_or_path", model_key) or model_key
    num_params = sum(p.numel() for p in model.parameters())

    # Per-device: memory used (params) and optional total capacity
    device_stats: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}

    for p in model.parameters():
        if p.device.type == "meta":
            continue
        dev_key = str(p.device)
        if dev_key not in seen:
            seen[dev_key] = {
                "device": dev_key,
                "memory_bytes": 0,
                "memory_gb": 0.0,
            }
        seen[dev_key]["memory_bytes"] += p.numel() * (p.element_size() or 4)

    for dev_key, info in seen.items():
        info["memory_gb"] = round(info["memory_bytes"] / (1024**3), 3)
        if info["device"].startswith("cuda"):
            idx = int(info["device"].split(":")[-1]) if ":" in info["device"] else 0
            try:
                total = torch.cuda.get_device_properties(idx).total_memory
                info["capacity_gb"] = round(total / (1024**3), 2)
            except Exception:
                info["capacity_gb"] = None
        else:
            info["capacity_gb"] = None
        device_stats.append(info)

    device_stats.sort(key=lambda x: (0 if x["device"] == "cpu" else 1, x["device"]))

    config_dict = {}
    if hasattr(config, "to_dict"):
        try:
            config_dict = config.to_dict()
        except Exception:
            config_dict = {}

    # Module structure: same as print(model) (PyTorch __str__ tree)
    try:
        modules_str = str(model)
        max_lines = 4000
        lines = modules_str.split("\n")
        if len(lines) > max_lines:
            modules_str = "\n".join(lines[:max_lines]) + "\n... (truncated)"
    except Exception:
        modules_str = ""

    return {
        "model_key": model_key,
        "name": name,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_parameters": num_params,
        "device_status": device_stats,
        "config": config_dict,
        "modules": modules_str,
    }


def _simple_chat_prompt_to_ids(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool = True):
    """Build prompt string for models without chat template (e.g. GPT-2). Used by circuit/QA."""
    parts = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            parts.append(f"User: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}\n")
        else:
            parts.append(f"{content}\n")
    if add_generation_prompt:
        parts.append("Assistant: ")
    text = "".join(parts)
    return tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)


def _circuit_forward(
    tokenizer,
    model,
    device: torch.device,
    full_ids: List[int],
) -> Dict[str, Any]:
    """Run forward on full_ids and return tokens, layer attention from last, top next token."""
    tokens = tokenizer.convert_ids_to_tokens(full_ids)
    full_input = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(full_input, output_attentions=True)
    attentions = outputs.attentions
    n_layers = len(attentions)
    layer_attention_from_last: List[List[float]] = []
    for layer_attn in attentions:
        last_to_all = layer_attn[0, :, -1, :].float().mean(0).cpu().tolist()
        layer_attention_from_last.append(last_to_all)
    logits = outputs.logits
    next_id = logits[0, -1, :].argmax(dim=-1).item()
    top_next_token = tokenizer.convert_ids_to_tokens([next_id])[0]
    return {
        "tokens": tokens,
        "layer_attention_from_last": layer_attention_from_last,
        "top_next_token": top_next_token,
        "num_layers": n_layers,
        "seq_len": full_input.shape[1],
    }


def circuit_run_simple(model_key: str, text: str) -> Dict[str, Any]:
    """단순 Forward: 주어진 입력 텍스트를 인코딩한 뒤 forward, 해석 결과 반환."""
    tokenizer, model = load_llm(model_key)
    device = get_model_device(model)
    enc = tokenizer(text.strip(), return_tensors="pt", add_special_tokens=True)
    full_ids = enc["input_ids"][0].tolist()
    return _circuit_forward(tokenizer, model, device, full_ids)


def circuit_run_generate(
    model_key: str,
    prompt: str,
    max_new_tokens: int = 64,
) -> Dict[str, Any]:
    """생성기반: 프롬프트로 생성한 뒤 전체 시퀀스에 대해 forward, 해석 결과 반환."""
    tokenizer, model = load_llm(model_key)
    device = get_model_device(model)
    enc = tokenizer(prompt.strip(), return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    with torch.inference_mode():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    full_ids = out_ids[0].tolist()
    return _circuit_forward(tokenizer, model, device, full_ids)


def _user_qa_to_ids(tokenizer, user_prompt: str, generation: str):
    """Build token ids from user prompt + generation using chat template or simple format."""
    messages = [
        {"role": "user", "content": user_prompt.strip()},
        {"role": "assistant", "content": generation.strip()},
    ]
    if getattr(tokenizer, "apply_chat_template", None):
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            )
        except Exception:
            input_ids = _simple_chat_prompt_to_ids(tokenizer, messages, add_generation_prompt=False)
    else:
        input_ids = _simple_chat_prompt_to_ids(tokenizer, messages, add_generation_prompt=False)
    if input_ids.dim() == 1:
        return input_ids.tolist()
    return input_ids[0].tolist()


def circuit_run_user_qa(
    model_key: str,
    user_prompt: str,
    generation: str,
) -> Dict[str, Any]:
    """사용자 입력 질문-대답: User prompt + Generation을 chat_template으로 붙인 뒤 forward, 해석 결과 반환."""
    tokenizer, model = load_llm(model_key)
    device = get_model_device(model)
    full_ids = _user_qa_to_ids(tokenizer, user_prompt, generation)
    return _circuit_forward(tokenizer, model, device, full_ids)


def circuit_run(
    model_key: str,
    prompt: str,
    max_new_tokens: int = 64,
) -> Dict[str, Any]:
    """Alias for backward compatibility: same as circuit_run_generate."""
    return circuit_run_generate(model_key, prompt, max_new_tokens=max_new_tokens)


@lru_cache(maxsize=2)
def load_llm(model_key: str):
    """
    Load and cache a model/tokenizer pair.

    Args:
        model_key: Key from AVAILABLE_MODELS.

    Returns:
        (tokenizer, model) tuple.
    """
    # Validate model exists in config list (prevent random remote fetches).
    allowed = set(get_config_models())
    if model_key not in allowed:
        raise ValueError(f"Unknown model key: {model_key}. Allowed: {sorted(allowed)}")

    model_name = resolve_model_name(model_key)
    model, tokenizer = load_base_model(model_name)
    return tokenizer, model


def load_base_model(base_model_name: str):
    """
    User-style loader: returns (model, tokenizer).

    Important: Some models + `device_map="auto"` can return `outputs.attentions`
    as tuples of Nones. Since this app *requires* attentions, we auto-fallback
    to a simple CPU load if we detect that issue.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    def _load(device_map):
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            output_attentions=True,
        )
        model.eval()
        return model

    # Try accelerate-style loading first (user requested).
    model = _load(device_map="auto")

    # Check if model is on meta device (no actual data)
    def _is_meta_device(model):
        """Check if any parameter is on meta device."""
        try:
            # Check parameters first (most reliable)
            # If any parameter is on meta, the model is on meta
            for param in model.parameters():
                if param.device.type == "meta":
                    return True
            # If no parameters are on meta, model is not on meta
            # We avoid checking model.device directly as it may trigger errors
            return False
        except Exception:
            # If we can't check parameters, assume it might be problematic
            # and let the fallback handle it
            return False

    # Smoke test: verify attentions are real tensors, not None, and not on meta device.
    try:
        if _is_meta_device(model):
            raise RuntimeError("Model is on meta device")
        with torch.inference_mode():
            enc = tokenizer("hello", return_tensors="pt")
            # Get actual device from model (not meta)
            # Safely get device from parameters
            device = next(model.parameters()).device
            if device.type == "meta":
                raise RuntimeError("Model device is meta")
            out = model(**enc.to(device), output_attentions=True)
        if not out.attentions or any(a is None for a in out.attentions):
            raise RuntimeError("attentions are None")
    except (RuntimeError, Exception) as e:
        # Fallback to plain CPU load to guarantee attentions exist.
        # This handles meta device, None attentions, and other issues
        print(f"Warning: Falling back to CPU load due to: {e}")
        # Reload without device_map to avoid meta device issues
        # When device_map=None, model loads to CPU by default
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,  # Explicitly None to avoid meta device
            output_attentions=True,
        )
        model.eval()
        # Explicitly ensure model is on CPU (not meta)
        model = model.to(torch.device("cpu"))

    return model, tokenizer


def _find_token_positions(sequence_ids: List[int], pattern_ids: List[int]) -> List[int]:
    """
    Find token positions for a pattern (sub-sequence) within a sequence.
    Returns starting indices for each match.
    """
    if not pattern_ids or len(pattern_ids) > len(sequence_ids):
        return []
    matches: List[int] = []
    for i in range(len(sequence_ids) - len(pattern_ids) + 1):
        if sequence_ids[i : i + len(pattern_ids)] == pattern_ids:
            matches.append(i)
    return matches


def _token_mask(input_ids: List[int], phrases: Sequence[str], tokenizer) -> torch.Tensor:
    """
    Build a boolean mask of positions that match any phrase.
    """
    mask = torch.zeros(len(input_ids), dtype=torch.bool)
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            continue
        pattern_ids = tokenizer.encode(phrase, add_special_tokens=False)
        for start in _find_token_positions(input_ids, pattern_ids):
            mask[start : start + len(pattern_ids)] = True
    return mask


def compute_head_scores(
    tokenizer,
    model,
    text: str,
    positives: Sequence[str],
    negatives: Sequence[str],
    mode: str = "final_token",
) -> Tuple[List[List[float]], List[str]]:
    """
    Compute per-head scores = mean(attn to positives) - mean(attn to negatives)
    for the final token, across all layers/heads.

    Returns:
        scores: List[layers][heads]
        tokens: decoded tokens of the input (for potential UI tooltips)
    """
    if not text.strip():
        raise ValueError("Input text is empty.")

    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", None)
    input_ids_list: List[int] = input_ids[0].tolist()

    pos_mask = _token_mask(input_ids_list, positives, tokenizer)
    neg_mask = _token_mask(input_ids_list, negatives, tokenizer)

    with torch.inference_mode():
        device = get_model_device(model)
        outputs = model(
            **encoded.to(device),
            output_attentions=True,
        )

    attentions: Sequence[torch.Tensor] = outputs.attentions
    # Each attention: (batch, heads, seq, seq)
    last_index = attention_mask.sum().item() - 1 if attention_mask is not None else input_ids.size(1) - 1

    scores: List[List[float]] = []
    for layer_attn in attentions:
        head_scores: List[float] = []
        # Move to CPU for lightweight processing.
        layer_attn_cpu = layer_attn[0].detach().cpu()  # (heads, seq, seq)

        if mode == "final_token":
            # Query = final token
            query_slice = layer_attn_cpu[:, last_index, :]  # (heads, seq)
        elif mode == "mean_query":
            # Query = mean over all tokens
            query_slice = layer_attn_cpu.mean(dim=1)  # (heads, seq)
        else:
            raise ValueError(f"Unknown score mode: {mode}")

        pos_scores = query_slice[:, pos_mask].mean(dim=1) if pos_mask.any() else torch.zeros(query_slice.size(0))
        neg_scores = query_slice[:, neg_mask].mean(dim=1) if neg_mask.any() else torch.zeros(query_slice.size(0))
        head_scores_tensor = pos_scores - neg_scores
        head_scores.extend(head_scores_tensor.tolist())
        scores.append(head_scores)

    decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids_list)
    return scores, decoded_tokens