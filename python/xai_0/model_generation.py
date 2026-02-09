"""
XAI Level 0: Generation logic - text completion, chat completion, token counts.
Uses load_llm and get_model_device from python.model_load.
"""

from typing import Any, Dict, List

import torch

from python.model_load import get_model_device, load_llm


def _simple_chat_prompt_to_ids(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool = True):
    """Build prompt string for models without chat template (e.g. GPT-2)."""
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


def chat_completion(
    model_key: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 50,
) -> str:
    """
    Multi-round chat completion using the model's chat template when available.

    Args:
        model_key: Model key from config.
        messages: List of {"role": "user"|"assistant"|"system", "content": "..."}.
        max_new_tokens: Max tokens to generate.
        do_sample: Whether to sample (vs greedy).
        temperature: Sampling temperature when do_sample is True.
        top_p: Top-p (nucleus) when do_sample is True.
        top_k: Top-k when do_sample is True.

    Returns:
        Assistant reply text (only the newly generated part).
    """
    tokenizer, model = load_llm(model_key)
    device = get_model_device(model)

    if getattr(tokenizer, "apply_chat_template", None):
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            input_ids = _simple_chat_prompt_to_ids(tokenizer, messages)
    else:
        input_ids = _simple_chat_prompt_to_ids(tokenizer, messages)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)
    prompt_length = input_ids.shape[1]

    gen_kw: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kw["temperature"] = temperature
        if top_p < 1.0:
            gen_kw["top_p"] = top_p
        if top_k > 0:
            gen_kw["top_k"] = top_k

    with torch.inference_mode():
        out = model.generate(input_ids, **gen_kw)

    new_ids = out[0][prompt_length:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def text_completion(
    model_key: str,
    prompt: str,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    top_p: float = 1.0,
    top_k: int = 50,
) -> str:
    """
    Causal LM completion from a single prompt string.
    Uses do_sample with temperature, top_p, top_k when temperature > 0.
    """
    tokenizer, model = load_llm(model_key)
    device = get_model_device(model)
    enc = tokenizer(prompt.strip(), return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    prompt_length = input_ids.shape[1]

    do_sample = temperature > 0
    gen_kw: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kw["temperature"] = temperature
        if top_p < 1.0:
            gen_kw["top_p"] = top_p
        if top_k > 0:
            gen_kw["top_k"] = top_k

    with torch.inference_mode():
        out = model.generate(input_ids, **gen_kw)

    new_ids = out[0][prompt_length:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def get_text_token_count(model_key: str, text: str) -> int:
    """Return the number of tokens for the given text (e.g. generated reply)."""
    tokenizer, _ = load_llm(model_key)
    enc = tokenizer.encode(text.strip(), add_special_tokens=False)
    return len(enc)


def get_cache_token_count(model_key: str, messages: List[Dict[str, str]]) -> int:
    """Return the number of input tokens for the given messages (prompt length)."""
    tokenizer, _ = load_llm(model_key)
    if getattr(tokenizer, "apply_chat_template", None):
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            input_ids = _simple_chat_prompt_to_ids(tokenizer, messages)
    else:
        input_ids = _simple_chat_prompt_to_ids(tokenizer, messages)
    if input_ids.dim() == 1:
        return input_ids.numel()
    return input_ids.shape[1]
