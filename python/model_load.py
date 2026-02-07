"""
Model loading and management for PnPXAI Tool.
Manages Loaded Model + Treatment combination for memory-efficient session handling.
Models are listed from config.yaml (llms:); loading logic follows backup/utils.py.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to use backup utils if available
try:
    from utils import get_config_models, load_llm, get_model_status
except ImportError:
    # Fallback when running from project root with backup
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from backup.utils import get_config_models, load_llm, get_model_status
    except ImportError:
        from functools import lru_cache
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Project root config.yaml
        _CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
        _MODEL_ALIASES: Dict[str, str] = {}

        def get_config_models() -> List[str]:
            """Read model names from config.yaml llms: list."""
            if not _CONFIG_PATH.exists():
                return []
            models: List[str] = []
            in_llms = False
            for raw in _CONFIG_PATH.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("llms:"):
                    in_llms = True
                    continue
                if not in_llms:
                    continue
                if line.startswith("-"):
                    item = line[1:].strip()
                    if item:
                        models.append(item)
                else:
                    break
            return models

        def _resolve_model_name(model_key: str) -> str:
            return _MODEL_ALIASES.get(model_key, model_key)

        def _load_base_model(base_model_name: str):
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                output_attentions=True,
            )
            model.eval()
            try:
                with torch.inference_mode():
                    enc = tokenizer("hello", return_tensors="pt")
                    device = next(model.parameters()).device
                    if device.type == "meta":
                        raise RuntimeError("meta device")
                    out = model(**enc.to(device), output_attentions=True)
                if not out.attentions or any(a is None for a in out.attentions):
                    raise RuntimeError("attentions None")
            except Exception as e:
                print(f"Warning: Falling back to CPU load due to: {e}")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=None,
                    output_attentions=True,
                )
                model.eval()
                model = model.to(torch.device("cpu"))
            return model, tokenizer

        @lru_cache(maxsize=2)
        def load_llm(model_key: str) -> Tuple[Any, Any]:
            allowed = set(get_config_models())
            if model_key not in allowed:
                raise ValueError(f"Unknown model key: {model_key}. Allowed: {sorted(allowed)}")
            model_name = _resolve_model_name(model_key)
            model, tokenizer = _load_base_model(model_name)
            return tokenizer, model

        def get_model_status(model_key: str) -> Dict[str, Any]:
            tokenizer, model = load_llm(model_key)
            config = model.config
            num_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
            num_heads = getattr(config, "num_attention_heads", None) or getattr(config, "n_head", None)
            name = getattr(config, "name_or_path", model_key) or model_key
            num_params = sum(p.numel() for p in model.parameters())
            device_stats: List[Dict[str, Any]] = []
            seen: Dict[str, Dict[str, Any]] = {}
            for p in model.parameters():
                if p.device.type == "meta":
                    continue
                dev_key = str(p.device)
                if dev_key not in seen:
                    seen[dev_key] = {"device": dev_key, "memory_bytes": 0, "memory_gb": 0.0}
                seen[dev_key]["memory_bytes"] += p.numel() * (p.element_size() or 4)
            for dev_key, info in seen.items():
                info["memory_gb"] = round(info["memory_bytes"] / (1024**3), 3)
                if info["device"].startswith("cuda"):
                    idx = int(info["device"].split(":")[-1]) if ":" in info["device"] else 0
                    try:
                        info["capacity_gb"] = round(torch.cuda.get_device_properties(idx).total_memory / (1024**3), 2)
                    except Exception:
                        info["capacity_gb"] = None
                else:
                    info["capacity_gb"] = None
                device_stats.append(info)
            device_stats.sort(key=lambda x: (0 if x["device"] == "cpu" else 1, x["device"]))
            return {
                "model_key": model_key,
                "name": name,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "num_parameters": num_params,
                "device_status": device_stats,
            }


def get_available_models() -> List[str]:
    """Return list of available model keys from config."""
    return get_config_models()


def load_model(model_key: str) -> None:
    """Load model into memory (warm cache)."""
    load_llm(model_key)


def get_status(model_key: str) -> Dict[str, Any]:
    """Get model status (layers, heads, etc.)."""
    return get_model_status(model_key)
