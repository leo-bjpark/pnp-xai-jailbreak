"""
Residual Concept Detection (2.0.1).

- Load dataset from saved variable (pipeline)
- Run model with forward hooks:
  - attn: MLP input (value after attn block, i.e. residual + attn_output)
  - mlp: block output (value after MLP block)
- Compute positive - negative mean direction per block
- Output: direction vectors for attn, mlp across all layers
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch


def _build_hook_names_residual(
    layer_config: Dict[str, str],
    num_layers: int,
) -> List[str]:
    """
    Build output keys: attn_out, attn_block_out, mlp_out, mlp_block_out per layer.
    Returns list like [base.0.attn_out, base.0.attn_block_out, base.0.mlp_out, base.0.mlp_block_out, ...]
    """
    base = (layer_config.get("layer_base") or "").strip()
    if not base:
        return []
    names: List[str] = []
    for i in range(num_layers):
        layer_prefix = f"{base}.{i}"
        names.append(f"{layer_prefix}.attn_out")
        names.append(f"{layer_prefix}.attn_block_out")
        names.append(f"{layer_prefix}.mlp_out")
        names.append(f"{layer_prefix}.mlp_block_out")
    return names


def _get_num_layers(model: torch.nn.Module, layer_base: str) -> int:
    """Discover number of layers by inspecting model structure."""
    try:
        mod = _resolve_module(model, layer_base)
        if mod is not None and hasattr(mod, "__len__"):
            return len(mod)
    except Exception:
        pass
    count = 0
    prefix = layer_base + "."
    for name, _ in model.named_modules():
        if name and name.startswith(prefix):
            rest = name[len(prefix) :]
            if rest and rest.split(".")[0].isdigit():
                idx = int(rest.split(".")[0])
                count = max(count, idx + 1)
    return count


def _resolve_module(model: torch.nn.Module, name: str) -> Optional[torch.nn.Module]:
    """Get submodule by name (e.g. 'model.layers.0')."""
    try:
        return dict(model.named_modules()).get(name)
    except Exception:
        return None


def _extract_tensor(outp: Any) -> Optional[torch.Tensor]:
    """Extract hidden states from module output (handle tuple/cache)."""
    t = outp[0] if isinstance(outp, tuple) else outp
    return t.detach() if t is not None else None


def _agg_token_sum_count(
    t: torch.Tensor, token_location: Union[str, List[int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    (batch, seq, hidden) -> (sum over tokens, count per example).
    Returns (batch, hidden), (batch,) for memory-efficient aggregation.
    """
    if token_location == "full":
        s = t.sum(dim=1)
        n = t.new_full((t.shape[0],), t.shape[1], dtype=torch.long)
        return s, n
    idx = token_location
    idx_list = [idx] if isinstance(idx, int) else list(idx)
    valid = [i for i in idx_list if 0 <= i < t.shape[1]]
    if not valid:
        s = t.sum(dim=1)
        n = t.new_full((t.shape[0],), t.shape[1], dtype=torch.long)
        return s, n
    sel = t[:, valid, :]
    s = sel.sum(dim=1)
    n = s.new_full((t.shape[0],), len(valid), dtype=torch.long)
    return s, n


def _extract_input(inp: Any) -> Optional[torch.Tensor]:
    """Extract hidden states from module input (handle tuple)."""
    x = inp[0] if isinstance(inp, (tuple, list)) else inp
    return x.detach() if x is not None else None


def _get_layer_outputs_residual(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: List[str],
    layer_config: Dict[str, str],
    num_layers: int,
    output_keys: List[str],
    token_location: Union[str, List[int]],
    device: torch.device,
    batch_size: int,
    progress_callback=None,
) -> Dict[str, torch.Tensor]:
    """
    Run model with hooks:
    - attn: MLP input (value after attn block)
    - mlp: block output (value after MLP block)
    Returns: {output_key: tensor of shape (total_examples, model_dim)}
    """
    base = (layer_config.get("layer_base") or "").strip()
    attn_name = (layer_config.get("attn_name") or "self_attn").strip()
    mlp_name = (layer_config.get("mlp_name") or "mlp").strip()
    outputs_sum: Dict[str, List[torch.Tensor]] = {k: [] for k in output_keys}
    outputs_count: Dict[str, List[torch.Tensor]] = {k: [] for k in output_keys}
    handles: List[Any] = []

    def make_forward_hook(key: str):
        def hook(_mod, _inp, outp):
            t = _extract_tensor(outp)
            if t is not None:
                t = t.detach()
                s, n = _agg_token_sum_count(t, token_location)
                outputs_sum[key].append(s.cpu())
                outputs_count[key].append(n.cpu())
        return hook

    def make_pre_hook(key: str):
        def hook(_mod, inp):
            t = _extract_input(inp)
            if t is not None:
                t = t.detach()
                s, n = _agg_token_sum_count(t, token_location)
                outputs_sum[key].append(s.cpu())
                outputs_count[key].append(n.cpu())
        return hook

    for i in range(num_layers):
        layer_prefix = f"{base}.{i}"
        layer_mod = _resolve_module(model, layer_prefix)
        attn_mod = _resolve_module(model, f"{layer_prefix}.{attn_name}")
        mlp_mod = _resolve_module(model, f"{layer_prefix}.{mlp_name}")
        key_attn_out = f"{layer_prefix}.attn_out"
        key_attn_block = f"{layer_prefix}.attn_block_out"
        key_mlp_out = f"{layer_prefix}.mlp_out"
        key_mlp_block = f"{layer_prefix}.mlp_block_out"
        if attn_mod is not None:
            handles.append(attn_mod.register_forward_hook(make_forward_hook(key_attn_out)))
        if mlp_mod is not None:
            handles.append(mlp_mod.register_forward_pre_hook(make_pre_hook(key_attn_block)))
        if mlp_mod is not None:
            handles.append(mlp_mod.register_forward_hook(make_forward_hook(key_mlp_out)))
        if layer_mod is not None:
            handles.append(layer_mod.register_forward_hook(make_forward_hook(key_mlp_block)))

    try:
        bs = max(int(batch_size) if batch_size else 8, 1)
        total_batches = (len(texts) + bs - 1) // bs
        with torch.inference_mode():
            for batch_idx, start in enumerate(range(0, len(texts), bs)):
                if progress_callback:
                    progress_callback(batch_idx + 1, total_batches)
                batch_texts = texts[start : start + bs]
                enc = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                _ = model(**enc)

        result: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name in output_keys:
            if not outputs_sum[name]:
                continue
            result[name] = (
                torch.cat(outputs_sum[name], dim=0),
                torch.cat(outputs_count[name], dim=0),
            )
        return result
    finally:
        for h in handles:
            h.remove()


def _apply_process(ds, code: str):
    """Apply processing code to dataset."""
    if not (code or code.strip()):
        return ds
    namespace = {"json": __import__("json"), "re": __import__("re")}
    exec(code.strip(), namespace)
    fn = namespace.get("process")
    if not callable(fn):
        raise ValueError("Code must define process(example) -> dict")
    return ds.map(fn, batched=False, remove_columns=None, desc="Processing")


def run_residual_concept_detection(
    *,
    model_key: str,
    ds,  # HuggingFace Dataset (single split)
    pipeline: Dict[str, Any],
    text_key: str,
    label_key: str,
    positive_label: str,
    negative_label: str,
    layer_config: Dict[str, str],
    token_location: Union[str, List[int]],
    batch_size: int,
    progress_callback=None,
) -> Tuple[Dict[str, Any], int]:
    """
    Run residual concept detection.
    ds: loaded HuggingFace Dataset (single split, after random select if any).
    Returns (result_dict, status_code).
    """
    try:
        from python.model_load import load_llm
    except ImportError as e:
        return ({"error": f"Import error: {e}"}, 500)

    if hasattr(ds, "keys"):
        ds = ds[list(ds.keys())[0]]
    if ds.num_rows == 0:
        return ({"error": "Dataset is empty"}, 400)

    # Apply processing if any
    code = (pipeline.get("processing_code") or "").strip()
    if code:
        try:
            ds = _apply_process(ds, code)
        except Exception as e:
            return ({"error": f"Processing failed: {e}"}, 400)

    # Collect texts and labels (everything as string)
    positive_label = str(positive_label).strip()
    negative_label = str(negative_label).strip()
    texts: List[str] = []
    labels: List[str] = []
    for i in range(ds.num_rows):
        row = ds[i]
        t = row.get(text_key)
        l = row.get(label_key)
        if t is None or l is None:
            continue
        t_str = str(t).strip() if t else ""
        l_str = str(l).strip() if l else ""
        if not t_str:
            continue
        if l_str in (positive_label, negative_label):
            texts.append(t_str)
            labels.append(l_str)

    if not texts:
        # 디버깅을 위해 label_key의 가능한 값들을 문자열 기준으로 같이 보여준다
        seen: set[str] = set()
        for i in range(min(ds.num_rows, 200)):
            lv = ds[i].get(label_key)
            if lv is None:
                continue
            seen.add(str(lv).strip())
        return (
            {
                "error": f"No rows with label in [{positive_label!r}, {negative_label!r}]",
                "label_key": label_key,
                "available_labels_str": sorted(seen),
            },
            400,
        )

    # Load model (do not move model; it may be offloaded to cpu/disk)
    tokenizer, model = load_llm(model_key)
    device = next(
        (p.device for p in model.parameters() if p.device.type != "meta"),
        torch.device("cpu"),
    )

    layer_base = (layer_config.get("layer_base") or "").strip()
    if not layer_base:
        return ({"error": "layer_base required in layer_config"}, 400)
    num_layers = _get_num_layers(model, layer_base)
    if num_layers <= 0:
        return ({"error": f"Cannot discover num_layers for layer_base={layer_base!r}"}, 400)
    output_keys = _build_hook_names_residual(layer_config, num_layers)
    if not output_keys:
        return ({"error": "Failed to build hook names from layer_config"}, 400)

    hidden_size = (
        getattr(model.config, "hidden_size", None)
        or getattr(model.config, "d_model", None)
        or getattr(getattr(model.config, "text_config", None), "hidden_size", None)
    )
    if hidden_size is None and hasattr(model, "model"):
        try:
            text_model = getattr(model.model, "language_model", model.model)
            emb = getattr(text_model, "embed_tokens", None)
            if emb is not None and hasattr(emb, "embedding_dim"):
                hidden_size = emb.embedding_dim
            elif emb is not None:
                for p in emb.parameters():
                    hidden_size = p.shape[-1]
                    break
        except Exception:
            pass
    if hidden_size is None:
        return ({"error": "Cannot determine model hidden size"}, 500)

    # Resolve token_location
    if token_location == "full":
        pass
    elif isinstance(token_location, str) and token_location.strip().lower() == "full":
        token_location = "full"
    else:
        try:
            if isinstance(token_location, list):
                token_location = [int(x) for x in token_location]
            else:
                parts = str(token_location).replace(",", " ").split()
                token_location = [int(x) for x in parts if x.strip()]
        except (ValueError, TypeError):
            token_location = "full"

    # Get outputs: attn=MLP input, mlp=block output per layer (sums, counts)
    layer_outputs = _get_layer_outputs_residual(
        model, tokenizer, texts, layer_config, num_layers, output_keys,
        token_location, device, batch_size,
        progress_callback=progress_callback,
    )
    if not layer_outputs:
        return ({"error": "No layer outputs captured. Check layer module names."}, 400)

    # Split by label and compute positive - negative mean (weighted by token counts)
    pos_mask = [l == positive_label for l in labels]
    neg_mask = [l == negative_label for l in labels]
    pos_indices = [i for i, m in enumerate(pos_mask) if m]
    neg_indices = [i for i, m in enumerate(neg_mask) if m]
    if not pos_indices or not neg_indices:
        return ({"error": f"Need both positive and negative examples. Got {len(pos_indices)} pos, {len(neg_indices)} neg"}, 400)

    pos_idx = torch.tensor(pos_indices, dtype=torch.long)
    neg_idx = torch.tensor(neg_indices, dtype=torch.long)

    directions: Dict[str, List[float]] = {}
    for layer_name, (sums, counts) in layer_outputs.items():
        sums_f = sums.float()
        counts_f = counts.float().clamp(min=1e-12)
        pos_sum = sums_f[pos_idx].sum(dim=0)
        pos_cnt = counts_f[pos_idx].sum()
        neg_sum = sums_f[neg_idx].sum(dim=0)
        neg_cnt = counts_f[neg_idx].sum()
        pos_mean = pos_sum / pos_cnt.clamp(min=1e-12)
        neg_mean = neg_sum / neg_cnt.clamp(min=1e-12)
        direction = (pos_mean - neg_mean).cpu().tolist()
        directions[layer_name] = direction

    num_keys = len(directions)
    model_dim = len(next(iter(directions.values()), [])) if directions else 0

    return (
        {
            "status": "ok",
            "directions": directions,
            "num_keys": num_keys,
            "model_dim": model_dim,
            "n_positive": len(pos_indices),
            "n_negative": len(neg_indices),
            "total_examples": len(texts),
            "batch_size": int(batch_size),
            "num_batches": (len(texts) + int(batch_size) - 1) // int(batch_size) if batch_size else len(texts),
            "layer_base": layer_config.get("layer_base"),
            "attn_name": layer_config.get("attn_name"),
            "mlp_name": layer_config.get("mlp_name"),
        },
        200,
    )
