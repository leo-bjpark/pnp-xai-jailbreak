from flask import Blueprint, jsonify, request

from python.model_load import get_model_status


model_bp = Blueprint("model", __name__)


def _detect_layer_structure(model):
    """
    Inspect model to detect layers_base, attn_name, mlp_name, o_proj_name, down_proj_name.
    Returns dict with detected values; if a name doesn't match default, value is None.
    Defaults: self_attn, mlp, o_proj, down_proj.
    """
    DEFAULT_ATTN = "self_attn"
    DEFAULT_MLP = "mlp"
    DEFAULT_O_PROJ = "o_proj"
    DEFAULT_DOWN_PROJ = "down_proj"

    result = {
        "layers_base": None,
        "attn_name": None,
        "mlp_name": None,
        "o_proj_name": None,
        "down_proj_name": None,
    }
    layer_names = []
    for name, _ in model.named_modules():
        if not name or "layers." not in name and ".layers." not in name:
            continue
        parts = name.split(".")
        if parts[-1].isdigit():
            layer_names.append(name)
    layer_names = sorted(set(layer_names), key=lambda x: (x.count("."), x))
    if not layer_names:
        return result
    first = layer_names[0]
    if "." in first:
        base, idx = first.rsplit(".", 1)
        if idx.isdigit():
            result["layers_base"] = base

    prefix = result["layers_base"] + ".0."
    attn_candidates = {DEFAULT_ATTN, "attention", "self_attention"}
    mlp_candidates = {DEFAULT_MLP}

    first_layer_children = set()
    for name, _ in model.named_modules():
        if not name or not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        if "." in rest:
            first_layer_children.add(rest.split(".")[0])
        else:
            first_layer_children.add(rest)

    for c in first_layer_children:
        if c in attn_candidates:
            result["attn_name"] = c
            break
    if result["attn_name"] is None and first_layer_children:
        for c in first_layer_children:
            if "attn" in c.lower():
                result["attn_name"] = c
                break

    for c in first_layer_children:
        if c in mlp_candidates:
            result["mlp_name"] = c
            break
    if result["mlp_name"] is None and first_layer_children:
        for c in first_layer_children:
            if "mlp" in c.lower():
                result["mlp_name"] = c
                break

    attn_prefix = prefix + (result["attn_name"] or DEFAULT_ATTN) + "."
    mlp_prefix = prefix + (result["mlp_name"] or DEFAULT_MLP) + "."
    for name, _ in model.named_modules():
        if not name:
            continue
        if name.startswith(attn_prefix) and name.endswith("." + DEFAULT_O_PROJ):
            result["o_proj_name"] = DEFAULT_O_PROJ
            break
        if name == attn_prefix + DEFAULT_O_PROJ:
            result["o_proj_name"] = DEFAULT_O_PROJ
            break
    if result["o_proj_name"] is None:
        for name, _ in model.named_modules():
            if attn_prefix in name and "o_proj" in name:
                result["o_proj_name"] = "o_proj"
                break

    for name, _ in model.named_modules():
        if not name:
            continue
        if name.startswith(mlp_prefix) and name.endswith("." + DEFAULT_DOWN_PROJ):
            result["down_proj_name"] = DEFAULT_DOWN_PROJ
            break
        if name == mlp_prefix + DEFAULT_DOWN_PROJ:
            result["down_proj_name"] = DEFAULT_DOWN_PROJ
            break

    return result


def _empty_layer_structure():
    """Empty layer structure when detection fails."""
    return {
        "layers_base": None,
        "attn_name": None,
        "mlp_name": None,
        "o_proj_name": None,
        "down_proj_name": None,
    }


@model_bp.get("/api/model_layer_names")
def api_model_layer_names():
    """Return layer structure for the given model (layers_base, attn, mlp, o_proj, down_proj)."""
    model_key = request.args.get("model", "").strip()
    if not model_key:
        return jsonify({"error": "model query param required"}), 400
    try:
        from python.model_load import load_llm

        _, model = load_llm(model_key)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500
    try:
        names = []
        for name, _ in model.named_modules():
            if not name:
                continue
            if "layers." in name or ".layers." in name:
                parts = name.split(".")
                if parts[-1].isdigit():
                    names.append(name)
        names = sorted(set(names), key=lambda x: (x.count("."), x))
        layers_base = ""
        if names:
            first = names[0]
            if "." in first:
                base, idx = first.rsplit(".", 1)
                if idx.isdigit():
                    layers_base = base
        structure = _detect_layer_structure(model)
        if layers_base and not structure.get("layers_base"):
            structure["layers_base"] = layers_base
        return jsonify(
            {
                "layer_names": names,
                "layers_base": layers_base,
                "layer_structure": structure,
            }
        )
    except Exception:  # noqa: BLE001
        return jsonify(
            {
                "layer_names": [],
                "layers_base": "",
                "layer_structure": _empty_layer_structure(),
            }
        )


@model_bp.get("/api/model_status")
def api_model_status():
    model_key = request.args.get("model")
    if not model_key:
        return jsonify({"error": "model query param required"}), 400
    try:
        status = get_model_status(model_key)
        return jsonify(status)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400

