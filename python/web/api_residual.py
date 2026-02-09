from flask import Blueprint, jsonify

from python.memory.variable import get_residual_variable, save_residual_variable


residual_bp = Blueprint("residual", __name__)

# Note: residual-specific variable APIs moved into api_dataset.py for now.
# This blueprint is kept for future residual/XAI-2-only HTTP endpoints.


@residual_bp.get("/api/residual-vars/<path:var_name>")
def api_get_residual_var(var_name: str):
    """Get residual variable by name (directions dict)."""
    rv = get_residual_variable(var_name)
    if not rv:
        return jsonify({"error": "Variable not found"}), 404
    return jsonify(rv)


@residual_bp.post("/api/residual-vars/save")
def api_save_residual_var():
    """
    Save residual direction vectors to variable.
    Body: { directions, task_name, model, num_keys, model_dim, additional_naming? }.
    """
    from flask import request

    data = request.get_json(force=True) or {}
    directions = data.get("directions")
    if not directions or not isinstance(directions, dict):
        return jsonify({"error": "directions (dict) required"}), 400
    task_name = (data.get("task_name") or "").strip() or "Residual"
    model = (data.get("model") or "").strip() or "-"
    num_keys = data.get("num_keys")
    model_dim = data.get("model_dim")
    if num_keys is None:
        num_keys = len(directions)
    if model_dim is None and directions:
        first = next(iter(directions.values()), [])
        model_dim = len(first) if isinstance(first, (list, tuple)) else 0
    additional_naming = (data.get("additional_naming") or "").strip() or None
    try:
        var_name = save_residual_variable(
            directions=directions,
            task_name=task_name,
            model=model,
            num_keys=int(num_keys),
            model_dim=int(model_dim or 0),
            additional_naming=additional_naming,
        )
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "ok", "variable_name": var_name})

