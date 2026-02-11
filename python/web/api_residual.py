from flask import Blueprint, jsonify

from python.memory.variable import get_residual_variable, load_variable_pickle, save_residual_variable, save_variable_pickle, has_variable_pickle


residual_bp = Blueprint("residual", __name__)

# Note: residual-specific variable APIs moved into api_dataset.py for now.
# This blueprint is kept for future residual/XAI-2-only HTTP endpoints.


@residual_bp.get("/api/residual-vars/<path:var_id>")
def api_get_residual_var(var_id: str):
    """Get residual variable by id (directions dict)."""
    if has_variable_pickle(var_id):
        payload = load_variable_pickle(var_id)
        if isinstance(payload, dict) and payload.get("directions"):
            return jsonify(payload)
    rv = get_residual_variable(var_id)
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
        var_id, var_name = save_residual_variable(
            directions=directions,
            task_name=task_name,
            model=model,
            num_keys=int(num_keys),
            model_dim=int(model_dim or 0),
            additional_naming=additional_naming,
        )
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    payload = {
        "directions": directions,
        "task_name": task_name,
        "model": model,
        "num_keys": int(num_keys),
        "model_dim": int(model_dim or 0),
    }
    pickle_saved = save_variable_pickle(var_id, payload)
    return jsonify({"status": "ok", "variable_id": var_id, "variable_name": var_name, "pickle_saved": pickle_saved})
