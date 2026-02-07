"""
Circuit panel API: run prompt, get token sequence and layer-wise attention for visualization.
Modes: simple (forward only), generate (prompt → generate → forward), user_qa (user + generation → chat_template → forward).
"""

from flask import Blueprint, jsonify, request

from utils import circuit_run_generate, circuit_run_simple, circuit_run_user_qa

bp = Blueprint("circuit", __name__, url_prefix="/api")


@bp.post("/circuit_run")
def api_circuit_run():
    """Run circuit: mode=simple|generate|user_qa. All modes return tokens + layer attention + top next token."""
    data = request.get_json(force=True)
    model_key = data.get("model", "tiny-gpt2")
    mode = (data.get("mode") or "generate").strip().lower()
    max_new_tokens = data.get("max_new_tokens", 64)

    if mode == "simple":
        text = (data.get("prompt") or data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "prompt is required for simple mode"}), 400
        try:
            result = circuit_run_simple(model_key, text)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)
    if mode == "user_qa":
        user_prompt = (data.get("user_prompt") or "").strip()
        generation = (data.get("generation") or "").strip()
        if not user_prompt or not generation:
            return jsonify({"error": "user_prompt and generation are required for user_qa mode"}), 400
        try:
            result = circuit_run_user_qa(model_key, user_prompt, generation)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)
    # generate (default)
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required for generate mode"}), 400
    try:
        result = circuit_run_generate(model_key, prompt, max_new_tokens=max_new_tokens)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)
