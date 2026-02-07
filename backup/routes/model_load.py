"""
Model load related API: CUDA devices, model status, load, chat, cache.
"""

import os
import uuid

from flask import Blueprint, jsonify, request

from utils import (
    chat_completion,
    get_cache_token_count,
    get_config_models,
    get_model_status,
    load_llm,
)

bp = Blueprint("model_load", __name__, url_prefix="/api")

# Conversation cache for efficient multi-turn: conversation_id -> { "model_key", "messages" }
CONVERSATION_CACHE = {}


@bp.get("/models")
def api_models():
    return jsonify({"models": get_config_models()})


@bp.get("/cuda_devices")
def api_cuda_devices():
    return jsonify({"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "0")})


@bp.post("/set_cuda_devices")
def api_set_cuda_devices():
    data = request.get_json(force=True)
    devices = (data.get("devices") or "0").strip()
    if not devices:
        devices = "0"
    if not all(c.isdigit() or c in ", " for c in devices):
        return jsonify({"error": "Invalid format. Use e.g. 0,1,2"}), 400
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    return jsonify({"status": "ok", "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]})


@bp.get("/model_status")
def api_model_status():
    model_key = request.args.get("model")
    if not model_key:
        return jsonify({"error": "model query param required"}), 400
    try:
        status = get_model_status(model_key)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400
    return jsonify(status)


@bp.post("/load_model")
def api_load_model():
    data = request.get_json(force=True)
    model_key = data.get("model", "tiny-gpt2")
    try:
        load_llm(model_key)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "ok", "model": model_key})


@bp.post("/chat")
def api_chat():
    """Multi-round chat with optional server-side conversation cache."""
    data = request.get_json(force=True)
    model_key = data.get("model", "tiny-gpt2")
    max_new_tokens = data.get("max_new_tokens", 512)
    conversation_id = data.get("conversation_id")
    content = (data.get("content") or "").strip()
    messages = data.get("messages", [])

    if conversation_id and content:
        cached = CONVERSATION_CACHE.get(conversation_id)
        if not cached or cached["model_key"] != model_key:
            return jsonify({"error": "Cache miss or model changed. Send full messages or clear cache."}), 400
        messages = list(cached["messages"])
        messages.append({"role": "user", "content": content})
    elif not isinstance(messages, list) or len(messages) == 0:
        return jsonify({"error": "messages must be a non-empty list, or provide conversation_id and content"}), 400

    try:
        cache_token_count = get_cache_token_count(model_key, messages)
    except Exception:  # noqa: BLE001
        cache_token_count = None

    try:
        reply = chat_completion(
            model_key,
            messages,
            max_new_tokens=max_new_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400

    messages.append({"role": "assistant", "content": reply})
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    CONVERSATION_CACHE[conversation_id] = {"model_key": model_key, "messages": messages}
    payload = {
        "message": {"role": "assistant", "content": reply},
        "conversation_id": conversation_id,
    }
    if cache_token_count is not None:
        payload["cache_token_count"] = cache_token_count
    return jsonify(payload)


@bp.post("/chat_clear_cache")
def api_chat_clear_cache():
    """Clear server-side conversation cache for the given conversation_id."""
    data = request.get_json(force=True)
    conversation_id = data.get("conversation_id")
    if conversation_id and conversation_id in CONVERSATION_CACHE:
        del CONVERSATION_CACHE[conversation_id]
    return jsonify({"status": "ok"})
