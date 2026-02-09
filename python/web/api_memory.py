from flask import Blueprint, jsonify

from python.config_loader import get_xai_level_names
from python.memory import cache_store, task_result_store, variable_store
from python.memory.variable import clear_all as wm_clear_all
from python.model_load import clear_model_cache, get_model_status
from python.session_store import TASKS_FILE


memory_bp = Blueprint("memory", __name__)


@memory_bp.get("/api/memory/summary")
def api_memory_summary():
    """Return Session | Result | Variable memory usage in GB."""
    session_gb = 0.0
    result_gb = 0.0
    variable_gb = 0.0

    sess = cache_store.get_session()
    loaded_model = sess.get("loaded_model")
    session_caches = cache_store.list_session_caches()
    if loaded_model and session_caches:
        try:
            status = get_model_status(loaded_model)
            for dev in (status.get("device_status") or []):
                gb = float(dev.get("memory_gb") or 0)
                session_gb += gb
        except Exception:  # noqa: BLE001
            pass

    if TASKS_FILE.exists():
        result_gb = TASKS_FILE.stat().st_size / (1024**3)

    var_summary = variable_store.summarize_for_panel(loaded_model_key=loaded_model)
    if var_summary.get("loaded_model"):
        lm = var_summary["loaded_model"]
        g = lm.get("memory_gpu_gb")
        r = lm.get("memory_ram_gb")
        if g is not None:
            variable_gb += float(g)
        if r is not None:
            variable_gb += float(r)
    for v in var_summary.get("variables", []):
        mb = v.get("memory_ram_mb")
        if mb is not None:
            variable_gb += float(mb) / 1024

    return jsonify(
        {
            "session_gb": round(session_gb, 3),
            "result_gb": round(result_gb, 3),
            "variable_gb": round(variable_gb, 3),
        }
    )


@memory_bp.get("/api/memory/session/list")
def api_list_session_caches():
    """List Session caches. Key = Task | model | treatment | 이름."""
    items = cache_store.list_session_caches()
    return jsonify({"caches": items})


@memory_bp.post("/api/memory/session/register")
def api_register_session_cache():
    """Register a cache entry. Key = Task | model | treatment | 이름."""
    from flask import request

    SESSION_CACHE_NAMESPACE = "session"

    data = request.get_json(force=True) or {}
    task_id = (data.get("task_id") or "").strip()
    model = (data.get("model") or "").strip()
    treatment = (data.get("treatment") or "").strip()
    name = (data.get("name") or "").strip()
    if not task_id:
        return jsonify({"error": "task_id required"}), 400
    key = f"{task_id}|{model}|{treatment}|{name}"
    cache_store.put(SESSION_CACHE_NAMESPACE, key, {"key_parts": [task_id, model, treatment, name]})
    return jsonify({"status": "ok", "key": key})


@memory_bp.delete("/api/memory/session/unregister/<path:key>")
def api_unregister_session_cache(key: str):
    """Unregister one cache entry."""
    SESSION_CACHE_NAMESPACE = "session"
    cache_store.delete(SESSION_CACHE_NAMESPACE, key)
    return jsonify({"status": "ok"})


@memory_bp.post("/api/memory/session/clear")
def api_clear_session():
    """Clear all Session caches (loaded model, treatment, all Python caches)."""
    cache_store.terminate_all()
    clear_model_cache()
    return jsonify({"status": "ok"})


@memory_bp.post("/api/memory/result/clear")
def api_clear_result():
    """Clear all Task Results (tasks.json)."""
    xai_level_names = get_xai_level_names()
    task_result_store.clear_all(xai_level_names)
    return jsonify({"status": "ok"})


@memory_bp.post("/api/memory/variable/clear")
def api_clear_variable():
    """Clear all Variables (working memory)."""
    variable_store.clear_all()
    return jsonify({"status": "ok"})


@memory_bp.post("/api/empty_cache")
def api_empty_cache():
    """
    Reset: clear loaded model, session state, conversation cache,
    CUDA cache, and working‑memory variables.

    Called after user confirms Empty Cache (warning shown in UI).
    """
    cache_store.terminate_all()
    clear_model_cache()
    wm_clear_all()
    return jsonify({"status": "ok"})

