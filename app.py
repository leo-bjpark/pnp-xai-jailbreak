"""
PnP-XAI-LLM - VSCode-like XAI analysis tool.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on path for backup.utils
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, render_template, request

# Model list and loading via python.model_load (reads config.yaml, uses backup.utils internally)
from python.model_load import get_config_models, load_model, get_model_status

from python.config_loader import get_xai_level_names, get_xai_level_names_grouped
from python.session_store import (
    add_task,
    get_task_by_id,
    get_tasks,
    update_task_title,
    update_task_result,
    delete_task,
    get_raw_memory,
    import_memory,
)

app = Flask(__name__)

# In-memory session: current Loaded Model + Treatment
# Used for memory-efficient management and run confirmation
SESSION_STATE = {
    "loaded_model": None,
    "treatment": None,
}


@app.get("/panel")
def panel():
    """Standalone right panel window (opens in separate window, independent of main page)."""
    return render_template("panel.html")


@app.get("/")
def index():
    """Main IDE-like interface."""
    models = get_config_models()
    xai_level_names = get_xai_level_names()
    xai_level_grouped = get_xai_level_names_grouped()
    tasks = get_tasks(xai_level_names)
    return render_template(
        "index.html",
        models=models,
        tasks=tasks,
        xai_level_names=xai_level_names,
        xai_level_grouped=xai_level_grouped,
    )


@app.get("/task/<task_id>")
def task_view(task_id):
    """Generic task view - fetches task and renders by xai_level."""
    task = get_task_by_id(task_id)
    if not task:
        models = get_config_models()
        xai_level_names = get_xai_level_names()
        return render_template("index.html", models=models, tasks=get_tasks(xai_level_names), xai_level_names=xai_level_names, xai_level_grouped=get_xai_level_names_grouped(), error="Task not found")
    level_key = task.get("xai_level", "0.1")
    return _render_task(task_id, level_key)




def _task_template(level_key: str) -> str:
    """Template name for task view by XAI level."""
    if level_key == "0.1.1":
        return "XAI_0_1_1_completion.html"
    return "xai_task.html"


def _render_task(task_id: str, level_key: str):
    task = get_task_by_id(task_id)
    if not task:
        models = get_config_models()
        xai_level_names = get_xai_level_names()
        return render_template("index.html", models=models, tasks=get_tasks(xai_level_names), xai_level_names=xai_level_names, xai_level_grouped=get_xai_level_names_grouped(), error="Task not found")
    models = get_config_models()
    xai_level_names = get_xai_level_names()
    tasks = get_tasks(xai_level_names)
    template = _task_template(level_key)
    return render_template(template, task=task, models=models, tasks=tasks, xai_level_names=xai_level_names, xai_level_grouped=get_xai_level_names_grouped())


# ----- API: Tasks -----

@app.get("/api/tasks")
def api_get_tasks():
    xai_level_names = get_xai_level_names()
    return jsonify({"tasks": get_tasks(xai_level_names), "xai_level_names": xai_level_names})


@app.get("/api/tasks/<task_id>")
def api_get_task(task_id):
    task = get_task_by_id(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task)


@app.post("/api/tasks")
def api_create_task():
    data = request.get_json(force=True) or {}
    xai_level = data.get("xai_level", "0.1")
    title = data.get("title", "Untitled Task")
    model = data.get("model", "")
    treatment = data.get("treatment", "")
    result = data.get("result", {})

    task_id = add_task(xai_level, title, model, treatment, result)
    return jsonify({"task_id": task_id, "status": "ok"})


@app.patch("/api/tasks/<task_id>")
def api_update_task(task_id):
    data = request.get_json(force=True) or {}
    title = data.get("title")
    result = data.get("result")
    model = data.get("model")
    treatment = data.get("treatment")
    if title is not None:
        if update_task_title(task_id, title):
            return jsonify({"status": "ok"})
    if result is not None:
        if update_task_result(task_id, result, model=model, treatment=treatment):
            return jsonify({"status": "ok"})
    return jsonify({"error": "Update failed"}), 400


@app.delete("/api/tasks/<task_id>")
def api_delete_task(task_id):
    if delete_task(task_id):
        return jsonify({"status": "ok"})
    return jsonify({"error": "Task not found"}), 404


# ----- API: Session (Loaded Model + Treatment) -----

@app.get("/api/session")
def api_get_session():
    """Return current session state: loaded_model, treatment."""
    return jsonify({
        "loaded_model": SESSION_STATE["loaded_model"],
        "treatment": SESSION_STATE["treatment"],
    })


@app.post("/api/session")
def api_set_session():
    """Set session (after user confirms model load)."""
    data = request.get_json(force=True) or {}
    model = data.get("loaded_model")
    treatment = data.get("treatment", "")
    SESSION_STATE["loaded_model"] = model
    SESSION_STATE["treatment"] = treatment
    return jsonify({"status": "ok"})


# ----- API: Models -----

@app.get("/api/models")
def api_models():
    return jsonify({"models": get_config_models()})


@app.post("/api/load_model")
def api_load_model():
    """Load model and update session."""
    data = request.get_json(force=True) or {}
    model_key = data.get("model", "")
    treatment = data.get("treatment", SESSION_STATE.get("treatment") or "")
    if not model_key:
        return jsonify({"error": "model required"}), 400
    try:
        load_model(model_key)
        SESSION_STATE["loaded_model"] = model_key
        SESSION_STATE["treatment"] = treatment
        return jsonify({"status": "ok", "model": model_key})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/model_status")
def api_model_status():
    model_key = request.args.get("model")
    if not model_key:
        return jsonify({"error": "model query param required"}), 400
    try:
        status = get_model_status(model_key)
        return jsonify(status)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ----- API: Run (placeholder - extend for actual XAI computation) -----

# ----- API: Memory Export/Import -----

@app.get("/api/memory/export")
def api_memory_export():
    """Export full memory as JSON (for download)."""
    from flask import Response
    data = get_raw_memory()
    data["session"] = SESSION_STATE.copy()
    # Filename: XAI_Level_Export_{timestamp}.json
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Response(
        json.dumps(data, indent=2, ensure_ascii=False),
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment; filename=XAI_Level_Export_{ts}.json"},
    )


@app.post("/api/memory/import")
def api_memory_import():
    """Import memory from JSON. Expects JSON in request body."""
    data = request.get_json(force=True) or {}
    if not import_memory(data):
        return jsonify({"error": "Invalid format. Expected { tasks: {...} }"}), 400
    # Optionally restore session from imported data
    sess = data.get("session")
    if isinstance(sess, dict):
        SESSION_STATE["loaded_model"] = sess.get("loaded_model")
        SESSION_STATE["treatment"] = sess.get("treatment")
    return jsonify({"status": "ok"})


# ----- API: Run -----

@app.post("/api/run")
def api_run():
    """
    Execute XAI analysis.
    Client should ensure session matches input (model + treatment) before calling,
    or handle the confirm dialog when mismatch.
    For level 0.1.1 (Completion), runs text_completion with input_setting params.
    """
    data = request.get_json(force=True) or {}
    model = data.get("model", "")
    treatment = data.get("treatment", "")
    input_setting = data.get("input_setting", {})

    # Check session consistency
    current_model = SESSION_STATE.get("loaded_model")
    current_treatment = SESSION_STATE.get("treatment")

    if model != current_model or treatment != current_treatment:
        return jsonify({
            "error": "session_mismatch",
            "message": "Loaded Model + Treatment does not match the current session. Load the model with this setting?",
            "requested": {"model": model, "treatment": treatment},
            "current": {"model": current_model, "treatment": current_treatment},
        }), 400

    # Completion (0.1.1): input_string + generation params (input_string can be empty)
    input_string = (input_setting.get("input_string") or "").strip()
    if "input_string" in input_setting and current_model:
        try:
            temperature = float(input_setting.get("temperature", 0.7))
            max_new_tokens = int(input_setting.get("max_new_tokens", 256))
            top_p = float(input_setting.get("top_p", 1.0))
            top_k = int(input_setting.get("top_k", 50))
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid input_setting: temperature, max_new_tokens, top_p, top_k must be numbers"}), 400
        try:
            from backup.utils import text_completion
            generated_text = text_completion(
                model_key=current_model,
                prompt=input_string,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
            )
            result = {
                "status": "ok",
                "model": model,
                "treatment": treatment,
                "input_string": input_string,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "generated_text": generated_text,
            }
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Fallback: placeholder for other levels
    result = {
        "status": "ok",
        "message": "Analysis complete. (Actual XAI computation to be implemented)",
        "model": model,
        "treatment": treatment,
        "input_setting": input_setting,
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
