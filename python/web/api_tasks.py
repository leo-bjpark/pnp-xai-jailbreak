from flask import Blueprint, jsonify, request

from python.config_loader import get_xai_level_names
from python.session_store import (
    add_task,
    delete_task,
    get_task_by_id,
    get_tasks,
    update_task_result,
    update_task_title,
)


tasks_bp = Blueprint("tasks", __name__)


@tasks_bp.get("/api/tasks")
def api_get_tasks():
    xai_level_names = get_xai_level_names()
    return jsonify({"tasks": get_tasks(xai_level_names), "xai_level_names": xai_level_names})


@tasks_bp.get("/api/tasks/<task_id>")
def api_get_task(task_id):
    task = get_task_by_id(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task)


@tasks_bp.post("/api/tasks")
def api_create_task():
    data = request.get_json(force=True) or {}
    xai_level = data.get("xai_level", "0.1")
    title = data.get("title", "Untitled Task")
    model = data.get("model", "")
    treatment = data.get("treatment", "")
    result = data.get("result", {})

    task_id = add_task(xai_level, title, model, treatment, result)
    return jsonify({"task_id": task_id, "status": "ok"})


@tasks_bp.patch("/api/tasks/<task_id>")
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


@tasks_bp.delete("/api/tasks/<task_id>")
def api_delete_task(task_id):
    if delete_task(task_id):
        return jsonify({"status": "ok"})
    return jsonify({"error": "Task not found"}), 404

