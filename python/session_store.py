"""
Session and task data persistence for PnPXAI Tool.
- Stores created tasks by XAI level
- Restores analysis results when loading a task
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TASKS_FILE = DATA_DIR / "tasks.json"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_tasks() -> Dict[str, Any]:
    _ensure_data_dir()
    if not TASKS_FILE.exists():
        return {"xai_level_0": [], "xai_level_1": []}
    try:
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"xai_level_0": [], "xai_level_1": []}


def _save_tasks(data: Dict[str, Any]) -> None:
    _ensure_data_dir()
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _level_to_key(xai_level: str) -> str:
    """Convert level like 0.1, 1.1 to storage key xai_level_0_1, xai_level_1_1."""
    normalized = str(xai_level).replace(".", "_")
    return f"xai_level_{normalized}" if not normalized.startswith("xai_level_") else normalized


def get_tasks(xai_level_names: Optional[Dict[str, str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Return all tasks grouped by XAI level. Uses xai_level_names for structure if provided."""
    data = _load_tasks()
    if xai_level_names:
        result = {}
        for level in sorted(xai_level_names.keys(), key=lambda x: [int(p) for p in str(x).split(".")]):
            key = _level_to_key(level)
            result[key] = data.get(key, [])
        return result
    return data


def add_task(xai_level: str, title: str, model: str, treatment: str, result: Dict[str, Any]) -> str:
    """
    Add a new task. Returns task_id. xai_level e.g. "0.1", "1.1", "1.2".
    """
    data = _load_tasks()
    key = _level_to_key(xai_level)
    if key not in data:
        data[key] = []

    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(data[key])}"
    task = {
        "id": task_id,
        "title": title,
        "model": model,
        "treatment": treatment,
        "result": result,
        "created_at": datetime.now().isoformat(),
        "xai_level": xai_level,
    }
    data[key].append(task)
    _save_tasks(data)
    return task_id


def get_task_by_id(task_id: str) -> Optional[Dict[str, Any]]:
    """Get a single task by ID."""
    data = _load_tasks()
    for level_tasks in data.values():
        if isinstance(level_tasks, list):
            for t in level_tasks:
                if t.get("id") == task_id:
                    return t
    return None


def update_task_title(task_id: str, title: str) -> bool:
    """Update task title."""
    data = _load_tasks()
    for key in list(data.keys()):
        tasks = data.get(key, [])
        if isinstance(tasks, list):
            for t in tasks:
                if t.get("id") == task_id:
                    t["title"] = title
                    _save_tasks(data)
                    return True
    return False


def update_task_result(task_id: str, result: Dict[str, Any], model: str = None, treatment: str = None) -> bool:
    """Update task result (and optionally model, treatment)."""
    data = _load_tasks()
    for key in list(data.keys()):
        tasks = data.get(key, [])
        if isinstance(tasks, list):
            for t in tasks:
                if t.get("id") == task_id:
                    t["result"] = result
                    if model is not None:
                        t["model"] = model
                    if treatment is not None:
                        t["treatment"] = treatment
                    _save_tasks(data)
                    return True
    return False


def delete_task(task_id: str) -> bool:
    """Delete a task by ID."""
    data = _load_tasks()
    for key in list(data.keys()):
        tasks = data.get(key, [])
        if isinstance(tasks, list):
            for i, t in enumerate(tasks):
                if t.get("id") == task_id:
                    data[key].pop(i)
                    _save_tasks(data)
                    return True
    return False


def get_raw_memory() -> Dict[str, Any]:
    """Return full memory state as JSON-serializable dict for export."""
    data = _load_tasks()
    from datetime import datetime
    return {
        "version": "0.1",
        "format": "PnP-XAI-Memory",
        "exported_at": datetime.now().isoformat(),
        "tasks": data,
    }


def import_memory(data: Dict[str, Any]) -> bool:
    """Import from exported JSON. Returns True on success."""
    tasks = data.get("tasks")
    if not isinstance(tasks, dict):
        return False
    _save_tasks(tasks)
    return True


def clear_all_tasks(xai_level_names: Optional[Dict[str, str]] = None) -> None:
    """Delete all tasks. Keeps structure from xai_level_names if provided."""
    if xai_level_names:
        data = {}
        for level in sorted(xai_level_names.keys(), key=lambda x: [int(p) for p in str(x).split(".")]):
            key = _level_to_key(level)
            data[key] = []
    else:
        data = {"xai_level_0": [], "xai_level_1": []}
    _save_tasks(data)
