"""
Task Result store: Json Format (Save & Load & Delete).

- tasks.json 기반
- Lifecycle: save, load, delete
"""

from typing import Any, Dict, List, Optional

from python.session_store import (
    add_task as _add_task,
    get_task_by_id as _get_task_by_id,
    get_tasks as _get_tasks,
    update_task_title as _update_task_title,
    update_task_result as _update_task_result,
    delete_task as _delete_task,
    get_raw_memory as _get_raw_memory,
    import_memory as _import_memory,
    clear_all_tasks as _clear_all_tasks,
)


class TaskResultStore:
    """
    Task result persistence in JSON format.
    Lifecycle: save, load, delete.
    """

    def add(
        self,
        xai_level: str,
        title: str,
        model: str,
        treatment: str,
        result: Dict[str, Any],
    ) -> str:
        return _add_task(xai_level, title, model, treatment, result)

    def get_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        return _get_task_by_id(task_id)

    def list_all(self, xai_level_names: Optional[Dict[str, str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        return _get_tasks(xai_level_names)

    def update_title(self, task_id: str, title: str) -> bool:
        return _update_task_title(task_id, title)

    def update_result(
        self,
        task_id: str,
        result: Dict[str, Any],
        model: Optional[str] = None,
        treatment: Optional[str] = None,
    ) -> bool:
        return _update_task_result(task_id, result, model=model, treatment=treatment)

    def delete(self, task_id: str) -> bool:
        return _delete_task(task_id)

    def export_raw(self) -> Dict[str, Any]:
        return _get_raw_memory()

    def import_raw(self, data: Dict[str, Any]) -> bool:
        return _import_memory(data)

    def clear_all(self, xai_level_names: Optional[Dict[str, str]] = None) -> None:
        _clear_all_tasks(xai_level_names)


task_result_store = TaskResultStore()
