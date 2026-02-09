"""
Entity -> Memory Type 매핑.
새 entity (task_panel, object 등) 추가 시 register()로 등록.
"""

from typing import Any, Dict, Optional

from python.memory.types import MemoryType
from python.memory.cache_store import cache_store
from python.memory.task_result import task_result_store
from python.memory.variable import variable_store

# entity_name -> MemoryType
_ENTITY_REGISTRY: Dict[str, MemoryType] = {}

# MemoryType -> store instance
_STORES = {
    MemoryType.TASK_SESSION: cache_store,
    MemoryType.TASK_RESULT: task_result_store,
    MemoryType.VARIABLE: variable_store,
}


def register(entity_name: str, memory_type: MemoryType) -> None:
    """Entity를 메모리 유형에 등록."""
    _ENTITY_REGISTRY[entity_name] = memory_type


def get_store(entity_name: str) -> Optional[Any]:
    """Entity에 해당하는 store 반환. 미등록 시 None."""
    mt = _ENTITY_REGISTRY.get(entity_name)
    return _STORES.get(mt) if mt else None


def get_store_by_type(memory_type: MemoryType) -> Any:
    """메모리 유형에 해당하는 store 반환."""
    return _STORES[memory_type]
