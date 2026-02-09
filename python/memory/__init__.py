"""
Memory management: 3 types (Task Session, Task Result, Variables).
Define format and lifecycle per type; register entities to use them.
"""

from python.memory.types import MemoryType, MEMORY_TYPES
from python.memory.registry import register, get_store
from python.memory.cache_store import cache_store
from python.memory.task_result import task_result_store
from python.memory.variable import variable_store

__all__ = [
    "MemoryType",
    "MEMORY_TYPES",
    "register",
    "get_store",
    "cache_store",
    "task_result_store",
    "variable_store",
]
