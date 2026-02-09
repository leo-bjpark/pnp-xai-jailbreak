"""
3가지 메모리 유형 정의.

| 유형         | 포맷          | 수명주기                     |
|-------------|---------------|------------------------------|
| task_session| Python Cache  | Alive & Terminated           |
| task_result | Json Format   | Save & Load & Delete         |
| variable    | Python Address| Save & Load & Delete         |
"""

from enum import Enum
from typing import Dict, List, Any


class MemoryType(str, Enum):
    TASK_SESSION = "task_session"
    TASK_RESULT = "task_result"
    VARIABLE = "variable"


MEMORY_TYPES: Dict[str, Dict[str, Any]] = {
    MemoryType.TASK_SESSION: {
        "format": "cache",
        "description": "Python Cache",
        "lifecycle": ["alive", "terminated"],
    },
    MemoryType.TASK_RESULT: {
        "format": "json",
        "description": "Json Format",
        "lifecycle": ["save", "load", "delete"],
    },
    MemoryType.VARIABLE: {
        "format": "address",
        "description": "Python Address",
        "lifecycle": ["save", "load", "delete"],
    },
}
