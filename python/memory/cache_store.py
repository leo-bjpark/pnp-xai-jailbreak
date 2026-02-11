"""
Cache store: Python Cache (Alive & Terminated).

- session: loaded_model, treatment (in-memory only; not persisted).
- generic namespace cache: put/get/delete (공통 캐시, namespace로 구분)
- 사용처에서 반드시 namespace를 지정하여 위치 등록 후 사용
- Session is cleared when user leaves (switches to another task panel). Not restored on restart.
"""

from typing import Any, Dict, List, Optional


class CacheStore:
    """
    공통 세션 캐시. 모든 테스크가 동일하게 사용.
    Lifecycle: alive (in cache) / terminated (cleared).
    Session is in-memory only; not persisted to disk.
    """

    def __init__(self) -> None:
        self._session: Dict[str, Any] = {
            "loaded_model": None,
            "treatment": None,
        }
        self._cache: Dict[str, Dict[str, Any]] = {}  # namespace -> {key -> value}

    # ----- Session (loaded_model, treatment) -----

    def get_session(self) -> Dict[str, Any]:
        return {"loaded_model": self._session["loaded_model"], "treatment": self._session["treatment"]}

    def set_session(self, loaded_model: Optional[str], treatment: str = "") -> None:
        self._session["loaded_model"] = loaded_model
        self._session["treatment"] = treatment

    # ----- Generic namespace cache -----

    def put(self, namespace: str, key: str, value: Any) -> None:
        """namespace에 key-value 저장."""
        if namespace not in self._cache:
            self._cache[namespace] = {}
        self._cache[namespace][key] = value

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """namespace에서 key 조회."""
        return self._cache.get(namespace, {}).get(key)

    def delete(self, namespace: str, key: str) -> bool:
        """namespace에서 key 삭제."""
        if namespace in self._cache and key in self._cache[namespace]:
            del self._cache[namespace][key]
            return True
        return False

    def clear_namespace(self, namespace: str) -> None:
        """namespace 전체 삭제."""
        if namespace in self._cache:
            self._cache[namespace].clear()

    def list_namespaces(self) -> List[str]:
        """등록된 namespace 목록."""
        return list(self._cache.keys())

    def list_session_caches(self) -> List[Dict[str, Any]]:
        """Session namespace의 캐시 목록. Key = Task|model|treatment|이름."""
        ns = self._cache.get("session", {})
        items = []
        for key, meta in ns.items():
            parts = (meta or {}).get("key_parts", []) if isinstance(meta, dict) else []
            if not parts and isinstance(key, str):
                parts = key.split("|")
            items.append({
                "key": key,
                "task": parts[0] if len(parts) > 0 else "",
                "model": parts[1] if len(parts) > 1 else "",
                "treatment": parts[2] if len(parts) > 2 else "",
                "name": parts[3] if len(parts) > 3 else "",
            })
        return items

    def get_namespace(self, namespace: str) -> Dict[str, Any]:
        """Return the entire namespace map (live reference)."""
        return self._cache.get(namespace, {})

    def terminate_all(self) -> None:
        """전체 초기화: session + 모든 namespace cache."""
        self._session["loaded_model"] = None
        self._session["treatment"] = None
        self._cache.clear()


cache_store = CacheStore()
