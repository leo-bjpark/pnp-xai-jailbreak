"""
Variable store: Python Address (Save & Load & Delete).

- Variables panel: dataset pipeline variables, residual direction variables, loaded model status
- Lifecycle: save, load, delete
- NOTE: Heavy objects (HF datasets, tensors) are not stored here; we keep
  lightweight metadata and rough memory estimates for UI display.
  Residual directions dict is persisted to data/residual_variables.json.
"""

from __future__ import annotations

import json
import pickle
import random
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from python.model_load import get_model_status

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RESIDUAL_VARS_FILE = _DATA_DIR / "residual_variables.json"
DATA_VARS_FILE = _DATA_DIR / "data_variables.json"
PICKLE_DIR = _DATA_DIR / "variable_pickles"


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", (name or "").strip())
    return safe.strip("._") or "variable"


@dataclass
class DataVarMeta:
    """
    Metadata for a single working-memory entry from a dataset pipeline.
    Lightweight and JSON-serializable for Flask handlers.
    """

    uid: str
    nickname: str
    pipeline_id: str
    task_name: str
    data_name: str
    split: Optional[str]
    random_n: Optional[int]
    seed: Optional[int]
    dataset_info: Optional[Dict[str, Any]]
    processed_dataset_info: Optional[Dict[str, Any]]
    hf_load_options: Dict[str, Any]
    created_at: str
    type: str = "data"


class VariableStore:
    """
    Variables (dataset pipeline variables, etc.).
    Lifecycle: save, load, delete.
    """

    def __init__(self) -> None:
        self._data_vars: Dict[str, DataVarMeta] = {}
        self._residual_vars: Dict[str, Dict[str, Any]] = {}
        self._load_residual_vars()
        self._load_data_vars()

    def _generate_uid(self, prefix: str = "var") -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for _ in range(20):
            uid = f"{prefix}_{stamp}_{random.randint(100000, 999999)}"
            if uid not in self._data_vars and uid not in self._residual_vars:
                return uid
        return f"{prefix}_{stamp}_{random.randint(100000, 999999)}"

    def _migrate_pickle(self, legacy_name: str, uid: str) -> None:
        old_path = PICKLE_DIR / f"{_safe_name(legacy_name)}.pkl"
        new_path = PICKLE_DIR / f"{_safe_name(uid)}.pkl"
        if old_path.exists() and not new_path.exists():
            try:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                old_path.rename(new_path)
            except OSError:
                pass

    def _load_data_vars(self) -> None:
        """Load data variables from disk."""
        if not DATA_VARS_FILE.exists():
            return
        migrated = False
        try:
            with open(DATA_VARS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for key, d in (raw or {}).items():
                if not isinstance(d, dict):
                    continue
                if d.get("type") not in (None, "data"):
                    continue
                uid = (d.get("uid") or "").strip()
                nickname = (d.get("nickname") or d.get("variable_name") or d.get("name") or key or "").strip()
                legacy_name = ""
                if not uid:
                    uid = self._generate_uid("var")
                    migrated = True
                    legacy_name = (d.get("variable_name") or key or "").strip()
                meta = DataVarMeta(
                    uid=uid,
                    nickname=nickname or uid,
                    pipeline_id=d.get("pipeline_id", ""),
                    task_name=d.get("task_name", ""),
                    data_name=d.get("data_name", ""),
                    split=d.get("split"),
                    random_n=d.get("random_n"),
                    seed=d.get("seed"),
                    dataset_info=d.get("dataset_info"),
                    processed_dataset_info=d.get("processed_dataset_info"),
                    hf_load_options=d.get("hf_load_options", {}),
                    created_at=d.get("created_at", ""),
                )
                self._data_vars[uid] = meta
                if legacy_name:
                    self._migrate_pickle(legacy_name, uid)
            if migrated:
                self._save_data_vars()
        except (json.JSONDecodeError, IOError, TypeError):
            self._data_vars = {}

    def _save_data_vars(self) -> None:
        """Persist data variables to disk."""
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            data = {uid: asdict(meta) for uid, meta in self._data_vars.items()}
            with open(DATA_VARS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError:
            pass

    def _load_residual_vars(self) -> None:
        """Load residual variables from disk."""
        if not RESIDUAL_VARS_FILE.exists():
            return
        migrated = False
        try:
            with open(RESIDUAL_VARS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._residual_vars = {}
            for key, rv in (raw or {}).items():
                if not isinstance(rv, dict):
                    continue
                uid = (rv.get("uid") or "").strip()
                nickname = (rv.get("nickname") or rv.get("name") or key or "").strip()
                legacy_name = ""
                if not uid:
                    uid = self._generate_uid("res")
                    migrated = True
                    legacy_name = key
                record = dict(rv)
                record["uid"] = uid
                record["nickname"] = nickname or uid
                self._residual_vars[uid] = record
                if legacy_name:
                    self._migrate_pickle(legacy_name, uid)
            if migrated:
                self._save_residual_vars()
        except (json.JSONDecodeError, IOError):
            self._residual_vars = {}

    def _save_residual_vars(self) -> None:
        """Persist residual variables to disk."""
        RESIDUAL_VARS_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(RESIDUAL_VARS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._residual_vars, f, indent=2, ensure_ascii=False)
        except IOError:
            pass

    @staticmethod
    def _build_saved_var_name(
        path: str,
        split: Optional[str],
        random_n: Optional[int],
        seed: Optional[int],
        task_name: str,
        additional_naming: Optional[str] = None,
    ) -> str:
        """Build variable name: dataName/split/randomN/seed/taskName[/additionalNaming]."""
        parts = [
            (path or "").strip() or "-",
            (str(split).strip() if split is not None and str(split).strip() else "-"),
            (str(random_n) if random_n is not None else "-"),
            (str(seed) if seed is not None else "-"),
            (task_name or "").strip() or "-",
        ]
        base = "/".join(parts)
        extra = (additional_naming or "").strip()
        return base + (f"/{extra}" if extra else "")

    @staticmethod
    def _estimate_data_var_memory_mb(data: Dict[str, Any]) -> float:
        """Rough estimate: total_rows * ~500 bytes -> MB."""
        total_rows = 0
        for info in (data.get("dataset_info"), data.get("processed_dataset_info")):
            if not info:
                continue
            for n in (info.get("num_rows") or {}).values():
                if n is not None:
                    total_rows += int(n)
        if total_rows == 0:
            return 0.0
        return round(total_rows * 0.0005, 2)

    def save(
        self,
        pipeline_id: str,
        pipeline: Dict[str, Any],
        additional_naming: Optional[str] = None,
    ) -> str:
        """Register/update a working-memory entry from a dataset pipeline. Returns variable name."""
        path = (pipeline.get("hf_dataset_path") or "").strip()
        if not path:
            raise ValueError("Pipeline has no dataset path. Load a dataset first.")

        task_name = (pipeline.get("name") or "").strip() or "-"
        opts = pipeline.get("hf_load_options") or {}
        split = opts.get("split")
        random_n = opts.get("random_n")
        seed = opts.get("seed")

        var_name = self._build_saved_var_name(
            path=path,
            split=split,
            random_n=random_n,
            seed=seed,
            task_name=task_name,
            additional_naming=additional_naming,
        )

        uid = self._generate_uid("var")
        meta = DataVarMeta(
            uid=uid,
            nickname=var_name or uid,
            pipeline_id=pipeline_id,
            task_name=task_name,
            data_name=path,
            split=split,
            random_n=random_n,
            seed=seed,
            dataset_info=pipeline.get("dataset_info"),
            processed_dataset_info=pipeline.get("processed_dataset_info"),
            hf_load_options=opts,
            created_at=datetime.now().isoformat(),
        )
        self._data_vars[uid] = meta
        self._save_data_vars()
        return meta

    def save_residual(
        self,
        *,
        directions: Dict[str, Any],
        task_name: str,
        model: str,
        num_keys: int,
        model_dim: int,
        additional_naming: Optional[str] = None,
    ) -> str:
        """Save residual directions dict. Returns variable name."""
        base = f"residual/{task_name}/{model}/{num_keys}x{model_dim}"
        extra = (additional_naming or "").strip()
        nickname = base + (f"/{extra}" if extra else "")
        uid = self._generate_uid("res")
        self._residual_vars[uid] = {
            "uid": uid,
            "nickname": nickname,
            "directions": directions,
            "task_name": task_name,
            "model": model,
            "num_keys": num_keys,
            "model_dim": model_dim,
            "created_at": datetime.now().isoformat(),
        }
        self._save_residual_vars()
        return uid

    def resolve_id(self, name_or_id: str, preferred_type: Optional[str] = None) -> Optional[str]:
        if not name_or_id:
            return None
        if name_or_id in self._data_vars or name_or_id in self._residual_vars:
            return name_or_id
        matches = []
        if preferred_type in (None, "data"):
            for uid, meta in self._data_vars.items():
                if meta.nickname == name_or_id:
                    matches.append((uid, meta.created_at))
        if preferred_type in (None, "residual"):
            for uid, rv in self._residual_vars.items():
                if (rv.get("nickname") or "") == name_or_id:
                    matches.append((uid, rv.get("created_at", "")))
        if not matches:
            return None
        matches.sort(key=lambda x: x[1] or "", reverse=True)
        return matches[0][0]

    def _nickname_exists(self, nickname: str, exclude_uid: Optional[str] = None) -> bool:
        for uid, meta in self._data_vars.items():
            if uid == exclude_uid:
                continue
            if meta.nickname == nickname:
                return True
        for uid, rv in self._residual_vars.items():
            if uid == exclude_uid:
                continue
            if (rv.get("nickname") or "") == nickname:
                return True
        return False

    def get_residual(self, name: str) -> Optional[Dict[str, Any]]:
        """Get residual directions by variable id or nickname."""
        uid = self.resolve_id(name, preferred_type="residual")
        if not uid:
            return None
        return self._residual_vars.get(uid)

    def delete(self, name: str) -> bool:
        """Delete a variable by id or nickname. Returns True if deleted."""
        uid = self.resolve_id(name)
        if not uid:
            return False
        if uid in self._data_vars:
            nickname = self._data_vars[uid].nickname or ""
            del self._data_vars[uid]
            self._save_data_vars()
            self.delete_pickle(uid)
            if nickname and nickname != uid:
                self.delete_raw_pickle(nickname)
            return True
        if uid in self._residual_vars:
            nickname = (self._residual_vars[uid] or {}).get("nickname") or ""
            del self._residual_vars[uid]
            self._save_residual_vars()
            self.delete_pickle(uid)
            if nickname and nickname != uid:
                self.delete_raw_pickle(nickname)
            return True
        return False

    def clear_all(self) -> None:
        """Drop all registered working-memory entries."""
        self._data_vars.clear()
        self._residual_vars.clear()
        self._save_data_vars()
        self._save_residual_vars()

    def get_meta(self, name: str) -> Optional[DataVarMeta]:
        """Get variable metadata by id or nickname."""
        uid = self.resolve_id(name, preferred_type="data")
        if not uid:
            return None
        return self._data_vars.get(uid)

    def pickle_path(self, name: str) -> Path:
        uid = self.resolve_id(name) or name
        return PICKLE_DIR / f"{_safe_name(uid)}.pkl"

    def _raw_pickle_path(self, name: str) -> Path:
        return PICKLE_DIR / f"{_safe_name(name)}.pkl"

    def has_pickle(self, name: str) -> bool:
        return self.pickle_path(name).exists()

    def save_pickle(self, name: str, payload: Any) -> bool:
        """Persist actual variable payload as pickle. Returns True if saved."""
        if not name:
            return False
        try:
            PICKLE_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.pickle_path(name), "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception:
            return False

    def load_pickle(self, name: str) -> Optional[Any]:
        """Load pickled variable payload."""
        path = self.pickle_path(name)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def delete_pickle(self, name: str) -> None:
        path = self.pickle_path(name)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass

    def delete_raw_pickle(self, name: str) -> None:
        path = self._raw_pickle_path(name)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass

    def get_detail(self, name: str) -> Optional[Dict[str, Any]]:
        """Get full variable detail by id or nickname (data or residual)."""
        uid = self.resolve_id(name)
        if not uid:
            return None
        if uid in self._data_vars:
            meta = self._data_vars[uid]
            mem_mb = self._estimate_data_var_memory_mb(
                {
                    "dataset_info": meta.dataset_info,
                    "processed_dataset_info": meta.processed_dataset_info,
                }
            )
            return {
                "id": uid,
                "name": meta.nickname,
                "type": "data",
                "created_at": meta.created_at,
                "pipeline_id": meta.pipeline_id,
                "task_name": meta.task_name,
                "data_name": meta.data_name,
                "split": meta.split,
                "random_n": meta.random_n,
                "seed": meta.seed,
                "dataset_info": meta.dataset_info,
                "processed_dataset_info": meta.processed_dataset_info,
                "hf_load_options": meta.hf_load_options,
                "memory_ram_mb": mem_mb,
                "hd_path": str(self.pickle_path(uid)),
            }
        if uid in self._residual_vars:
            rv = self._residual_vars[uid] or {}
            num_keys = rv.get("num_keys", 0)
            model_dim = rv.get("model_dim", 0)
            mem_mb = round(num_keys * model_dim * 8 / (1024 * 1024), 2) if num_keys and model_dim else 0
            return {
                "id": uid,
                "name": rv.get("nickname") or uid,
                "type": "residual",
                "created_at": rv.get("created_at", ""),
                "task_name": rv.get("task_name", ""),
                "model": rv.get("model", ""),
                "num_keys": num_keys,
                "model_dim": model_dim,
                "memory_ram_mb": mem_mb,
                "hd_path": str(self.pickle_path(uid)),
            }
        return None

    def rename(self, old_name: str, new_name: str) -> bool:
        """Rename a variable nickname. Returns True if renamed."""
        if not old_name or not new_name or old_name == new_name:
            return False
        uid = self.resolve_id(old_name)
        if not uid:
            return False
        if self._nickname_exists(new_name, exclude_uid=uid):
            return False
        if uid in self._data_vars:
            meta = self._data_vars[uid]
            meta.nickname = new_name
            self._save_data_vars()
            return True
        if uid in self._residual_vars:
            rv = self._residual_vars[uid]
            rv["nickname"] = new_name
            self._save_residual_vars()
            return True
        return False

    def summarize_for_panel(self, loaded_model_key: Optional[str]) -> Dict[str, Any]:
        """
        Build the JSON payload for the Variables panel.
        Returns: { "loaded_model": {...}, "variables": [...] }
        """
        loaded_model = None
        if loaded_model_key:
            try:
                status = get_model_status(loaded_model_key)
                gpu_gb = 0.0
                ram_gb = 0.0
                for dev in (status.get("device_status") or []):
                    device = dev.get("device", "")
                    gb = float(dev.get("memory_gb") or 0)
                    if device.startswith("cuda"):
                        gpu_gb += gb
                    else:
                        ram_gb += gb
                loaded_model = {
                    "name": loaded_model_key,
                    "memory_gpu_gb": round(gpu_gb, 3),
                    "memory_ram_gb": round(ram_gb, 3),
                }
            except Exception:
                loaded_model = {
                    "name": loaded_model_key,
                    "memory_gpu_gb": None,
                    "memory_ram_gb": None,
                }

        variables = []
        for uid, meta in self._data_vars.items():
            meta_dict = asdict(meta)
            mem_mb = self._estimate_data_var_memory_mb(
                {
                    "dataset_info": meta.dataset_info,
                    "processed_dataset_info": meta.processed_dataset_info,
                }
            )
            variables.append({
                "id": uid,
                "name": meta.nickname,
                "pipeline_id": meta_dict.get("pipeline_id"),
                "task_name": meta_dict.get("task_name"),
                "created_at": meta_dict.get("created_at"),
                "memory_ram_mb": mem_mb,
                "type": meta_dict.get("type", "data"),
            })
        for uid, rv in self._residual_vars.items():
            num_keys = rv.get("num_keys", 0)
            model_dim = rv.get("model_dim", 0)
            mem_mb = round(num_keys * model_dim * 8 / (1024 * 1024), 2) if num_keys and model_dim else 0
            variables.append({
                "id": uid,
                "name": rv.get("nickname") or uid,
                "pipeline_id": None,
                "task_name": rv.get("task_name", ""),
                "created_at": rv.get("created_at", ""),
                "memory_ram_mb": mem_mb,
                "type": "residual",
            })

        return {"loaded_model": loaded_model, "variables": variables}


variable_store = VariableStore()


# ----- Module-level helpers (public API) -----


def summarize_for_panel(loaded_model_key: Optional[str]) -> Dict[str, Any]:
    """Build JSON payload for Variables panel."""
    return variable_store.summarize_for_panel(loaded_model_key=loaded_model_key)


def save_pipeline_variable(
    *, pipeline_id: str, pipeline: Dict[str, Any], additional_naming: Optional[str] = None
) -> Tuple[str, str]:
    """Register/update a working-memory entry from a dataset pipeline. Returns (uid, nickname)."""
    meta = variable_store.save(
        pipeline_id=pipeline_id,
        pipeline=pipeline,
        additional_naming=additional_naming,
    )
    return meta.uid, meta.nickname


def delete_variable(name: str) -> bool:
    """Delete a working-memory variable by name."""
    return variable_store.delete(name)


def rename_variable(old_name: str, new_name: str) -> bool:
    """Rename a working-memory variable by id or nickname (nickname only)."""
    return variable_store.rename(old_name, new_name)


def get_variable_detail(name: str) -> Optional[Dict[str, Any]]:
    """Get full detail for a variable (data or residual)."""
    return variable_store.get_detail(name)


def has_variable_pickle(name: str) -> bool:
    """Check if variable pickle exists on disk."""
    return variable_store.has_pickle(name)


def save_variable_pickle(name: str, payload: Any) -> bool:
    """Save variable payload as pickle."""
    return variable_store.save_pickle(name, payload)


def load_variable_pickle(name: str) -> Optional[Any]:
    """Load variable payload from pickle."""
    return variable_store.load_pickle(name)


def clear_all() -> None:
    """Clear all working-memory entries."""
    variable_store.clear_all()


def save_residual_variable(
    *,
    directions: Dict[str, Any],
    task_name: str,
    model: str,
    num_keys: int,
    model_dim: int,
    additional_naming: Optional[str] = None,
) -> Tuple[str, str]:
    """Save residual directions dict to variable store. Returns (uid, nickname)."""
    uid = variable_store.save_residual(
        directions=directions,
        task_name=task_name,
        model=model,
        num_keys=num_keys,
        model_dim=model_dim,
        additional_naming=additional_naming,
    )
    rv = variable_store.get_residual(uid) or {}
    return uid, (rv.get("nickname") or uid)


def get_residual_variable(name: str) -> Optional[Dict[str, Any]]:
    """Get residual directions by variable name."""
    return variable_store.get_residual(name)
