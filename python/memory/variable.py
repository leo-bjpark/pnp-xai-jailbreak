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
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from python.model_load import get_model_status

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RESIDUAL_VARS_FILE = _DATA_DIR / "residual_variables.json"
DATA_VARS_FILE = _DATA_DIR / "data_variables.json"


@dataclass
class DataVarMeta:
    """
    Metadata for a single working-memory entry from a dataset pipeline.
    Lightweight and JSON-serializable for Flask handlers.
    """

    variable_name: str
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

    def _load_data_vars(self) -> None:
        """Load data variables from disk."""
        if not DATA_VARS_FILE.exists():
            return
        try:
            with open(DATA_VARS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for name, d in (raw or {}).items():
                if isinstance(d, dict) and d.get("type") == "data":
                    self._data_vars[name] = DataVarMeta(
                        variable_name=d.get("variable_name", name),
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
        except (json.JSONDecodeError, IOError, TypeError):
            self._data_vars = {}

    def _save_data_vars(self) -> None:
        """Persist data variables to disk."""
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            data = {name: asdict(meta) for name, meta in self._data_vars.items()}
            with open(DATA_VARS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError:
            pass

    def _load_residual_vars(self) -> None:
        """Load residual variables from disk."""
        if not RESIDUAL_VARS_FILE.exists():
            return
        try:
            with open(RESIDUAL_VARS_FILE, "r", encoding="utf-8") as f:
                self._residual_vars = json.load(f)
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

        meta = DataVarMeta(
            variable_name=var_name,
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
        self._data_vars[var_name] = meta
        self._save_data_vars()
        return var_name

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
        var_name = base + (f"/{extra}" if extra else "")
        self._residual_vars[var_name] = {
            "directions": directions,
            "task_name": task_name,
            "model": model,
            "num_keys": num_keys,
            "model_dim": model_dim,
            "created_at": datetime.now().isoformat(),
        }
        self._save_residual_vars()
        return var_name

    def get_residual(self, name: str) -> Optional[Dict[str, Any]]:
        """Get residual directions by variable name. Returns full record with directions key."""
        return self._residual_vars.get(name)

    def delete(self, name: str) -> bool:
        """Delete a variable by exact name. Returns True if deleted."""
        if name in self._data_vars:
            del self._data_vars[name]
            self._save_data_vars()
            return True
        if name in self._residual_vars:
            del self._residual_vars[name]
            self._save_residual_vars()
            return True
        return False

    def clear_all(self) -> None:
        """Drop all registered working-memory entries."""
        self._data_vars.clear()
        self._residual_vars.clear()
        self._save_data_vars()
        self._save_residual_vars()

    def get_meta(self, name: str) -> Optional[DataVarMeta]:
        """Get variable metadata by name."""
        return self._data_vars.get(name)

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
        for name, meta in self._data_vars.items():
            meta_dict = asdict(meta)
            mem_mb = self._estimate_data_var_memory_mb(
                {
                    "dataset_info": meta.dataset_info,
                    "processed_dataset_info": meta.processed_dataset_info,
                }
            )
            variables.append({
                "name": name,
                "pipeline_id": meta_dict.get("pipeline_id"),
                "task_name": meta_dict.get("task_name"),
                "created_at": meta_dict.get("created_at"),
                "memory_ram_mb": mem_mb,
                "type": meta_dict.get("type", "data"),
            })
        for name, rv in self._residual_vars.items():
            num_keys = rv.get("num_keys", 0)
            model_dim = rv.get("model_dim", 0)
            mem_mb = round(num_keys * model_dim * 8 / (1024 * 1024), 2) if num_keys and model_dim else 0
            variables.append({
                "name": name,
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
) -> str:
    """Register/update a working-memory entry from a dataset pipeline. Returns variable name."""
    return variable_store.save(
        pipeline_id=pipeline_id,
        pipeline=pipeline,
        additional_naming=additional_naming,
    )


def delete_variable(name: str) -> bool:
    """Delete a working-memory variable by name."""
    return variable_store.delete(name)


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
) -> str:
    """Save residual directions dict to variable store. Returns variable name."""
    return variable_store.save_residual(
        directions=directions,
        task_name=task_name,
        model=model,
        num_keys=num_keys,
        model_dim=model_dim,
        additional_naming=additional_naming,
    )


def get_residual_variable(name: str) -> Optional[Dict[str, Any]]:
    """Get residual directions by variable name."""
    return variable_store.get_residual(name)
