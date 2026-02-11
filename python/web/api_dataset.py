import json
import re
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request

from python.dataset_pipeline_store import (
    add_pipeline,
    delete_pipeline,
    get_pipeline_by_id,
    get_pipelines,
    update_pipeline as update_pipeline_store,
)
from python.memory import cache_store, variable_store
from python.memory.variable import (
    delete_variable,
    get_residual_variable,
    get_variable_detail,
    has_variable_pickle,
    load_variable_pickle,
    rename_variable,
    save_pipeline_variable,
    save_residual_variable,
    save_variable_pickle,
)
from python.web.dataset_utils import (
    dataset_to_info,
    get_process_function,
    load_pipeline_dataset,
    random_select_dataset,
)


dataset_bp = Blueprint("dataset", __name__)

_VARIABLE_LOAD_NAMESPACE = "variable_loaded"
_EXPORT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "variable_exports"


def _sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return safe.strip("._") or "variable"


def _get_loaded_vars_map() -> dict:
    ns = cache_store.get_namespace(_VARIABLE_LOAD_NAMESPACE)
    return ns if isinstance(ns, dict) else {}


def _cache_object_name(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ""
    if payload.get("type") == "data":
        ds = payload.get("dataset")
        if ds is not None:
            return type(ds).__name__
        return "Dataset"
    if payload.get("type") == "residual":
        return "ResidualDirections"
    return payload.get("type", "")


def _infer_payload_device(payload) -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"
    try:
        if isinstance(payload, dict) and payload.get("type") == "data":
            return "cpu"
        if isinstance(payload, dict) and isinstance(payload.get("directions"), dict):
            sample = next(iter(payload["directions"].values()), None)
            if isinstance(sample, torch.Tensor):
                return "cuda" if sample.is_cuda else "cpu"
        if isinstance(payload, torch.Tensor):
            return "cuda" if payload.is_cuda else "cpu"
    except Exception:
        return "cpu"
    return "cpu"


@dataset_bp.get("/api/dataset-pipelines")
def api_list_pipelines():
    return jsonify({"pipelines": get_pipelines()})


@dataset_bp.post("/api/dataset-pipelines")
def api_create_pipeline():
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip() or "Unnamed"
    pipeline_id = add_pipeline(name=name, status="empty")
    pipeline = get_pipeline_by_id(pipeline_id)
    return jsonify({"status": "ok", "pipeline": pipeline, "id": pipeline_id})


@dataset_bp.get("/api/dataset-pipelines/<pipeline_id>")
def api_get_pipeline(pipeline_id: str):
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        return jsonify({"error": "Pipeline not found"}), 404
    return jsonify(pipeline)


@dataset_bp.patch("/api/dataset-pipelines/<pipeline_id>")
def api_update_pipeline(pipeline_id: str):
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        return jsonify({"error": "Pipeline not found"}), 404
    data = request.get_json(force=True) or {}
    if "name" in data:
        update_pipeline_store(pipeline_id, name=data.get("name"))
    if "status" in data:
        update_pipeline_store(pipeline_id, status=data.get("status"))
    if "hf_dataset_path" in data:
        update_pipeline_store(pipeline_id, hf_dataset_path=data.get("hf_dataset_path"))
    if "config" in data:
        update_pipeline_store(pipeline_id, config=data.get("config"))
    updated = get_pipeline_by_id(pipeline_id)
    return jsonify(updated)


@dataset_bp.delete("/api/dataset-pipelines/<pipeline_id>")
def api_delete_pipeline(pipeline_id: str):
    if not get_pipeline_by_id(pipeline_id):
        return jsonify({"error": "Pipeline not found"}), 404
    delete_pipeline(pipeline_id)
    return jsonify({"status": "ok"})


@dataset_bp.post("/api/dataset-pipelines/<pipeline_id>/process")
def api_apply_processing(pipeline_id: str):
    """Apply Python map function to pipeline's dataset. Body: { "processing_code": "def process(example): ..." }."""
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        return jsonify({"error": "Pipeline not found"}), 404
    data = request.get_json(force=True) or {}
    code = (data.get("processing_code") or data.get("code") or "").strip()
    if not code:
        return jsonify({"error": "processing_code is required"}), 400
    try:
        process_fn = get_process_function(code)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400
    try:
        ds, requested_split = load_pipeline_dataset(pipeline)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400
    try:
        if hasattr(ds, "keys"):
            ds = ds.map(process_fn, batched=False, remove_columns=None, desc="Processing")
        else:
            ds = ds.map(process_fn, batched=False, remove_columns=None, desc="Processing")
        processed_info = dataset_to_info(ds, requested_split=requested_split)
        update_pipeline_store(
            pipeline_id,
            status="processed",
            processing_code=code,
            processed_dataset_info=processed_info,
        )
        return jsonify(
            {
                "status": "ok",
                "processed_dataset_info": processed_info,
            }
        )
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@dataset_bp.post("/api/dataset-pipelines/<pipeline_id>/load")
def api_load_hf_dataset(pipeline_id: str):
    """Load dataset from Hugging Face. Accepts path, config name, split (e.g. train, test)."""
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        return jsonify({"error": "Pipeline not found"}), 404
    data = request.get_json(force=True) or {}
    path = (data.get("path") or data.get("hf_dataset_path") or "").strip()
    if not path:
        return jsonify({"error": "Dataset path required"}), 400
    config_name = (data.get("config") or data.get("config_name") or "").strip() or None
    split = data.get("split")
    if isinstance(split, str) and split.strip():
        split = split.strip()
    elif isinstance(split, list):
        split = "+".join(str(s).strip() for s in split if str(s).strip())
    else:
        split = None
    random_n = data.get("random_n") or data.get("n")
    if random_n is not None:
        try:
            random_n = int(random_n)
        except (TypeError, ValueError):
            random_n = None
    seed = data.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = None
    try:
        import datasets
    except ImportError:
        return jsonify({"error": "Install datasets: pip install datasets"}), 500
    try:
        load_kwargs = {"path": path}
        if config_name:
            load_kwargs["name"] = config_name
        if split:
            load_kwargs["split"] = split
        ds = datasets.load_dataset(**load_kwargs)
        if random_n is not None and random_n > 0 and seed is not None:
            ds = random_select_dataset(ds, n=random_n, seed=seed, requested_split=split)
        dataset_info = dataset_to_info(ds, requested_split=split)
        hf_load_options = {
            "config": config_name,
            "split": split,
            "random_n": random_n,
            "seed": seed,
        }
        update_pipeline_store(
            pipeline_id,
            hf_dataset_path=path,
            status="loaded",
            hf_load_options=hf_load_options,
            dataset_info=dataset_info,
        )
        return jsonify(
            {
                "status": "ok",
                "path": path,
                "load_options": hf_load_options,
                "dataset_info": dataset_info,
            }
        )
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@dataset_bp.get("/api/data-vars")
def api_list_data_vars():
    """List loaded model (with GPU/RAM) and saved working‑memory variables."""
    loaded = cache_store.get_session().get("loaded_model")
    payload = variable_store.summarize_for_panel(loaded_model_key=loaded)
    loaded_map = _get_loaded_vars_map()
    loaded_ids = set(loaded_map.keys())

    variables = []
    seen_ids = set()
    for v in payload.get("variables", []):
        var_id = (v.get("id") or "").strip()
        if not var_id:
            continue
        has_ram = var_id in loaded_ids
        has_gpu = False
        if has_ram:
            entry = loaded_map.get(var_id) or {}
            device = (entry.get("device") or "").lower()
            has_gpu = device.startswith("cuda") or device == "gpu"
        has_pickle = has_variable_pickle(var_id)
        has_disk = has_pickle
        if has_gpu and has_disk:
            status = "gpu_and_disk"
            status_label = "GPU + Disk"
        elif has_gpu and not has_disk:
            status = "gpu_only"
            status_label = "GPU only"
        elif has_ram and has_disk:
            status = "ram_and_disk"
            status_label = "RAM + Disk"
        elif has_ram and not has_disk:
            status = "ram_only"
            status_label = "RAM only"
        elif not has_ram and has_disk:
            status = "disk_only"
            status_label = "Disk only"
        else:
            status = "missing"
            status_label = "Missing"
        v.update(
            {
                "has_ram": has_ram,
                "has_gpu": has_gpu,
                "has_disk": has_disk,
                "has_pickle": has_pickle,
                "status": status,
                "status_label": status_label,
            }
        )
        variables.append(v)
        seen_ids.add(var_id)

    for var_id, entry in loaded_map.items():
        if var_id in seen_ids:
            continue
        payload_type = (entry or {}).get("type") if isinstance(entry, dict) else None
        display_name = (entry or {}).get("display_name") if isinstance(entry, dict) else None
        device = (entry or {}).get("device") if isinstance(entry, dict) else None
        has_gpu = bool(device) and str(device).lower().startswith("cuda")
        variables.append(
            {
                "id": var_id,
                "name": display_name or var_id,
                "type": payload_type or "data",
                "created_at": (entry or {}).get("loaded_at", "") if isinstance(entry, dict) else "",
                "memory_ram_mb": None,
                "has_ram": True,
                "has_gpu": has_gpu,
                "has_disk": False,
                "has_pickle": False,
                "status": "gpu_only" if has_gpu else "ram_only",
                "status_label": "GPU only" if has_gpu else "RAM only",
            }
        )

    payload["variables"] = variables
    return jsonify(payload)


@dataset_bp.delete("/api/data-vars/<path:var_id>")
def api_delete_data_var(var_id: str):
    """Remove a working‑memory variable by id (URL‑decoded)."""
    resolved = variable_store.resolve_id(var_id) or var_id
    if not delete_variable(var_id):
        return jsonify({"error": "Variable not found"}), 404
    if cache_store.get(_VARIABLE_LOAD_NAMESPACE, resolved) is not None:
        cache_store.delete(_VARIABLE_LOAD_NAMESPACE, resolved)
    return jsonify({"status": "ok"})


@dataset_bp.get("/api/data-vars/<path:var_id>/detail")
def api_data_var_detail(var_id: str):
    """Get a variable's detail payload for the panel."""
    detail = get_variable_detail(var_id)
    if not detail:
        loaded_entry = cache_store.get(_VARIABLE_LOAD_NAMESPACE, var_id)
        if loaded_entry is None:
            return jsonify({"error": "Variable not found"}), 404
        payload_type = (loaded_entry or {}).get("type") if isinstance(loaded_entry, dict) else None
        detail = {
            "id": var_id,
            "name": (loaded_entry or {}).get("display_name") if isinstance(loaded_entry, dict) else var_id,
            "type": payload_type or "data",
            "created_at": (loaded_entry or {}).get("loaded_at", "") if isinstance(loaded_entry, dict) else "",
            "memory_ram_mb": None,
            "has_disk": False,
            "hd_path": str(variable_store.pickle_path(var_id)),
        }
    loaded_entry = cache_store.get(_VARIABLE_LOAD_NAMESPACE, detail.get("id") or var_id)
    detail["is_loaded"] = loaded_entry is not None
    detail["has_pickle"] = has_variable_pickle(detail.get("id") or var_id)
    detail["has_disk"] = detail["has_pickle"]
    if loaded_entry is not None:
        detail["cache_object_name"] = _cache_object_name(loaded_entry)
        detail["device"] = (loaded_entry or {}).get("device", "cpu")
    else:
        detail["cache_object_name"] = ""
    return jsonify(detail)


@dataset_bp.post("/api/data-vars/<path:var_id>/rename")
def api_rename_data_var(var_id: str):
    data = request.get_json(force=True) or {}
    new_name = (data.get("new_name") or "").strip()
    if not new_name:
        return jsonify({"error": "new_name required"}), 400
    resolved = variable_store.resolve_id(var_id) or var_id
    if not rename_variable(var_id, new_name):
        return jsonify({"error": "Rename failed"}), 400
    loaded = cache_store.get(_VARIABLE_LOAD_NAMESPACE, resolved)
    if loaded is not None:
        loaded["display_name"] = new_name
        cache_store.put(_VARIABLE_LOAD_NAMESPACE, resolved, loaded)
    return jsonify({"status": "ok", "new_name": new_name, "id": resolved})


@dataset_bp.post("/api/data-vars/<path:var_id>/load")
def api_load_data_var(var_id: str):
    """Load a variable into memory cache for quick inspection."""
    detail = get_variable_detail(var_id)
    if not detail:
        return jsonify({"error": "Variable not found"}), 404
    resolved = detail.get("id") or var_id
    if not has_variable_pickle(resolved):
        return jsonify({"error": "No HD snapshot found for this variable. Save to disk first."}), 400
    payload = load_variable_pickle(resolved)
    if payload is None:
        return jsonify({"error": "Failed to load HD snapshot."}), 500
    device = _infer_payload_device(payload)
    if detail.get("type") == "residual":
        cache_store.put(
            _VARIABLE_LOAD_NAMESPACE,
            resolved,
            {
                "type": "residual",
                "data": payload,
                "loaded_at": datetime.now().isoformat(),
                "object_name": "ResidualDirections",
                "display_name": detail.get("name") or resolved,
                "device": device,
            },
        )
    else:
        ds = payload.get("dataset") if isinstance(payload, dict) else payload
        cache_store.put(
            _VARIABLE_LOAD_NAMESPACE,
            resolved,
            {
                "type": "data",
                "dataset": ds,
                "requested_split": payload.get("split") if isinstance(payload, dict) else None,
                "loaded_at": datetime.now().isoformat(),
                "object_name": type(ds).__name__ if ds is not None else "Dataset",
                "display_name": detail.get("name") or resolved,
                "device": device,
            },
        )
    return jsonify({"status": "ok"})


@dataset_bp.post("/api/data-vars/<path:var_id>/unload")
def api_unload_data_var(var_id: str):
    """Unload a variable from memory cache."""
    resolved = variable_store.resolve_id(var_id) or var_id
    if cache_store.get(_VARIABLE_LOAD_NAMESPACE, resolved) is not None:
        cache_store.delete(_VARIABLE_LOAD_NAMESPACE, resolved)
    return jsonify({"status": "ok"})


@dataset_bp.post("/api/data-vars/<path:var_id>/export")
def api_export_data_var(var_id: str):
    """Export a variable snapshot to disk (JSON)."""
    detail = get_variable_detail(var_id)
    if not detail:
        return jsonify({"error": "Variable not found"}), 404
    export = {"detail": detail, "exported_at": datetime.now().isoformat()}
    if detail.get("type") == "residual":
        rv = get_residual_variable(detail.get("id") or var_id)
        if rv:
            export["residual"] = rv
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe = _sanitize_filename(detail.get("name") or detail.get("id") or var_id)
    filename = f"{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = _EXPORT_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    return jsonify({"status": "ok", "path": str(out_path)})



@dataset_bp.post("/api/data-vars/save")
def api_save_data_var():
    """
    Save current pipeline state to global working memory.

    Requires: Load → Apply Processing → Save. Only processed data is saved.
    Body: { pipeline_id, additional_naming? }.
    Variable name uses dataset path / split / random_n / seed / task name;
    if additional_naming is given, it is appended at the end.
    """
    data = request.get_json(force=True) or {}
    pipeline_id = (data.get("pipeline_id") or "").strip()
    additional_naming = (data.get("additional_naming") or "").strip() or None
    if not pipeline_id:
        return jsonify({"error": "pipeline_id required"}), 400
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        return jsonify({"error": "Pipeline not found"}), 404
    if pipeline.get("status") != "processed":
        return (
            jsonify(
                {
                    "error": "Apply Processing first before Save To Variable. Load → Apply Processing → Save.",
                }
            ),
            400,
        )
    try:
        var_id, var_name = save_pipeline_variable(
            pipeline_id=pipeline_id,
            pipeline=pipeline,
            additional_naming=additional_naming,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    pickle_saved = False
    loaded = False
    try:
        ds, requested_split = load_pipeline_dataset(pipeline)
        code = (pipeline.get("processing_code") or "").strip()
        if pipeline.get("status") == "processed" and code:
            process_fn = get_process_function(code)
            ds = ds.map(process_fn, batched=False, remove_columns=None, desc="Processing")
        payload = {"type": "data", "dataset": ds, "split": requested_split}
        pickle_saved = save_variable_pickle(var_id, payload)
        if pickle_saved:
            device = _infer_payload_device(payload)
            cache_store.put(
                _VARIABLE_LOAD_NAMESPACE,
                var_id,
                {
                    "type": "data",
                    "dataset": ds,
                    "requested_split": requested_split,
                    "loaded_at": datetime.now().isoformat(),
                    "object_name": type(ds).__name__,
                    "display_name": var_name,
                    "device": device,
                },
            )
            loaded = True
    except Exception:
        pickle_saved = False
    return jsonify({
        "status": "ok",
        "variable_id": var_id,
        "variable_name": var_name,
        "pickle_saved": pickle_saved,
        "loaded": loaded,
    })


@dataset_bp.get("/api/residual-vars/<path:var_id>")
def api_get_residual_var(var_id: str):
    """Get residual variable by id (directions dict)."""
    if has_variable_pickle(var_id):
        payload = load_variable_pickle(var_id)
        if isinstance(payload, dict) and payload.get("directions"):
            return jsonify(payload)
    rv = get_residual_variable(var_id)
    if not rv:
        return jsonify({"error": "Variable not found"}), 404
    return jsonify(rv)


@dataset_bp.post("/api/residual-vars/save")
def api_save_residual_var():
    """
    Save residual direction vectors to variable.
    Body: { directions, task_name, model, num_keys, model_dim, additional_naming? }.
    """
    data = request.get_json(force=True) or {}
    directions = data.get("directions")
    if not directions or not isinstance(directions, dict):
        return jsonify({"error": "directions (dict) required"}), 400
    task_name = (data.get("task_name") or "").strip() or "Residual"
    model = (data.get("model") or "").strip() or "-"
    num_keys = data.get("num_keys")
    model_dim = data.get("model_dim")
    if num_keys is None:
        num_keys = len(directions)
    if model_dim is None and directions:
        first = next(iter(directions.values()), [])
        model_dim = len(first) if isinstance(first, (list, tuple)) else 0
    additional_naming = (data.get("additional_naming") or "").strip() or None
    try:
        var_id, var_name = save_residual_variable(
            directions=directions,
            task_name=task_name,
            model=model,
            num_keys=int(num_keys),
            model_dim=int(model_dim or 0),
            additional_naming=additional_naming,
        )
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    payload = {
        "directions": directions,
        "task_name": task_name,
        "model": model,
        "num_keys": int(num_keys),
        "model_dim": int(model_dim or 0),
        "created_at": datetime.now().isoformat(),
    }
    pickle_saved = save_variable_pickle(var_id, payload)
    loaded = False
    if pickle_saved:
        device = _infer_payload_device(payload)
        cache_store.put(
            _VARIABLE_LOAD_NAMESPACE,
            var_id,
            {
                "type": "residual",
                "data": payload,
                "loaded_at": datetime.now().isoformat(),
                "object_name": "ResidualDirections",
                "display_name": var_name,
                "device": device,
            },
        )
        loaded = True
    return jsonify({
        "status": "ok",
        "variable_id": var_id,
        "variable_name": var_name,
        "pickle_saved": pickle_saved,
        "loaded": loaded,
    })
