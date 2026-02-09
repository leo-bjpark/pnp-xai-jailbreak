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
    get_residual_variable,
    save_pipeline_variable,
    save_residual_variable,
)
from python.web.dataset_utils import (
    dataset_to_info,
    get_process_function,
    load_pipeline_dataset,
    random_select_dataset,
)


dataset_bp = Blueprint("dataset", __name__)


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
    return jsonify(payload)


@dataset_bp.delete("/api/data-vars/<path:var_name>")
def api_delete_data_var(var_name: str):
    """Remove a working‑memory variable by name (URL‑decoded)."""
    if not variable_store.delete_variable(var_name):
        return jsonify({"error": "Variable not found"}), 404
    return jsonify({"status": "ok"})


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
        var_name = save_pipeline_variable(
            pipeline_id=pipeline_id,
            pipeline=pipeline,
            additional_naming=additional_naming,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "ok", "variable_name": var_name})


@dataset_bp.get("/api/residual-vars/<path:var_name>")
def api_get_residual_var(var_name: str):
    """Get residual variable by name (directions dict)."""
    rv = get_residual_variable(var_name)
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
        var_name = save_residual_variable(
            directions=directions,
            task_name=task_name,
            model=model,
            num_keys=int(num_keys),
            model_dim=int(model_dim or 0),
            additional_naming=additional_naming,
        )
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "ok", "variable_name": var_name})

