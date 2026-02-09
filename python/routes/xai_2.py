"""
XAI Level 2 API handlers: Internal Mechanism Analysis.
"""

from typing import Any, Dict

from python.xai_2.residual_concept_detection import run_residual_concept_detection


def run_residual_concept(
    *,
    model: str,
    input_setting: Dict[str, Any],
    load_dataset_fn,
    progress_callback=None,
) -> tuple[Dict[str, Any], int]:
    """
    Run Residual Concept Detection (2.0.1).
    input_setting: variable_name, text_key, label_key, positive_label, negative_label,
                   layer_base, attn_name, mlp_name, o_proj_name, down_proj_name,
                   token_location ("full" or "0,1,2"), batch_size.
    """
    var_name = (input_setting.get("variable_name") or "").strip()
    if not var_name:
        return ({"error": "variable_name required (saved dataset variable)"}, 400)

    from python.memory.variable import variable_store
    from python.dataset_pipeline_store import get_pipeline_by_id

    meta = variable_store.get_meta(var_name)
    if not meta:
        return ({"error": f"Variable not found: {var_name!r}"}, 404)
    pipeline_id = meta.pipeline_id
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        return ({"error": f"Pipeline not found: {pipeline_id}"}, 404)

    text_key = (input_setting.get("text_key") or "").strip()
    label_key = (input_setting.get("label_key") or "").strip()
    positive_label = (input_setting.get("positive_label") or "").strip()
    negative_label = (input_setting.get("negative_label") or "").strip()
    if not all([text_key, label_key, positive_label, negative_label]):
        return ({"error": "text_key, label_key, positive_label, negative_label required"}, 400)

    layer_base = (input_setting.get("layer_base") or "").strip()
    attn_name = (input_setting.get("attn_name") or "self_attn").strip()
    mlp_name = (input_setting.get("mlp_name") or "mlp").strip()
    o_proj_name = (input_setting.get("o_proj_name") or "o_proj").strip()
    down_proj_name = (input_setting.get("down_proj_name") or "down_proj").strip()
    if not layer_base:
        return ({"error": "layer_base required. Hover over Loaded Model in the left sidebar to configure Layers, attn, mlp, o_proj, down_proj."}, 400)

    layer_config = {
        "layer_base": layer_base,
        "attn_name": attn_name,
        "mlp_name": mlp_name,
        "o_proj_name": o_proj_name,
        "down_proj_name": down_proj_name,
    }

    token_location = input_setting.get("token_location") or "full"

    # Batch size
    raw_bs = input_setting.get("batch_size")
    try:
        batch_size = int(raw_bs) if raw_bs is not None and str(raw_bs).strip() != "" else 8
    except (TypeError, ValueError):
        batch_size = 8
    if batch_size <= 0:
        batch_size = 8

    try:
        ds, _ = load_dataset_fn(pipeline)
    except Exception as e:
        return ({"error": f"Failed to load dataset: {e}"}, 500)

    return run_residual_concept_detection(
        model_key=model,
        ds=ds,
        pipeline=pipeline,
        text_key=text_key,
        label_key=label_key,
        positive_label=positive_label,
        negative_label=negative_label,
        layer_config=layer_config,
        token_location=token_location,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )


def run_placeholder(*, model: str, treatment: str, input_setting: Dict[str, Any]) -> tuple[Dict[str, Any], int]:
    """
    Placeholder for level 2+ XAI analysis (non-2.0.1).
    Returns (result_dict, status_code).
    """
    return (
        {
            "status": "ok",
            "message": "Analysis complete. (Actual XAI computation to be implemented)",
            "model": model,
            "treatment": treatment,
            "input_setting": input_setting,
        },
        200,
    )
