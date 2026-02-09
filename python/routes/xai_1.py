"""
XAI Level 1 API handlers: Response Attribution.
"""

from typing import Any, Dict


def run_attribution(
    *,
    model: str,
    treatment: str,
    current_model: str,
    input_string: str,
    system_instruction: str,
    attribution_method: str,
    input_setting: Dict[str, Any],
) -> tuple[Dict[str, Any], int]:
    """
    Handle response attribution (1.0.1).
    Returns (result_dict, status_code).
    """
    try:
        temperature = float(input_setting.get("temperature", 0.7))
        max_new_tokens = int(input_setting.get("max_new_tokens", 256))
        top_p = float(input_setting.get("top_p", 1.0))
        top_k = int(input_setting.get("top_k", 50))
    except (TypeError, ValueError):
        return ({"error": "Invalid input_setting: temperature, max_new_tokens, top_p, top_k must be numbers"}, 400)

    try:
        from python.xai_1.input_attribution import compute_input_attribution

        result = compute_input_attribution(
            model_key=current_model,
            input_string=input_string,
            system_instruction=system_instruction,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            attribution_method=attribution_method,
        )
        result["status"] = "ok"
        result["model"] = model
        result["treatment"] = treatment
        return (result, 200)
    except Exception as e:
        return ({"error": str(e)}, 500)
