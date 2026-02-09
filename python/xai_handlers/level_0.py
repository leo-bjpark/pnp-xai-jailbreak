"""
XAI Level 0 API handlers: Completion, Conversation.
"""

from typing import Any, Dict, List, Optional

from python.xai_0.model_generation import (
    chat_completion,
    get_cache_token_count,
    get_text_token_count,
    text_completion,
)


def run_conversation(
    *,
    model: str,
    treatment: str,
    current_model: str,
    messages_input: List[Dict],
    system_instruction: str,
    input_setting: Dict[str, Any],
) -> tuple[Dict[str, Any], int]:
    """
    Handle multi-turn conversation (0.1.2).
    messages_input: JS에서 관리하는 전체 대화 내역.
    Returns (result_dict, status_code). On error, result_dict has "error" key.
    """
    try:
        temperature = float(input_setting.get("temperature", 0.7))
        max_new_tokens = int(input_setting.get("max_new_tokens", 256))
        top_p = float(input_setting.get("top_p", 1.0))
        top_k = int(input_setting.get("top_k", 50))
    except (TypeError, ValueError):
        return ({"error": "Invalid input_setting: temperature, max_new_tokens, top_p, top_k must be numbers"}, 400)

    def _ensure_system_at_start(messages: list) -> list:
        if not system_instruction:
            return messages
        rest = messages
        while rest and (rest[0].get("role") or "").lower() == "system":
            rest = rest[1:]
        return [{"role": "system", "content": system_instruction}] + rest

    if not messages_input or not isinstance(messages_input, list):
        return ({"error": "messages must be a non-empty list (managed by JS)"}, 400)

    messages = [
        {"role": m.get("role", "user"), "content": (m.get("content") or "").strip()}
        for m in messages_input
        if (m.get("content") or "").strip()
    ]
    if not messages:
        return ({"error": "messages must contain at least one non-empty message"}, 400)
    messages = _ensure_system_at_start(messages)

    try:
        input_tokens = get_cache_token_count(current_model, messages)
        do_sample = temperature > 0
        generated_text = chat_completion(
            model_key=current_model,
            messages=messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        generated_tokens = get_text_token_count(current_model, generated_text)
        messages.append({"role": "assistant", "content": generated_text})

        conversation_list = {
            "instruction": system_instruction,
            "messages": [
                {"role": "user" if m.get("role") == "user" else "ai", "content": (m.get("content") or "").strip()}
                for m in messages
                if (m.get("role") or "").lower() in ("user", "assistant")
            ],
        }

        result = {
            "status": "ok",
            "model": model,
            "treatment": treatment,
            "generated_text": generated_text,
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "cache_message_count": len(messages),
            "conversation_list": conversation_list,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
        }
        return (result, 200)
    except Exception as e:
        return ({"error": str(e)}, 500)


def run_completion(
    *,
    model: str,
    treatment: str,
    current_model: str,
    input_string: str,
    input_setting: Dict[str, Any],
) -> tuple[Dict[str, Any], int]:
    """
    Handle text completion (0.1.1).
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
        generated_text = text_completion(
            model_key=current_model,
            prompt=input_string,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
        )
        result = {
            "status": "ok",
            "model": model,
            "treatment": treatment,
            "input_string": input_string,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "generated_text": generated_text,
        }
        return (result, 200)
    except Exception as e:
        return ({"error": str(e)}, 500)
