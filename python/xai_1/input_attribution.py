"""
Input attribution for response: which input tokens contributed to the generated output.
Uses input_grad: gradient of (generated output) w.r.t. input embeddings, then abs and normalize.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from python.model_load import get_model_device, load_llm
from python.model_generation import text_completion


def _get_input_tokens_and_scores_input_grad(
    model_key: str,
    input_string: str,
    generated_text: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
) -> Tuple[List[str], List[float], torch.Tensor]:
    """
    Compute per-input-token attribution via gradient of generated output log-prob w.r.t. input embeddings.
    Returns (token_strings, normalized_scores_0_1, output_ids_used).
    """
    tokenizer, model = load_llm(model_key)
    device = get_model_device(model)
    model.eval()

    # Tokenize prompt (same as text_completion)
    enc = tokenizer(input_string.strip(), return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    prompt_length = input_ids.shape[1]

    # Get generated token ids: run generation again to have exact ids (deterministic if temperature=0)
    do_sample = temperature > 0
    gen_kw: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kw["temperature"] = temperature
        if top_p < 1.0:
            gen_kw["top_p"] = top_p
        if top_k > 0:
            gen_kw["top_k"] = top_k

    with torch.no_grad():
        out = model.generate(input_ids, **gen_kw)
    output_ids = out[0][prompt_length:].unsqueeze(0)  # [1, gen_len]
    gen_len = output_ids.shape[1]
    if gen_len == 0:
        # No generated tokens: return uniform or zero scores
        token_strs = tokenizer.convert_ids_to_tokens(input_ids[0])
        return token_strs, [0.0] * len(token_strs), output_ids

    # Input embeddings with gradient
    embed_layer = model.get_input_embeddings()
    prompt_embeds = embed_layer(input_ids).detach().clone().requires_grad_(True)
    output_embeds = embed_layer(output_ids)  # [1, gen_len, hidden]
    full_embeds = torch.cat([prompt_embeds, output_embeds], dim=1)  # [1, prompt_len + gen_len, hidden]

    seq_len = full_embeds.shape[1]
    attention_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)

    # Forward to get logits; position i predicts token at i+1
    outputs = model(inputs_embeds=full_embeds, attention_mask=attention_mask)
    logits = outputs.logits  # [1, seq_len, vocab_size]

    # Log-prob of generated tokens: positions [prompt_len-1, ..., prompt_len+gen_len-2] predict [prompt_len, ..., prompt_len+gen_len-1]
    logits_for_gen = logits[0, prompt_length - 1 : prompt_length - 1 + gen_len]  # [gen_len, vocab_size]
    target = output_ids[0]  # [gen_len]
    loss = -F.cross_entropy(logits_for_gen, target, reduction="sum")
    loss.backward()

    # Gradient w.r.t. prompt embeddings: [1, prompt_len, hidden]
    grad = prompt_embeds.grad
    if grad is None:
        token_strs = tokenizer.convert_ids_to_tokens(input_ids[0])
        return token_strs, [0.0] * len(token_strs), output_ids

    # Per-token importance: sum of absolute gradient over hidden dim
    importance = grad.abs().sum(dim=-1).squeeze(0)  # [prompt_len]
    importance = importance.detach().cpu().float()

    # Normalize to 0--1
    min_val = importance.min().item()
    max_val = importance.max().item()
    if max_val > min_val:
        scores = ((importance - min_val) / (max_val - min_val)).tolist()
    else:
        scores = [0.0] * importance.shape[0]

    token_strs = tokenizer.convert_ids_to_tokens(input_ids[0])
    return token_strs, scores, output_ids


def compute_input_attribution(
    model_key: str,
    input_string: str,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    top_p: float = 1.0,
    top_k: int = 50,
    attribution_method: str = "input_grad",
) -> Dict[str, Any]:
    """
    Run completion then compute input token attribution.

    Returns:
        generated_text: str
        input_tokens: list of token strings (from tokenizer)
        token_scores: list of float in [0, 1], same length as input_tokens
        attribution_method: str
        (and same generation params as in result)
    """
    # 1) Generate
    generated_text = text_completion(
        model_key=model_key,
        prompt=input_string,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
    )

    if not attribution_method or attribution_method.lower() == "none":
        tokenizer, _ = load_llm(model_key)
        enc = tokenizer(input_string.strip(), return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"]
        token_strs = tokenizer.convert_ids_to_tokens(input_ids[0])
        return {
            "generated_text": generated_text,
            "input_tokens": token_strs,
            "token_scores": [0.0] * len(token_strs),
            "attribution_method": "none",
            "input_string": input_string.strip(),
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
        }

    if attribution_method.lower() == "input_grad":
        token_strs, scores, _ = _get_input_tokens_and_scores_input_grad(
            model_key=model_key,
            input_string=input_string,
            generated_text=generated_text,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
        )
        return {
            "generated_text": generated_text,
            "input_tokens": token_strs,
            "token_scores": scores,
            "attribution_method": "input_grad",
            "input_string": input_string.strip(),
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
        }

    raise ValueError(f"Unknown attribution_method: {attribution_method}")
