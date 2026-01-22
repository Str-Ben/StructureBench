#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shared runner for text-only generation tasks.
Each dataset script provides minimal callbacks; this module handles model I/O and generation loop.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from common.text_model_utils import (
    build_model_inputs,
    generate,
    load_text_model_and_tokenizer,
)


@dataclass
class TextTaskConfig:
    model_name_or_path: str
    device: str = "cuda"
    trust_remote_code: bool = True
    torch_dtype: str = "float16"
    attn_implementation: Optional[str] = None
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0


def run_text_task(
    data_iter: Iterable[Dict[str, Any]],
    build_prompt_fn: Callable[[Dict[str, Any]], str],
    postprocess_fn: Optional[Callable[[str, Dict[str, Any]], None]],
    task_config: TextTaskConfig,
):
    """
    Generic per-sample generation loop.
    - data_iter: iterable of dataset rows (already truncated to num_samples)
    - build_prompt_fn(sample) -> prompt string
    - postprocess_fn(output_text, sample): optional hook for evaluation/logging
    """
    model, tokenizer, config = load_text_model_and_tokenizer(
        model_name_or_path=task_config.model_name_or_path,
        device=task_config.device,
        trust_remote_code=task_config.trust_remote_code,
        torch_dtype=task_config.torch_dtype,
        attn_implementation=task_config.attn_implementation,
    )
    is_encoder_decoder = getattr(config, "is_encoder_decoder", False)

    for idx, sample in enumerate(data_iter):
        prompt = build_prompt_fn(sample)

        model_inputs = build_model_inputs(
            tokenizer,
            prompt,
            is_encoder_decoder=is_encoder_decoder,
        )

        output_text = generate(
            model,
            tokenizer,
            model_inputs,
            is_encoder_decoder=is_encoder_decoder,
            max_new_tokens=task_config.max_new_tokens,
            temperature=task_config.temperature,
            top_p=task_config.top_p,
        )

        if postprocess_fn:
            postprocess_fn(output_text, sample)
        else:
            print(f"[Sample {idx}]")
            print(output_text.strip())
