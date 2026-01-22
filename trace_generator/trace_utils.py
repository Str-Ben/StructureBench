#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities for stepwise trace generation with xgrammar masks."""
from __future__ import annotations

from collections import deque
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from common.text_model_utils import build_chat_text, get_model_runtime_device

try:
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        RepetitionPenaltyLogitsProcessor,
    )
    try:
        from transformers.generation.logits_process import FrequencyPenaltyLogitsProcessor
    except ImportError:  # pragma: no cover - optional
        FrequencyPenaltyLogitsProcessor = None
except Exception:  # pragma: no cover - optional
    LogitsProcessorList = None
    RepetitionPenaltyLogitsProcessor = None
    FrequencyPenaltyLogitsProcessor = None


def get_field(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def ensure_prompt_text(item: Any, idx: int) -> str:
    prompt_text = get_field(item, "prompt_text", None)
    if prompt_text is None:
        prompt_text = get_field(item, "prompt", None)
    if prompt_text is None:
        raise ValueError(
            f"Prompt missing for sample {idx}; expected 'prompt_text' or 'prompt'."
        )
    return str(prompt_text)


def _resolve_prompt_max_length(tokenizer) -> int:
    max_length = getattr(tokenizer, "model_max_length", None)
    if max_length is None or max_length > 1_000_000:
        max_length = getattr(tokenizer, "max_position_embeddings", None)
    if max_length is None or max_length > 1_000_000:
        max_length = 4096
    return int(max_length)


def _get_prompt_length(tokenizer, text: str) -> Optional[int]:
    try:
        encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    except Exception:
        return None
    input_ids = encoded.get("input_ids")
    if isinstance(input_ids, list):
        return len(input_ids)
    return None


def prepare_prompt_inputs(
    tokenizer,
    prompt: str,
    is_encoder_decoder: bool,
) -> Tuple[str, Dict[str, torch.Tensor], bool]:
    prompt_text = build_chat_text(tokenizer, prompt)
    max_length = _resolve_prompt_max_length(tokenizer)
    full_length = _get_prompt_length(tokenizer, prompt_text)
    if is_encoder_decoder:
        model_inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
    else:
        model_inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
    prompt_truncated = full_length is not None and full_length > max_length
    return prompt_text, model_inputs, prompt_truncated


def build_penalty_processors(
    repetition_penalty: float,
    frequency_penalty: float,
):
    if LogitsProcessorList is None:
        return None
    processors = []
    if (
        repetition_penalty != 1.0
        and RepetitionPenaltyLogitsProcessor is not None
    ):
        processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if (
        frequency_penalty != 0.0
        and FrequencyPenaltyLogitsProcessor is not None
    ):
        processors.append(FrequencyPenaltyLogitsProcessor(frequency_penalty))
    if not processors:
        return None
    return LogitsProcessorList(processors)


def apply_processors(processors, input_ids, logits):
    if processors is None:
        return logits
    return processors(input_ids, logits)


def build_xgrammar_processor(
    tokenizer,
    config,
    constraint_content: Optional[Any],
    constraint_type: Optional[str],
):
    if constraint_content is None or constraint_type is None:
        return None
    try:
        import xgrammar as xgr
    except ImportError as exc:  # pragma: no cover - optional
        raise ImportError("xgrammar not installed, please run: pip install xgrammar") from exc

    vocab_size = getattr(config, "vocab_size", None) or len(tokenizer)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=vocab_size,
    )
    compiler = xgr.GrammarCompiler(tokenizer_info)

    if constraint_type == "json":
        if isinstance(constraint_content, dict):
            json_schema = json.dumps(constraint_content, ensure_ascii=False)
        else:
            json_schema = str(constraint_content)
        grammar_obj = xgr.Grammar.from_json_schema(json_schema)
    elif constraint_type == "regex":
        grammar_obj = xgr.Grammar.from_regex(str(constraint_content))
    else:
        grammar_obj = constraint_content

    compiled_grammar = compiler.compile_grammar(grammar_obj)
    return _XGrammarLogitsProcessor(compiled_grammar, backend="torch_native")


class _XGrammarLogitsProcessor:
    """xgrammar LogitsProcessor that avoids Triton-only kernels."""

    def __init__(self, compiled_grammar, backend: str = "torch_native"):
        try:
            import xgrammar as xgr
        except ImportError as exc:  # pragma: no cover - optional
            raise ImportError(
                "xgrammar not installed, please run: pip install xgrammar"
            ) from exc
        import inspect

        self._xgr = xgr
        self.backend = backend
        try:
            self._supports_backend = (
                "backend" in inspect.signature(self._xgr.apply_token_bitmask_inplace).parameters
            )
        except Exception:
            self._supports_backend = False
        self.matchers: List[xgr.GrammarMatcher] = []
        self.compiled_grammars = (
            compiled_grammar if isinstance(compiled_grammar, list) else [compiled_grammar]
        )
        self._compiled_grammars_template = list(self.compiled_grammars)
        self.full_vocab_size = self.compiled_grammars[0].tokenizer_info.vocab_size
        self.token_bitmask = None
        self.prefilled = False
        self.batch_size = 0

    def reset(self) -> None:
        """Reset matcher state before tracing a new sample."""
        self.matchers = []
        self.token_bitmask = None
        self.prefilled = False
        self.batch_size = 0
        self.compiled_grammars = list(self._compiled_grammars_template)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if len(self.matchers) == 0:
            self.batch_size = input_ids.shape[0]
            self.compiled_grammars = (
                self.compiled_grammars
                if len(self.compiled_grammars) > 1
                else self.compiled_grammars * self.batch_size
            )
            if len(self.compiled_grammars) != self.batch_size:
                raise RuntimeError(
                    "The number of compiled grammars must match the batch size."
                )
            self.matchers = [
                self._xgr.GrammarMatcher(self.compiled_grammars[i])
                for i in range(self.batch_size)
            ]
            self.token_bitmask = self._xgr.allocate_token_bitmask(
                self.batch_size,
                self.full_vocab_size,
            )

        if input_ids.shape[0] != self.batch_size:
            raise RuntimeError(
                "Expect input_ids.shape[0] to match the logits processor batch size."
            )

        if not self.prefilled:
            self.prefilled = True
        else:
            for i in range(self.batch_size):
                if not self.matchers[i].is_terminated():
                    sampled_token = input_ids[i][-1]
                    assert self.matchers[i].accept_token(sampled_token)

        for i in range(self.batch_size):
            if not self.matchers[i].is_terminated():
                self.matchers[i].fill_next_token_bitmask(self.token_bitmask, i)

        scores_cpu = scores if scores.device.type == "cpu" else scores.to("cpu")
        bitmask = self.token_bitmask
        if self._supports_backend:
            self._xgr.apply_token_bitmask_inplace(
                scores_cpu,
                bitmask,
                backend=self.backend,
            )
        else:
            self._xgr.apply_token_bitmask_inplace(
                scores_cpu,
                bitmask,
            )
        return scores_cpu


def decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [int(token_id)],
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        return ""


def _build_token_record(tokenizer, token_id: int, logit: float) -> Dict[str, Any]:
    return {
        "id": int(token_id),
        "string": decode_token(tokenizer, token_id),
        "logit": float(logit),
    }


def _build_topk_records(
    tokenizer,
    token_ids: List[int],
    logits: List[float],
) -> List[Dict[str, Any]]:
    return [
        _build_token_record(tokenizer, token_id, logit)
        for token_id, logit in zip(token_ids, logits)
    ]


def _resolve_eos_token_ids(tokenizer, config) -> set[int]:
    eos_ids: set[int] = set()
    for source in (getattr(tokenizer, "eos_token_id", None), getattr(config, "eos_token_id", None)):
        if source is None:
            continue
        if isinstance(source, (list, tuple, set)):
            eos_ids.update(int(token_id) for token_id in source if token_id is not None)
        else:
            eos_ids.add(int(source))
    return eos_ids


def _resolve_decoder_start_token_id(tokenizer, config) -> Optional[int]:
    for source in (
        getattr(config, "decoder_start_token_id", None),
        getattr(config, "bos_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
    ):
        if source is not None:
            return int(source)
    return None


def _count_allowed(masked_logits: torch.Tensor) -> Tuple[int, float]:
    allowed_mask = torch.isfinite(masked_logits)
    allowed_count = int(allowed_mask.sum().item())
    vocab_size = int(masked_logits.shape[-1])
    ratio = float(allowed_count) / float(vocab_size) if vocab_size else 0.0
    return allowed_count, ratio

def compute_masked_step_stats(steps: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    total_steps = len(steps)
    masked_step_count = sum(
        1 for step in steps if step.get("case") == "masked"
    )
    masked_steps_ratio = (
        float(masked_step_count) / float(total_steps) if total_steps else 0.0
    )
    return masked_step_count, total_steps, masked_steps_ratio


def run_trace_for_prompt(
    model,
    tokenizer,
    config,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    frequency_penalty: float,
    grammar_processor: Optional[Any],
    repeat_window: int,
) -> Tuple[str, str, bool, List[Dict[str, Any]], Optional[str]]:
    is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
    prompt_text, model_inputs, prompt_truncated = prepare_prompt_inputs(
        tokenizer,
        prompt,
        is_encoder_decoder,
    )
    if grammar_processor is not None and hasattr(grammar_processor, "reset"):
        grammar_processor.reset()
    device = get_model_runtime_device(model)
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    penalty_processors = build_penalty_processors(
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
    )
    eos_token_ids = _resolve_eos_token_ids(tokenizer, config)
    repeat_history = (
        deque(maxlen=repeat_window) if repeat_window and repeat_window > 0 else None
    )

    steps: List[Dict[str, Any]] = []
    output_token_ids: List[int] = []
    error: Optional[str] = None

    def _apply_grammar(input_ids, logits):
        if grammar_processor is None:
            return logits
        return grammar_processor(input_ids, logits)

    # Greedy decode only; temperature/top_p are recorded but not used for sampling.
    with torch.inference_mode():
        if is_encoder_decoder:
            encoder_input_ids = model_inputs["input_ids"]
            encoder_attention_mask = model_inputs.get("attention_mask")
            encoder = model.get_encoder()
            encoder_outputs = encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
            )

            decoder_start_token_id = _resolve_decoder_start_token_id(tokenizer, config)
            if decoder_start_token_id is None:
                return prompt_text, "", prompt_truncated, [], "missing_decoder_start_token_id"

            decoder_input_ids = torch.tensor(
                [[decoder_start_token_id]],
                device=device,
                dtype=encoder_input_ids.dtype,
            )
            decoder_attention_mask = torch.ones_like(decoder_input_ids)
            for step_idx in range(max_new_tokens):
                outputs = model(
                    encoder_outputs=encoder_outputs,
                    attention_mask=encoder_attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    use_cache=False,
                )
                logits = outputs.logits[:, -1, :]

                pre_mask_logits = apply_processors(
                    penalty_processors,
                    decoder_input_ids,
                    logits,
                )
                topk = torch.topk(
                    pre_mask_logits,
                    k=min(5, pre_mask_logits.shape[-1]),
                    dim=-1,
                )
                top_ids = topk.indices[0].tolist()
                top_vals = topk.values[0].tolist()
                pre_top1_id = int(top_ids[0])
                pre_top1_logit = float(top_vals[0])
                logit_gap = (
                    float(top_vals[0] - top_vals[1]) if len(top_vals) > 1 else None
                )

                masked_logits = _apply_grammar(decoder_input_ids, pre_mask_logits)
                allowed_count, allowed_ratio = _count_allowed(masked_logits)
                if allowed_count == 0:
                    error = "all_tokens_masked"
                    break

                post_token_id = int(torch.argmax(masked_logits, dim=-1).item())
                post_token_logit = float(masked_logits[0, post_token_id].item())
                is_eos = post_token_id in eos_token_ids if eos_token_ids else False
                repeat_in_window = False
                if repeat_history is not None:
                    repeat_in_window = post_token_id in repeat_history
                    repeat_history.append(post_token_id)

                top1_allowed = True
                if grammar_processor is not None:
                    top1_allowed = bool(torch.isfinite(masked_logits[0, pre_top1_id]).item())
                case = "same" if top1_allowed else "masked"

                step_entry: Dict[str, Any] = {
                    "step_idx": step_idx,
                    "case": case,
                    "repeat_in_window": repeat_in_window,
                    "is_eos": is_eos,
                    "pre_mask_top1": _build_token_record(
                        tokenizer,
                        pre_top1_id,
                        pre_top1_logit,
                    ),
                    "logit_gap_top1_top2": logit_gap,
                }
                if case == "masked":
                    step_entry["pre_mask_top5"] = _build_topk_records(
                        tokenizer,
                        top_ids,
                        top_vals,
                    )
                    step_entry["post_mask_token"] = _build_token_record(
                        tokenizer,
                        post_token_id,
                        post_token_logit,
                    )
                    step_entry["allowed_token_count"] = allowed_count
                    step_entry["allowed_ratio"] = allowed_ratio

                steps.append(step_entry)
                output_token_ids.append(post_token_id)

                new_token = torch.tensor([[post_token_id]], device=device, dtype=decoder_input_ids.dtype)
                decoder_input_ids = torch.cat([decoder_input_ids, new_token], dim=1)
                decoder_attention_mask = torch.cat(
                    [decoder_attention_mask, torch.ones_like(new_token)],
                    dim=1,
                )

                if is_eos:
                    break
        else:
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs.get("attention_mask")
            for step_idx in range(max_new_tokens):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = outputs.logits[:, -1, :]

                pre_mask_logits = apply_processors(
                    penalty_processors,
                    input_ids,
                    logits,
                )
                topk = torch.topk(
                    pre_mask_logits,
                    k=min(5, pre_mask_logits.shape[-1]),
                    dim=-1,
                )
                top_ids = topk.indices[0].tolist()
                top_vals = topk.values[0].tolist()
                pre_top1_id = int(top_ids[0])
                pre_top1_logit = float(top_vals[0])
                logit_gap = (
                    float(top_vals[0] - top_vals[1]) if len(top_vals) > 1 else None
                )

                masked_logits = _apply_grammar(input_ids, pre_mask_logits)
                allowed_count, allowed_ratio = _count_allowed(masked_logits)
                if allowed_count == 0:
                    error = "all_tokens_masked"
                    break

                post_token_id = int(torch.argmax(masked_logits, dim=-1).item())
                post_token_logit = float(masked_logits[0, post_token_id].item())
                is_eos = post_token_id in eos_token_ids if eos_token_ids else False
                repeat_in_window = False
                if repeat_history is not None:
                    repeat_in_window = post_token_id in repeat_history
                    repeat_history.append(post_token_id)

                top1_allowed = True
                if grammar_processor is not None:
                    top1_allowed = bool(torch.isfinite(masked_logits[0, pre_top1_id]).item())
                case = "same" if top1_allowed else "masked"

                step_entry = {
                    "step_idx": step_idx,
                    "case": case,
                    "repeat_in_window": repeat_in_window,
                    "is_eos": is_eos,
                    "pre_mask_top1": _build_token_record(
                        tokenizer,
                        pre_top1_id,
                        pre_top1_logit,
                    ),
                    "logit_gap_top1_top2": logit_gap,
                }
                if case == "masked":
                    step_entry["pre_mask_top5"] = _build_topk_records(
                        tokenizer,
                        top_ids,
                        top_vals,
                    )
                    step_entry["post_mask_token"] = _build_token_record(
                        tokenizer,
                        post_token_id,
                        post_token_logit,
                    )
                    step_entry["allowed_token_count"] = allowed_count
                    step_entry["allowed_ratio"] = allowed_ratio

                steps.append(step_entry)
                output_token_ids.append(post_token_id)

                new_token = torch.tensor([[post_token_id]], device=device, dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, new_token], dim=1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(new_token, dtype=attention_mask.dtype)],
                        dim=1,
                    )

                if is_eos:
                    break

    output_text = tokenizer.decode(
        output_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return prompt_text, output_text, prompt_truncated, steps, error
