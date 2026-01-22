#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shared utilities for loading text-only transformers models and running generation.
Used by dataset scripts to keep a consistent CLI and behavior across models.
"""

from typing import Any, Dict, Iterable, List, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from transformers import cache_utils

# Some community models (e.g., MiniCPM) expect newer cache utils symbols; provide shims if missing.
if not hasattr(cache_utils, "CacheLayerMixin"):  # pragma: no cover - environment guard
    class CacheLayerMixin:  # type: ignore
        pass
    cache_utils.CacheLayerMixin = CacheLayerMixin
if not hasattr(cache_utils, "DynamicLayer"):  # pragma: no cover - environment guard
    class DynamicLayer:  # type: ignore
        pass
    cache_utils.DynamicLayer = DynamicLayer

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


def resolve_device_and_dtype(device: str = "cuda", torch_dtype: str = "auto") -> Tuple[str, torch.dtype]:
    """
    Return a device string and dtype.
    The dtype is forced to fp16 to keep all pipelines consistent.
    """
    device_key = device.lower()
    device_str = "cuda" if device_key.startswith("cuda") and torch.cuda.is_available() else "cpu"
    return device_str, torch.float16


def get_model_runtime_device(model) -> torch.device:
    """Pick a real device for running inputs, working with sharded/device_map models."""
    device = None
    try:
        device = model.device
    except Exception:
        device = None
    if isinstance(device, torch.device) and device.type != "meta":
        return device
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_model_runtime_dtype(model) -> torch.dtype:
    """Infer model dtype with a safe fallback."""
    for param in model.parameters():
        if param.dtype is not None:
            return param.dtype
    return torch.float16


def load_text_model_and_tokenizer(
    model_name_or_path: str,
    device: str = "cuda",
    trust_remote_code: bool = True,
    torch_dtype: str = "auto",
    attn_implementation: str = None,
):
    """
    Load a causal LM or encoder-decoder LM with a tokenizer.
    Handles pad/eos token setup and dtype/device selection.
    """
    _, dtype = resolve_device_and_dtype(device, torch_dtype)

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    is_encoder_decoder = getattr(config, "is_encoder_decoder", False)
    model_cls = AutoModelForSeq2SeqLM if is_encoder_decoder else AutoModelForCausalLM

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        if isinstance(tokenizer, bool):
            tokenizer = None
    except Exception:
        tokenizer = None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            trust_remote_code=trust_remote_code,
        )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
        "device_map": "auto",
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = model_cls.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    model.eval()

    return model, tokenizer, config


def build_chat_text(tokenizer, prompt: str) -> str:
    """Use chat template when available to wrap the user prompt."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except ValueError:
            # Some base models expose apply_chat_template but lack a template; fall back to plain prompt.
            return prompt
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
    return prompt


def build_model_inputs(tokenizer, prompt: str, is_encoder_decoder: bool):
    """Tokenize prompt for either causal or encoder-decoder models."""
    text = build_chat_text(tokenizer, prompt)
    max_length = getattr(tokenizer, "model_max_length", None)
    if max_length is None or max_length > 1000000:
        max_length = getattr(tokenizer, "max_position_embeddings", None)
    if max_length is None or max_length > 1000000:
        max_length = 4096
    if is_encoder_decoder:
        return tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )


def generate(
    model,
    tokenizer,
    model_inputs: Dict[str, torch.Tensor],
    is_encoder_decoder: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
) -> str:
    """Run generation and strip the prompt part."""
    target_device = get_model_runtime_device(model)
    inputs = {k: v.to(target_device) for k, v in model_inputs.items()}

    supports_repetition = hasattr(model.generation_config, "repetition_penalty")
    supports_frequency = hasattr(model.generation_config, "frequency_penalty")

    logits_processors = None
    if LogitsProcessorList is not None:
        processors = []
        if (
            not supports_repetition
            and repetition_penalty != 1.0
            and RepetitionPenaltyLogitsProcessor is not None
        ):
            processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if (
            not supports_frequency
            and frequency_penalty != 0.0
            and FrequencyPenaltyLogitsProcessor is not None
        ):
            processors.append(FrequencyPenaltyLogitsProcessor(frequency_penalty))
        if processors:
            logits_processors = LogitsProcessorList(processors)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": (temperature > 0.0),
        "temperature": temperature if temperature > 0.0 else None,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if supports_repetition:
        generation_kwargs["repetition_penalty"] = repetition_penalty
    if supports_frequency:
        generation_kwargs["frequency_penalty"] = frequency_penalty
    if logits_processors is not None:
        generation_kwargs["logits_processor"] = logits_processors

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )

    generated_ids = outputs[0]
    input_ids = inputs["input_ids"][0]
    if generated_ids.shape[0] >= input_ids.shape[0]:
        new_tokens = generated_ids[input_ids.shape[0]:]
    else:
        new_tokens = generated_ids
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def add_common_model_args(parser):
    """Attach shared CLI arguments for model loading and generation."""
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="本地模型路径或 HuggingFace Hub 模型名，例如 'Models/MiniCPM4-0.5B/MiniCPM4-0.5B'。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备：'cuda' 或 'cpu'。",
    )
    parser.add_argument(
        "--no_trust_remote_code",
        action="store_true",
        help="如果不希望使用 trust_remote_code，可加此参数。",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        help="模型精度强制为 float16（参数值仅为兼容旧接口，不再生效）。",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="注意力实现（如 flash_attention_2/sdpa），部分模型需要显式指定。",
    )
    return parser
