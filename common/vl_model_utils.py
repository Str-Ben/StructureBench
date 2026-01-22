#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for loading vision-language models (Qwen-VL, MiniCPM-V, etc.)
and running chat-style generation with image+text messages.
"""

import inspect
from typing import Any, Dict, List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from common import text_model_utils as _text_utils
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


def _fallback_resolve_device_and_dtype(device: str = "cuda", torch_dtype: str = "auto"):
    device_key = device.lower()
    device_str = "cuda" if device_key.startswith("cuda") and torch.cuda.is_available() else "cpu"
    return device_str, torch.float16


def _fallback_get_model_runtime_device(model) -> torch.device:
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


def _fallback_get_model_runtime_dtype(model) -> torch.dtype:
    for param in model.parameters():
        if param.dtype is not None:
            return param.dtype
    return torch.float16


resolve_device_and_dtype = getattr(
    _text_utils, "resolve_device_and_dtype", _fallback_resolve_device_and_dtype
)
get_model_runtime_device = getattr(
    _text_utils, "get_model_runtime_device", _fallback_get_model_runtime_device
)
get_model_runtime_dtype = getattr(
    _text_utils, "get_model_runtime_dtype", _fallback_get_model_runtime_dtype
)


def load_vl_model_and_processor(
    model_name_or_path: str,
    device: str = "cuda",
    torch_dtype: str = "auto",
    trust_remote_code: bool = True,
):
    """Load a VLM and its processor with sensible defaults."""
    device_str, dtype = resolve_device_and_dtype(device, torch_dtype)

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    model_type = getattr(config, "model_type", "").lower()
    architectures = [arch.lower() for arch in (getattr(config, "architectures", None) or [])]

    is_minicpmv = model_type == "minicpmv" or any("minicpmv" in arch for arch in architectures)
    use_device_map = not is_minicpmv and device_str != "cpu"

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if use_device_map:
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = None
        model_kwargs["low_cpu_mem_usage"] = False

    model = None
    is_qwen3_vl = model_type == "qwen3_vl" or any(
        "qwen3vlforconditionalgeneration" in arch for arch in architectures
    )
    if is_qwen3_vl:
        load_errors = []
        try:
            from transformers import Qwen3VLForConditionalGeneration  # type: ignore

            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                **model_kwargs,
            )
        except Exception as exc:
            load_errors.append(("Qwen3VLForConditionalGeneration", exc))
            model = None

        if model is None:
            try:
                from transformers import AutoModelForVision2Seq  # type: ignore

                model = AutoModelForVision2Seq.from_pretrained(
                    model_name_or_path,
                    **model_kwargs,
                )
            except Exception as exc:
                load_errors.append(("AutoModelForVision2Seq", exc))
                model = None

        if model is None:
            error_details = "; ".join(
                f"{name}: {type(err).__name__}: {err}" for name, err in load_errors
            )
            raise RuntimeError(
                "Failed to load Qwen3-VL model. "
                "Please ensure `transformers` is up to date and Qwen3-VL classes are available. "
                f"Errors: {error_details}"
            ) from load_errors[-1][1]

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )

    if not use_device_map:
        model.to(device_str)

    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=False,
    )

    return model, processor


def build_vl_inputs(messages: List[Dict[str, Any]], processor, model):
    """
    Construct tensor inputs for VLM chat templates.
    If the processor lacks `apply_chat_template` (e.g., MiniCPM-V 2.6),
    fall back to calling the processor directly.
    """
    cls_name = processor.__class__.__name__.lower()

    # MiniCPM-V 系列没有 chat_template，需要手动拆出图片与文本
    if "minicpmv" in cls_name:
        image = None
        text_parts = []
        for msg in messages:
            for item in msg.get("content", []):
                if item.get("type") == "image":
                    image = item.get("image")
                elif item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
        text = "\n\n".join([t for t in text_parts if t is not None])
        if "<image" not in text:
            text = "<image>./</image>\n" + text
        if image is None:
            raise ValueError("MiniCPM-V inputs require an image in messages.")
        result = processor(
            text=text,
            images=image,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(result, "data"):
            return result.data
        if isinstance(result, dict):
            return result
        raise TypeError(f"Unexpected MiniCPM-V processor output type: {type(result)}")

    if hasattr(processor, "apply_chat_template") and getattr(processor, "chat_template", None):
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    # Fallback path for processors without chat template (MiniCPM-V)
    return processor(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )


def generate_vl(
    model,
    processor,
    inputs: Dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
):
    """Run generation and decode new tokens."""
    device = get_model_runtime_device(model)
    model_dtype = get_model_runtime_dtype(model)
    def _move(obj):
        if hasattr(obj, "to"):
            return obj.to(
                device=device,
                dtype=model_dtype if hasattr(obj, "dtype") and obj.dtype.is_floating_point else None,
            )
        if isinstance(obj, list):
            return [_move(o) for o in obj]
        return obj
    # Normalize inputs from different processors
    if isinstance(inputs, list):
        inputs = inputs[0] if len(inputs) == 1 else inputs
    if hasattr(inputs, "data"):
        inputs = inputs.data
    normalized_inputs = {}
    list_as_batch_keys = {"pixel_values", "image_sizes", "image_bound", "tgt_sizes"}
    for k, v in inputs.items():
        if k in list_as_batch_keys:
            normalized_inputs[k] = _move(v)
            continue
        # Collapse nested single-item lists
        while isinstance(v, list) and len(v) == 1:
            v = v[0]
        # If still a list of tensors, stack them
        if isinstance(v, list):
            v = torch.stack(v)
        normalized_inputs[k] = _move(v)
    inputs = normalized_inputs
    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not accepts_var_kw:
            allowed = {name for name in sig.parameters if name != "self"}
            inputs = {k: v for k, v in inputs.items() if k in allowed}

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
    }
    if supports_repetition:
        generation_kwargs["repetition_penalty"] = repetition_penalty
    if supports_frequency:
        generation_kwargs["frequency_penalty"] = frequency_penalty
    if logits_processors is not None:
        generation_kwargs["logits_processor"] = logits_processors
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        if getattr(tokenizer, "pad_token_id", None) is not None:
            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
        if getattr(tokenizer, "eos_token_id", None) is not None:
            generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )

    generated_ids = outputs[0]
    input_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[input_length:]
    return processor.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def generate_minicpm_chat(
    model,
    processor,
    image,
    text_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
):
    """
    MiniCPM-V 提供了 chat 接口，直接用模型内部对齐逻辑推理。
    """
    sampling = temperature > 0.0
    msgs = [{"role": "user", "content": text_prompt}]
    image_input = image
    if isinstance(image, list) and len(image) == 1:
        image_input = image[0]
    outputs = model.chat(
        image=image_input,
        msgs=msgs,
        tokenizer=getattr(processor, "tokenizer", None),
        processor=processor,
        max_new_tokens=max_new_tokens,
        sampling=sampling,
        top_p=top_p,
        temperature=temperature if sampling else None,
    )
    if isinstance(outputs, list):
        return outputs[0]
    return outputs
