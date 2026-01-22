#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""vl_constrained_genenration.py

- outlines 对 VLM 的输入更偏“高层”（prompt + Image/Chat），本封装支持通过 kwargs 传入 prompt_text/image。

"""

import importlib
import inspect
import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image as PILImage
from transformers import AutoConfig, LogitsProcessorList

logger = logging.getLogger(__name__)


def _get_model_runtime_device(model) -> torch.device:
    """尽量从 model 上拿到实际运行 device（兼容 device_map/meta tensor）。"""
    try:
        device = model.device
        if isinstance(device, torch.device) and device.type != "meta":
            return device
    except Exception:
        pass
    try:
        for p in model.parameters():
            if p.device.type != "meta":
                return p.device
    except Exception:
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_model_runtime_dtype(model) -> torch.dtype:
    try:
        for p in model.parameters():
            if p.dtype is not None:
                return p.dtype
    except Exception:
        pass
    return torch.float16


def _normalize_and_move_vl_inputs(model, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Reference common.vl_model_utils.generate_vl's input normalization and move logic."""
    device = _get_model_runtime_device(model)
    model_dtype = _get_model_runtime_dtype(model)

    def _move(obj):
        if hasattr(obj, "to"):
            # 仅对浮点 tensor cast dtype
            try:
                if hasattr(obj, "dtype") and getattr(obj.dtype, "is_floating_point", False):
                    return obj.to(device=device, dtype=model_dtype)
            except Exception:
                pass
            return obj.to(device=device)
        if isinstance(obj, list):
            return [_move(o) for o in obj]
        return obj

    if inputs is None:
        return {}

    if isinstance(inputs, list):
        inputs = inputs[0] if len(inputs) == 1 else inputs

    if hasattr(inputs, "data"):
        inputs = inputs.data  # type: ignore[assignment]

    if not isinstance(inputs, dict):
        raise TypeError(f"VLM inputs must be dict-like, got: {type(inputs)}")

    normalized: Dict[str, Any] = {}
    list_as_batch_keys = {"pixel_values", "image_sizes", "image_bound", "tgt_sizes"}

    for k, v in inputs.items():
        if k in list_as_batch_keys:
            normalized[k] = _move(v)
            continue

        while isinstance(v, list) and len(v) == 1:
            v = v[0]

        if isinstance(v, list):
            try:
                v = torch.stack(v)
            except Exception:
                pass

        normalized[k] = _move(v)

    return normalized


def _safe_add_frequency_penalty_kwargs(model, gen_kwargs: Dict[str, Any], frequency_penalty: float):
    """Only pass frequency_penalty if model.generate explicitly supports it (to avoid TypeError)."""
    if frequency_penalty == 0.0:
        return
    try:
        sig = inspect.signature(model.generate)
        if "frequency_penalty" in sig.parameters:
            gen_kwargs["frequency_penalty"] = frequency_penalty
    except Exception:
        return


def _prepare_outlines_vlm_input(
    processor: Any,
    prompt_text: Any,
    image: Any,
):
    """将 (prompt_text, image) 归一化为 Outlines 可识别的 VLM 输入。

    - 统一保证 image 为 PIL.Image.Image, RGB, 且设置 image.format
    返回：outlines_input(list)
        [prompt_text(str), outlines.inputs.Image(PIL.Image)]
    """
    if not isinstance(prompt_text, str):
        raise TypeError(f"prompt_text must be str, actual: {type(prompt_text)}")
    if image is None:
        raise TypeError("image cannot be empty")

    if not isinstance(image, PILImage.Image):
        raise TypeError(f"image must be PIL.Image.Image, actual: {type(image)}")
    if image.mode != "RGB":
        image = image.convert("RGB")
    if getattr(image, "format", None) is None:
        image.format = "PNG"

    try:
        from outlines.inputs import Image as OutlinesImage

        return [prompt_text, OutlinesImage(image)]
    except Exception as e:
        raise TypeError(f"Cannot convert (prompt_text, image) to Outlines multimodal input: {e}")


class GenerationMode(str, Enum):
    PROMPT = "prompt"
    XGRAMMAR = "xgrammar"
    GUIDANCE = "guidance"
    OUTLINES = "outlines"
    LLAMA_CPP = "llama_cpp"


class VLConstrainedGeneratorBase(ABC):
    def __init__(self, model: Any, processor: Any = None, tokenizer: Any = None, **kwargs):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.config = kwargs

    @abstractmethod
    def generate(self, inputs: Dict[str, Any], **kwargs) -> str:
        raise NotImplementedError


class PromptOnlyVLGenerator(VLConstrainedGeneratorBase):
    def generate(
        self,
        inputs: Dict[str, Any],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        **kwargs,
    ) -> str:
        if self.processor is None:
            raise ValueError("PromptOnlyVLGenerator requires `processor` for decoding.")

        moved_inputs = _normalize_and_move_vl_inputs(self.model, inputs)
        
        #print(kwargs)
        gen_kwargs: Dict[str, Any] = {
            **moved_inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": (temperature > 0.0),
            "temperature": (temperature if temperature > 0.0 else None),
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "tokenizer": kwargs['tokenizer'],
        }
        _safe_add_frequency_penalty_kwargs(self.model, gen_kwargs, frequency_penalty)

        del gen_kwargs['image_sizes']
        with torch.inference_mode():
            outputs = self.model.generate(**gen_kwargs)

        generated_ids = outputs[0]
        input_len = moved_inputs.get("input_ids").shape[1]
        new_tokens = generated_ids[:]
        tokenizer = kwargs['tokenizer']
        #print("input:", tokenizer.decode(moved_inputs.get("input_ids")[0]))
        #print("output:", tokenizer.decode(generated_ids))
        #print(input_len)
        #print(new_tokens)
        return self.processor.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


class OutlinesVLGenerator(VLConstrainedGeneratorBase):
    def generate(
        self,
        inputs: Any,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        json_schema: Optional[Union[str, Dict[str, Any]]] = None,
        cfg: Optional[str] = None,
        regex: Optional[str] = None,
        **kwargs,
    ) -> str:
        if self.processor is None:
            raise ValueError("OutlinesVLGenerator requires `processor` (HF processor / ProcessorMixin).")

        try:
            import outlines
            from outlines.types import JsonSchema, CFG, Regex

            if json_schema is None and cfg is None and regex is None:
                if isinstance(inputs, dict):
                    return PromptOnlyVLGenerator(self.model, processor=self.processor, tokenizer=self.tokenizer).generate(
                        inputs=inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                ol_model = outlines.from_transformers(self.model, self.processor)

                
                setattr(ol_model, "hf_tokenizer", self.tokenizer)
                return ol_model(inputs, max_new_tokens=max_new_tokens)

            print("&"*100)
            print("outlinesvlgenerator generate")
            print("&"*100)
            
            if json_schema is not None:
                if isinstance(json_schema, dict):
                    schema_obj = json_schema
                elif isinstance(json_schema, str):
                    try:
                        schema_obj = json.loads(json_schema)
                    except json.JSONDecodeError:
                        if os.path.exists(json_schema):
                            with open(json_schema, "r", encoding="utf-8") as f:
                                schema_obj = json.load(f)
                        else:
                            raise ValueError(f"Invalid JSON schema: {json_schema}")
                else:
                    raise ValueError("json_schema must be dict/str/path")
                output_type = JsonSchema(schema_obj)
            elif cfg is not None:
                output_type = CFG(cfg)
            else:
                output_type = Regex(regex)

            ol_model = outlines.from_transformers(self.model, self.processor)
           
            setattr(ol_model, "hf_tokenizer", self.tokenizer)
            gen_kwargs = dict(output_type=output_type, max_new_tokens=max_new_tokens)
            if temperature is not None and float(temperature) > 0.0:
                gen_kwargs.update(temperature=float(temperature), top_p=top_p, do_sample=True)
            else:
                gen_kwargs.update(do_sample=False)

            prompt_text = kwargs.get("prompt_text")
            image = kwargs.get("image")
            if isinstance(prompt_text, str) and image is not None:
                inputs = _prepare_outlines_vlm_input(self.processor, prompt_text, image)

            result = ol_model(inputs, **gen_kwargs)
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"Outlines VLM constrained generation failed: {e}, falling back to prompt-only")
            if isinstance(inputs, dict):
                return PromptOnlyVLGenerator(self.model, processor=self.processor, tokenizer=self.tokenizer).generate(
                    inputs=inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            raise


class XGrammarVLGenerator(VLConstrainedGeneratorBase):
    def __init__(self, model: Any, processor: Any = None, tokenizer: Any = None, **kwargs):
        super().__init__(model, processor=processor, tokenizer=tokenizer, **kwargs)
        if self.tokenizer is None:
            self.tokenizer = getattr(processor, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("XGrammarVLGenerator requires a tokenizer (pass tokenizer or processor.tokenizer).")

        try:
            import xgrammar as xgr

            self.xgr = xgr
            config = AutoConfig.from_pretrained(
                getattr(self.tokenizer, "name_or_path", None) or kwargs.get("model_name_or_path"),
                trust_remote_code=kwargs.get("trust_remote_code"),
            )
            tokenizer_info = self.xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=config.vocab_size)
            self.grammar_compiler = self.xgr.GrammarCompiler(tokenizer_info)
        except ImportError:
            raise ImportError("xgrammar not installed, please run: pip install xgrammar")

    def generate(
        self,
        inputs: Dict[str, Any],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        json_schema: Optional[Union[str, Dict[str, Any]]] = None,
        ebnf: Optional[str] = None,
        regex: Optional[str] = None,
        **kwargs,
    ) -> str:
        if json_schema is None and ebnf is None and regex is None:
            return PromptOnlyVLGenerator(self.model, processor=self.processor, tokenizer=self.tokenizer).generate(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
            )

        if json_schema is not None:
            grammar_obj = self._build_json_constraint(json_schema)
        elif regex is not None:
            grammar_obj = self._build_regex_constraint(regex)
        else:
            grammar_obj = self._build_ebnf_constraint(ebnf)

        compiled_grammar = self.grammar_compiler.compile_grammar(grammar_obj)
        xgr_processor = LogitsProcessorList([self.xgr.contrib.hf.LogitsProcessor(compiled_grammar)])

        moved_inputs = _normalize_and_move_vl_inputs(self.model, inputs)

        gen_kwargs: Dict[str, Any] = {
            **moved_inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": (temperature > 0.0),
            "temperature": (temperature if temperature > 0.0 else None),
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "logits_processor": xgr_processor,
        }
        _safe_add_frequency_penalty_kwargs(self.model, gen_kwargs, frequency_penalty)

        with torch.inference_mode():
            outputs = self.model.generate(**gen_kwargs)

        generated_ids = outputs[0]
        input_len = moved_inputs.get("input_ids").shape[1]
        new_tokens = generated_ids[input_len:]
        return self.processor.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _build_json_constraint(self, json_schema: Union[str, Dict[str, Any]]) -> Any:
        if isinstance(json_schema, dict):
            json_schema = json.dumps(json_schema, ensure_ascii=False)
        return self.xgr.Grammar.from_json_schema(json_schema)

    def _build_regex_constraint(self, regex: str) -> Any:
        return self.xgr.Grammar.from_regex(regex)

    def _build_ebnf_constraint(self, ebnf_str: str) -> str:
        return ebnf_str


class GuidanceVLGenerator(VLConstrainedGeneratorBase):
    def generate(self, inputs: Dict[str, Any], **kwargs) -> str:
        raise NotImplementedError


class LlamaCppVLGenerator(VLConstrainedGeneratorBase):
    def __init__(self, model: Any, processor: Any = None, tokenizer: Any = None, **kwargs):
        super().__init__(model, processor=processor, tokenizer=tokenizer, **kwargs)
        raise NotImplementedError("llama_cpp only supports pure text gguf inference, does not support VLM inputs (pixel_values, etc.)")


class VLConstrainedGenerator:
    _generators = {
        GenerationMode.PROMPT: PromptOnlyVLGenerator,
        GenerationMode.XGRAMMAR: XGrammarVLGenerator,
        GenerationMode.OUTLINES: OutlinesVLGenerator,
        GenerationMode.GUIDANCE: GuidanceVLGenerator,
        GenerationMode.LLAMA_CPP: LlamaCppVLGenerator,
    }

    def __init__(
        self,
        mode: Union[str, GenerationMode] = GenerationMode.PROMPT,
        model: Any = None,
        processor: Any = None,
        tokenizer: Any = None,
        **kwargs,
    ):
        if isinstance(mode, str):
            mode = GenerationMode(mode)

        generator_cls = self._generators.get(mode)
        if generator_cls is None:
            raise ValueError(f"unsupported generation mode: {mode}")
        
        self.tokenizer = tokenizer
        self.generator = generator_cls(model=model, processor=processor, tokenizer=tokenizer, **kwargs)

    def generate(self, inputs: Dict[str, Any], **kwargs) -> str:
        return self.generator.generate(inputs=inputs, tokenizer=self.tokenizer,  **kwargs)
