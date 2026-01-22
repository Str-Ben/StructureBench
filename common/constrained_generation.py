#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
constrained_generation.py

约束生成框架的单文件版本，包含所有生成器实现。
"""

import logging
import json
import os
import importlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Union, Dict, Optional

import torch
from transformers import LogitsProcessor, LogitsProcessorList, AutoConfig

try:
    from transformers.generation.logits_process import (
        RepetitionPenaltyLogitsProcessor,
        FrequencyPenaltyLogitsProcessor,
    )
except Exception:  # pragma: no cover - optional/compat
    RepetitionPenaltyLogitsProcessor = None
    FrequencyPenaltyLogitsProcessor = None

logger = logging.getLogger(__name__)


# ============== Utilities ==============

def _build_model_inputs_with_chat(tokenizer, prompt: str, model):
    """统一使用 chat_template 构造输入；若不支持则模拟 chat 形式。"""
    messages = [{"role": "user", "content": prompt}]
    text = None
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            kwargs = dict(tokenize=False, add_generation_prompt=True)
            try:
                name_or_path = str(getattr(tokenizer, "name_or_path", "") or "")
                model_name = str(getattr(getattr(model, "config", None), "name_or_path", "") or "")
                hint = (name_or_path + " " + model_name).lower()
                if "qwen" in hint:
                    kwargs["enable_thinking"] = False
            except Exception:
                pass

            text = tokenizer.apply_chat_template(
                messages,
                **kwargs,
            )
        except TypeError:
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                )
            except Exception:
                text = None
    if text is None:
        text = f"User: {prompt}\nAssistant:"
    return tokenizer(text, return_tensors="pt")




# ============== Base Classes and Enums ==============

class GenerationMode(str, Enum):
    PROMPT = "prompt"
    XGRAMMAR = "xgrammar"
    GUIDANCE = "guidance"
    OUTLINES = "outlines"
    LLAMA_CPP = "llama_cpp"


class ConstrainedGeneratorBase(ABC):

    def __init__(self, model: Any, tokenizer: Any, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.config = kwargs

    @abstractmethod
    def generate(self, prompt: Optional[str] = None, **kwargs) -> str:
        pass


# ============== Generator Implementations ==============

class PromptOnlyGenerator(ConstrainedGeneratorBase):

    def generate(
        self,
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        **kwargs,
    ) -> str:
        model_inputs = _build_model_inputs_with_chat(self.tokenizer, prompt, self.model)
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

        # 参考 text_model_utils.py 的 penalty 实现
        supports_repetition = hasattr(self.model.generation_config, "repetition_penalty")
        supports_frequency = hasattr(self.model.generation_config, "frequency_penalty")

        logits_processors = []
        if not supports_repetition and repetition_penalty != 1.0 and RepetitionPenaltyLogitsProcessor:
            logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if not supports_frequency and frequency_penalty != 0.0 and FrequencyPenaltyLogitsProcessor:
            logits_processors.append(FrequencyPenaltyLogitsProcessor(frequency_penalty))

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": (temperature > 0.0),
            "temperature": temperature if temperature > 0.0 else None,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if supports_repetition:
            generation_kwargs["repetition_penalty"] = repetition_penalty
        if supports_frequency:
            generation_kwargs["frequency_penalty"] = frequency_penalty
        if logits_processors:
            generation_kwargs["logits_processor"] = LogitsProcessorList(logits_processors)

        with torch.inference_mode():
            outputs = self.model.generate(**model_inputs, **generation_kwargs)
        is_encdec = getattr(getattr(self.model, "config", object), "is_encoder_decoder", False)
        if is_encdec:
            new_tokens = outputs[0]
        else:
            cut = model_inputs["input_ids"].shape[1]
            new_tokens = outputs[0][cut:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class XGrammarGenerator(ConstrainedGeneratorBase):
    def __init__(self, model: Any, tokenizer: Any, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        try:
            import xgrammar as xgr

            self.xgr = xgr
            config = AutoConfig.from_pretrained(
                self.tokenizer.name_or_path, trust_remote_code=kwargs.get("trust_remote_code")
            )
            tokenizer_info = self.xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=config.vocab_size)
            self.grammar_compiler = self.xgr.GrammarCompiler(tokenizer_info)
        except ImportError:
            raise ImportError("xgrammar not installed, please run: pip install xgrammar")

    def generate(
        self,
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        json_schema: Optional[Union[str, Dict[str, Any]]] = None,
        ebnf: Optional[str] = None,
        regex: Optional[str] = None,
        **kwargs,
    ) -> str:
        if json_schema is None and ebnf is None and regex is None:
            return PromptOnlyGenerator(self.model, self.tokenizer).generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=kwargs.get("repetition_penalty", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            )

        if json_schema is not None:
            grammar_obj = self._build_json_constraint(json_schema)
        elif regex is not None:
            grammar_obj = self._build_regex_constraint(regex)
        else:
            grammar_obj = self._build_ebnf_constraint(ebnf)

        compiled_grammar = self.grammar_compiler.compile_grammar(grammar_obj)

        print("&"*100) 
        print(compiled_grammar)
        print("&"*100) 

        model_inputs = _build_model_inputs_with_chat(self.tokenizer, prompt, self.model)
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
        logits_processor = self._create_logits_processor(compiled_grammar)

        # penalty 支持（参考 text_model_utils.py）
        repetition_penalty = float(kwargs.get("repetition_penalty", 1.0))
        frequency_penalty = float(kwargs.get("frequency_penalty", 0.0))
        supports_repetition = hasattr(self.model.generation_config, "repetition_penalty")
        supports_frequency = hasattr(self.model.generation_config, "frequency_penalty")

        penalty_processors = []
        if not supports_repetition and repetition_penalty != 1.0 and RepetitionPenaltyLogitsProcessor:
            penalty_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if not supports_frequency and frequency_penalty != 0.0 and FrequencyPenaltyLogitsProcessor:
            penalty_processors.append(FrequencyPenaltyLogitsProcessor(frequency_penalty))

        # 合并：先 penalty，再 grammar 约束
        all_processors = []
        if penalty_processors:
            all_processors.extend(penalty_processors)
        if logits_processor is not None:
            # xgrammar 的 logits_processor 本身是 LogitsProcessorList
            try:
                all_processors.extend(list(logits_processor))
            except Exception:
                all_processors.append(logits_processor)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": (temperature > 0.0),
            "temperature": temperature if temperature > 0.0 else None,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if supports_repetition:
            generation_kwargs["repetition_penalty"] = repetition_penalty
        if supports_frequency:
            generation_kwargs["frequency_penalty"] = frequency_penalty
        if all_processors:
            generation_kwargs["logits_processor"] = LogitsProcessorList(all_processors)

        with torch.inference_mode():
            outputs = self.model.generate(
                **model_inputs,
                **generation_kwargs,
            )

        is_encdec = getattr(getattr(self.model, "config", object), "is_encoder_decoder", False)
        if is_encdec:
            new_tokens = outputs[0]
        else:
            cut = model_inputs["input_ids"].shape[1]
            new_tokens = outputs[0][cut:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _build_json_constraint(self, json_schema: Union[str, Dict[str, Any]]) -> Any:
        if isinstance(json_schema, dict):
            json_schema = json.dumps(json_schema, ensure_ascii=False)
        return self.xgr.Grammar.from_json_schema(json_schema)

    def _build_regex_constraint(self, regex: str) -> Any:
        return self.xgr.Grammar.from_regex(regex)

    def _build_ebnf_constraint(self, ebnf_str: str) -> str:
        return ebnf_str

    def _create_logits_processor(self, compiled_grammar: Any) -> Any:
        return LogitsProcessorList([self.xgr.contrib.hf.LogitsProcessor(compiled_grammar)])


class GuidanceGenerator(ConstrainedGeneratorBase):

    def __init__(self, model: Any, tokenizer: Any, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        try:
            import guidance

            self.guidance = guidance
            self.model_name_or_path = kwargs.get("model_name_or_path")
        except ImportError:
            raise ImportError("guidance not installed, please run: pip install guidance")

    def generate(
        self,
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        json_schema: Optional[Union[str, Dict[str, Any]]] = None,
        regex: Optional[str] = None,
        guidance_template: Optional[str] = None,
        **kwargs,
    ) -> str:
        try:
            from guidance import models, gen, json as gen_json
        except ImportError:
            raise ImportError("guidance library import failed, please ensure it is correctly installed")

        repetition_penalty = float(kwargs.get("repetition_penalty", 1.0))
        frequency_penalty = float(kwargs.get("frequency_penalty", 0.0))

        if json_schema is None and regex is None and guidance_template is None:
            return PromptOnlyGenerator(self.model, self.tokenizer).generate(
                prompt, max_new_tokens, temperature, top_p, repetition_penalty, frequency_penalty
            )

        try:
            lm = models.Transformers(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.model.device if hasattr(self.model, "device") else 0,
            )
        except Exception as e:
            logger.error(f"Guidance model initialization failed: {e}", exc_info=True)
            self.last_error = e  # type: ignore[attr-defined]
            return ""


        try:
            lm += prompt

            if json_schema is not None:
                if isinstance(json_schema, str):
                    try:
                        schema_dict = json.loads(json_schema)
                    except json.JSONDecodeError:
                        if os.path.exists(json_schema):
                            with open(json_schema, "r", encoding="utf-8") as f:
                                schema_dict = json.load(f)
                        else:
                            raise ValueError(f"Invalid JSON schema: {json_schema}")
                else:
                    schema_dict = json_schema

                # guidance 路径无法可靠透传 transformers 原生 repetition_penalty/frequency_penalty，
                # 这里采用“尽力而为”：如果 guidance 的 gen/json 支持 logits_processor，则注入。
                processors = []
                if repetition_penalty != 1.0 and RepetitionPenaltyLogitsProcessor is not None:
                    processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
                if frequency_penalty != 0.0 and FrequencyPenaltyLogitsProcessor is not None:
                    processors.append(FrequencyPenaltyLogitsProcessor(frequency_penalty))
                extra = {}
                if processors:
                    extra["logits_processor"] = LogitsProcessorList(processors)

                lm += gen_json(
                    name="output",
                    schema=schema_dict,
                    max_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0.0 else 0.0,
                    **extra,
                )
                result = lm["output"]
                if isinstance(result, dict):
                    return json.dumps(result, ensure_ascii=False)
                return str(result)

            elif regex is not None:
                processors = []
                if repetition_penalty != 1.0 and RepetitionPenaltyLogitsProcessor is not None:
                    processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
                if frequency_penalty != 0.0 and FrequencyPenaltyLogitsProcessor is not None:
                    processors.append(FrequencyPenaltyLogitsProcessor(frequency_penalty))
                extra = {}
                if processors:
                    extra["logits_processor"] = LogitsProcessorList(processors)

                lm += gen(
                    name="output",
                    regex=regex,
                    max_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0.0 else 0.0,
                    **extra,
                )
                return str(lm["output"])

            elif guidance_template is not None:
                try:
                    # 动态从 guidance_templates 模块加载约束函数
                    templates_module = importlib.import_module("guidance_templates")
                    template_function = getattr(templates_module, guidance_template)
                except (ImportError, AttributeError) as e:
                    logger.error(f"Failed to load '{guidance_template}' from guidance_templates: {e}", exc_info=True)
                    self.last_error = e  # type: ignore[attr-defined]
                    return ""

                # 允许模板函数接收 kwargs（如 max_new_tokens/temperature/top_p/penalty 等），
                # 但模板是否使用由其自身决定。
                lm += template_function(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    frequency_penalty=frequency_penalty,
                    **kwargs,
                )

                full_text = str(lm)
                result = full_text[len(prompt) :] if full_text.startswith(prompt) else full_text
                return result.strip()

        except Exception as e:
            logger.error(f"Guidance constrained generation failed: {e}", exc_info=True)
            self.last_error = e  # type: ignore[attr-defined]
            return ""


class OutlinesGenerator(ConstrainedGeneratorBase):
    def __init__(self, model: Any, tokenizer: Any, **kwargs):
        super().__init__(model, tokenizer, **kwargs)

    def generate(
        self,
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        json_schema: Optional[Union[str, Dict[str, Any]]] = None,
        cfg: Optional[str] = None,
        regex: Optional[str] = None,
        **kwargs,
    ) -> str:
        if json_schema is None and cfg is None and regex is None:
            logger.warning("No constraints specified, using prompt-only mode")
            return PromptOnlyGenerator(self.model, self.tokenizer).generate(prompt, max_new_tokens, temperature, top_p)

        print("&"*100)
        print("outlines generate")
        print("&"*100)
        try:
            import outlines
            from outlines.types import JsonSchema, CFG, Regex

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

            elif regex is not None:
                output_type = Regex(regex)
            else:
                raise ValueError("Must specify one of json_schema / cfg / regex")

            ol_model = outlines.from_transformers(self.model, self.tokenizer)

            gen_kwargs = dict(output_type=output_type, max_new_tokens=max_new_tokens)
            if temperature is not None and float(temperature) > 0.0:
                gen_kwargs.update(temperature=float(temperature), top_p=top_p, do_sample=True)
            else:
                gen_kwargs.update(do_sample=False)
            result = ol_model(prompt, **gen_kwargs)
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Outlines constrained generation failed: {e}", exc_info=True)
            self.last_error = e  # type: ignore[attr-defined]
            return ""


class LlamaCppGenerator(ConstrainedGeneratorBase):
    def __init__(self, model: Any, tokenizer: Any, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        try:
            from llama_cpp import Llama, LlamaGrammar

            self.Llama = Llama
            self.LlamaGrammar = LlamaGrammar
            gguf_path = kwargs.get("gguf_model_path", "/path/to/your/model.gguf")
            self.model = self.Llama(model_path=gguf_path, n_ctx=2048)
        except ImportError:
            raise ImportError("llama-cpp-python not installed, please run: pip install llama-cpp-python")

    def generate(
        self,
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        json_schema: Optional[Union[str, Dict[str, Any]]] = None,
        gbnf_grammar: Optional[str] = None,
        **kwargs,
    ) -> str:
        if json_schema is None and gbnf_grammar is None:
            logger.warning("No constraints specified, using prompt-only mode")
            return PromptOnlyGenerator(self.model, self.tokenizer).generate(prompt, max_new_tokens, temperature, top_p)

        if gbnf_grammar:
            grammar = self.LlamaGrammar.from_string(gbnf_grammar)
        elif json_schema:
            from .json_schema import json_schema_to_gbnf

            if isinstance(json_schema, dict):
                json_schema = json.dumps(json_schema)
            gbnf_string = json_schema_to_gbnf(json_schema)
            grammar = self.LlamaGrammar.from_string(gbnf_string)

        output = self.model(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            grammar=grammar,
            stop=None,
        )

        return output["choices"][0]["text"]


# ============== Factory Class ==============

class ConstrainedGenerator:
    _generators = {
        GenerationMode.PROMPT: PromptOnlyGenerator,
        GenerationMode.XGRAMMAR: XGrammarGenerator,
        GenerationMode.GUIDANCE: GuidanceGenerator,
        GenerationMode.OUTLINES: OutlinesGenerator,
        GenerationMode.LLAMA_CPP: LlamaCppGenerator,
    }

    def __init__(
        self,
        mode: Union[str, GenerationMode] = GenerationMode.PROMPT,
        model: Any = None,
        tokenizer: Any = None,
        **kwargs,
    ):
        if isinstance(mode, str):
            try:
                mode = GenerationMode(mode)
            except ValueError:
                raise ValueError(f"Invalid generation mode: {mode}. Supported modes: {[m.value for m in GenerationMode]}")

        self.mode = mode
        generator_cls = self._generators.get(mode)
        if generator_cls is None:
            raise ValueError(f"unsupported generation mode: {mode}")

        self.generator = generator_cls(model, tokenizer, **kwargs)
        logger.info(f"initialized {mode.value} constrained generator")

    def generate(self, prompt: Optional[str] = None, **kwargs) -> str:
        return self.generator.generate(prompt=prompt, **kwargs)
