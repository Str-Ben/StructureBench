#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
chequessample_generate.py

在 cheques_sample_data 数据集上调用 Qwen3-VL-2B-Instruct 进行生成：
- 从本地 parquet 数据加载支票图像 + 结构化标注
- 为每条样本构造包含图像的 prompt，请模型输出结构化 JSON
- 不做正确性验证，直接打印模型生成

用法示例（在 datacode 目录下执行）：

python -m data.chequessample_generate \
  --model_name_or_path ../Models/Qwen3-VL-2B-Instruct \
  --data_file ../datasets/cheques_sample_data/data/validation-00000-of-00001-4b3af28127dd79d2.parquet \
  --split validation \
  --num_samples 10 \
  --max_new_tokens 128 \
  --device cuda \
  --temperature 0
"""

import argparse
import ast
import io
import json
import textwrap
from typing import Any, Dict, Optional, Tuple

from datasets import load_dataset
from PIL import Image
from common.cli import add_common_gen_args, add_common_model_args
from common.vl_model_utils import (
    build_vl_inputs,
    generate_minicpm_chat,
    generate_vl,
    load_vl_model_and_processor,
)
from transformers import AutoTokenizer
from common.vl_constrained_genenration import VLConstrainedGenerator
from common.cg_utils import load_template
try:
    import jsonschema

    JSONSCHEMA_AVAILABLE = True
except Exception:  # noqa: BLE001
    JSONSCHEMA_AVAILABLE = False
    jsonschema = None


# ============== Prompt 模板 ==============

PROMPT_TEMPLATE = """You are a cheque OCR assistant. Given a cheque image, extract fields into JSON with this schema:

{{gt_parse": {{cheque_details": [{{"amt_in_words": string}}, {{"amt_in_figures": string}}, {{"payee_name": string}}, {{"bank_name": string}}, {{"cheque_date": string}}]}}}}

Only output the JSON WITHOUT any line breaks. Follow the value formatting in the example (dates, numbers, names) exactly. For amt_in_words, keep the OCR-like abbreviated spellings as seen in the image (common patterns: For->Four, Eigt->Eight, Thre->Three, On->One, Foty->Forty, Seenty->Seventy, Sixt->Sixty, Eihty->Eighty, Eihteen->Eighteen, Twlve->Twelve, Fiteen->Fifteen, Forteen->Fourteen, Nineten->Nineteen, Sxten->Sixteen); do not auto-correct.
{example_block}"""


def load_image_from_bytes(image_field: Dict[str, Any]) -> Image.Image:
    """将 parquet 里的图片字段转为 PIL Image；兼容已解码的 PIL 对象。"""
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    img_bytes = image_field["bytes"]
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# ============== 校验 & 辅助函数 ==============

CHEQUE_JSON_SCHEMA = {
    "type": "object",
    "required": ["gt_parse"],
    "properties": {
        "gt_parse": {
            "type": "object",
            "required": ["cheque_details"],
            "properties": {
                "cheque_details": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "minProperties": 1,
                        "maxProperties": 1,
                        "additionalProperties": {"type": "string"},
                    },
                }
            },
            "additionalProperties": False,
        }
    },
    "additionalProperties": False,
}


def _basic_schema_check(obj: Any) -> Tuple[bool, str]:
    """在 jsonschema 不可用时做一个保底的结构校验。"""
    if not isinstance(obj, dict):
        return False, "root is not an object"
    gt_parse = obj.get("gt_parse")
    if not isinstance(gt_parse, dict):
        return False, "gt_parse missing or not an object"
    details = gt_parse.get("cheque_details")
    if not isinstance(details, list) or not details:
        return False, "cheque_details missing or not a non-empty list"
    for i, item in enumerate(details):
        if not isinstance(item, dict) or not item:
            return False, f"cheque_details[{i}] is not an object with one field"
        if len(item) != 1:
            return False, f"cheque_details[{i}] should contain exactly one field"
        value = next(iter(item.values()))
        if not isinstance(value, str):
            return False, f"cheque_details[{i}] value is not a string"
    return True, "basic structure passed"


def _extract_json_candidate(output_text: str) -> str:
    """尝试从模型输出中提取出 JSON 文本片段。"""
    text = output_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    start = text.find("{")
    end = text.rfind("}")
    candidate = text[start : end + 1] if start != -1 and end != -1 and end > start else text
    return candidate.strip()


_QUOTE_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "＂": '"',
        "＇": "'",
        ".": "/"
    }
)


def _normalize_text(text: str) -> str:
    """标准化引号与空白，避免后处理中误判。"""
    normalized = text.translate(_QUOTE_TRANSLATION)
    return " ".join(normalized.split())


def _parse_json_text(text: str) -> Tuple[bool, Optional[Any], str]:
    """解析 JSON 或 Python 字面量，避免粗暴替换引号导致内容变化。"""
    candidate = _extract_json_candidate(text)
    try:
        return True, json.loads(candidate), "json parsed"
    except Exception as exc:  # noqa: BLE001
        json_error = exc

    try:
        parsed = ast.literal_eval(candidate)
    except Exception as exc:  # noqa: BLE001
        return False, None, f"json parse failed: {json_error}; literal_eval failed: {exc}"

    if isinstance(parsed, str):
        try:
            return True, json.loads(parsed), "json parsed (from literal string)"
        except Exception as exc:  # noqa: BLE001
            return False, None, f"literal_eval produced string but json parse failed: {exc}"
    return True, parsed, "literal_eval parsed"


def validate_output_json(output_text: str) -> Tuple[bool, bool, Optional[dict], str]:
    """
    校验模型输出：
    - 返回 (json_parsed, schema_ok, parsed_obj, reason)
    - json_parsed=False 表示无法解析 JSON
    - schema_ok=False 表示通过 JSON 解析但不满足 schema
    """
    parsed, parsed_json, reason = _parse_json_text(output_text)
    if not parsed:
        return False, False, None, reason

    if JSONSCHEMA_AVAILABLE:
        try:
            jsonschema.validate(parsed_json, CHEQUE_JSON_SCHEMA)
            return True, True, parsed_json, "json parsed and validated against schema"
        except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            return True, False, parsed_json, f"schema validation failed: {exc.message}"
        except jsonschema.SchemaError as exc:  # type: ignore[attr-defined]
            # schema 本身异常时，降级为基础校验
            ok, reason = _basic_schema_check(parsed_json)
            return True, ok, parsed_json, f"schema invalid ({exc}), basic check: {reason}"

    ok, reason = _basic_schema_check(parsed_json)
    return True, ok, parsed_json, f"json parsed (basic schema check): {reason}"


def _flatten_cheque_details(obj: dict, normalize: bool = False) -> Optional[Dict[str, str]]:
    """把 cheque_details 展平为 {field: value}，便于做语义比对。"""
    try:
        details = obj["gt_parse"]["cheque_details"]
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(details, list):
        return None
    flattened: Dict[str, str] = {}
    for item in details:
        if not isinstance(item, dict) or len(item) != 1:
            return None
        key, value = next(iter(item.items()))
        if not isinstance(value, str):
            return None
        key_text = str(key).strip()
        value_text = value.strip()
        if normalize:
            key_text = _normalize_text(key_text)
            value_text = _normalize_text(value_text)
            if key_text == "cheque_date":
                value_text = value_text.replace(".", "/")
        flattened[key_text] = value_text
    return flattened


# ============== 主流程 ==============

def run_generation(
    model_name_or_path: str,
    data_file: str,
    split: str,
    num_samples: int,
    max_new_tokens: int,
    device: str = "cuda",
    temperature: float = 0.0,
    top_p: float = 1.0,
    trust_remote_code: bool = True,
    torch_dtype: str = "auto",
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    mode: str = "prompt",
):
    # 1. 加载数据集（本地 parquet）
    ds_dict = load_dataset(
        "parquet",
        data_files={split: data_file},
    )
    assert split in ds_dict, f"split {split!r} not found in data file; available: {list(ds_dict.keys())}"
    ds = ds_dict[split]

    print(f"Loaded dataset file: {data_file} [{split}]")
    print(f"num_rows = {len(ds)}")

    # 2. 加载模型 & processor
    model, processor = load_vl_model_and_processor(
        model_name_or_path=model_name_or_path,
        device=device,
        trust_remote_code=trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )

    generator = VLConstrainedGenerator(
        mode=mode,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        trust_remote_code=trust_remote_code,
        model_name_or_path=model_name_or_path,
    )
    constraint_content, constraint_type = CHEQUE_JSON_SCHEMA, "json"

    # 3. 遍历前 num_samples 个样本
    if num_samples == -1:
        n = len(ds)
    else:
        n = min(num_samples, len(ds))
    print(f"\n=== Start generation (first {n} samples) ===\n")

    example_json = ds[0].get("ground_truth", "") if len(ds) > 0 else ""
    example_block = ""
    if example_json:
        example_block = (
            "\n\nExample output (match this style):\n"
            f"{example_json}\n"
            "JSON for the new cheque image:"
    )
    prompt_text = PROMPT_TEMPLATE.format(example_block=example_block)
    correct = 0
    syntax_errors = 0
    semantic_errors = 0

    for idx in range(n):
        sample = ds[idx]
        image = load_image_from_bytes(sample["image"])
        gold_json = sample.get("ground_truth")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Print short preview (without showing the image)
        print("#" * 80)
        print(f"[Sample {idx}]")
        print("- Prompt preview (truncated) -")
        print(textwrap.shorten(prompt_text, width=200, placeholder=" ..."))
        print("-" * 80)

        inputs = build_vl_inputs(messages, processor, model)
        

        # For outlines/guidance, we need to pass the prompt_text explicitly
        if mode in ("outlines", "guidance"):
            inputs["prompt_text"] = prompt_text

        if mode == "outlines":
            outlines_prompt = None

            # MiniCPM-V: tokenizer/processor chat_template 通常无法处理 content(list[image,text])，这里手动抽取文本。
            cls_name = processor.__class__.__name__.lower()
            if "minicpm" in cls_name:
                text_parts = []
                for msg in messages:
                    for item in msg.get("content", []):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                text = "\n\n".join([t for t in text_parts if t is not None])
                if "<image" not in text:
                    text = "<image>./</image>\n" + text
                outlines_prompt = text
            else:
                # Prefer tokenizer chat template (some processors don't ship one)
                if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
                    # 注意：许多 chat_template 不支持多模态 content(list)。这里只用于支持纯文本模板。
                    outlines_prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                # Fallback to processor chat template if available
                elif hasattr(processor, "apply_chat_template") and getattr(processor, "chat_template", None):
                    outlines_prompt = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                # Final fallback: plain text prompt
                else:
                    outlines_prompt = prompt_text
        else:
            outlines_prompt = prompt_text

        output_text = generator.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_text=outlines_prompt,
            image=image,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            json_schema=constraint_content,
        )

        print("Model output:")
        print(output_text.strip())
        if gold_json is not None:
            parsed, schema_ok, parsed_obj, reason = validate_output_json(output_text)
            if not parsed or not schema_ok or parsed_obj is None:
                syntax_errors += 1
                status = "syntax error (parse/schema)"
            else:
                if isinstance(gold_json, dict):
                    gold_parsed, gold_obj, gold_reason = True, gold_json, "gold already dict"
                else:
                    gold_parsed, gold_obj, gold_reason = _parse_json_text(str(gold_json))
                if not gold_parsed or not isinstance(gold_obj, dict):
                    semantic_errors += 1
                    status = f"semantic error (gold parse failed: {gold_reason})"
                else:
                    gold_fields = _flatten_cheque_details(gold_obj, normalize=True)
                    pred_fields = _flatten_cheque_details(parsed_obj, normalize=True)
                    semantic_match = (
                        gold_fields is not None
                        and pred_fields is not None
                        and gold_fields == pred_fields
                    )
                    if semantic_match:
                        correct += 1
                        status = "correct"
                    else:
                        semantic_errors += 1
                        status = "semantic error (content mismatch)"

            print(f"[gold json] {gold_json}")
            if parsed_obj is not None:
                print(f"[parsed obj] {parsed_obj}")
            print(f"[validation] {status} ({reason})")
        print("\n")

    if n > 0:
        acc = correct / n if n else 0.0
        print("=" * 80)
        print(f"Accuracy: {correct}/{n} = {acc:.2%}")
        print(f"Syntax errors (parse/schema): {syntax_errors}")
        print(f"Semantic errors (schema ok but mismatched content): {semantic_errors}")
        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run cheque OCR-style generation with a multimodal model.",
    )
    parser = add_common_model_args(parser)
    parser = add_common_gen_args(parser)
    parser.add_argument(
        "--data_file",
        type=str,
        default="datasets/cheques_sample_data/data/validation-00000-of-00001-4b3af28127dd79d2.parquet",
        help="cheques_sample_data data file (parquet).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split; must match the key in data_files. Default: validation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to generate from the dataset.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="prompt",
        choices=["prompt", "xgrammar", "guidance", "outlines", "llama_cpp"],
        help="生成模式：prompt（仅 prompt）、xgrammar、guidance、outlines、llama_cpp。",
    )
    
    return parser.parse_args()



def main():
    args = parse_args()
    trust_remote_code = not args.no_trust_remote_code

    run_generation(
        model_name_or_path=args.model_name_or_path,
        data_file=args.data_file,
        split=args.split,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        trust_remote_code=trust_remote_code,
        torch_dtype=args.torch_dtype,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
