#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
jsonschemabench_generate.py

在 JSONSchemaBench 上调用任意 transformers 模型进行“无约束”生成：
- 从本地 parquet 数据文件加载 json_schema，而不是在线下载
- 为每个 json_schema 构造统一 prompt
- 使用 transformers 的模型直接生成 JSON（不使用 xgrammar / Guidance 等约束框架）

用法示例（在 datacode 目录下执行）：

python -m data.jsschemabench_generate \
  --model_name_or_path ../Models/MiniCPM4-0.5B \
  --data_file ../datasets/JSONSchemaBench/Github_trivial/test-00000-of-00001.parquet \
  --split test \
  --num_samples 10 \
  --max_new_tokens 256 \
  --device cuda \
  --temperature 0
"""

import argparse
import json
import textwrap
from typing import Optional, Tuple

from datasets import load_dataset
try:
    import jsonschema

    JSONSCHEMA_AVAILABLE = True
except Exception:  # noqa: BLE001
    JSONSCHEMA_AVAILABLE = False
    jsonschema = None

from common.cli import add_common_gen_args, add_common_model_args
from common.text_model_utils import load_text_model_and_tokenizer
from common.constrained_generation import ConstrainedGenerator
from common.cg_utils import load_template, generate_with_generator, append_jsonl_log


# ============== Prompt 模板 ==============

PROMPT_TEMPLATE = """You are given a JSON Schema that defines the structure of a JSON object.
Follow the example and produce a single valid JSON object that conforms to the schema.

Example Schema:
{example_schema}
Example Output:
{example_output}

Now generate for the new schema.

JSON Schema:
{schema_text}

Output JSON:
"""


EXAMPLE_SCHEMA = """{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "number"}
  },
  "required": ["name", "age"]
}"""

EXAMPLE_OUTPUT = """{"name": "Alice", "age": 30}"""


def build_prompt(schema_text: str) -> str:
    """把 json_schema 文本嵌入到统一的自然语言 prompt 中。"""
    return PROMPT_TEMPLATE.format(
        example_schema=EXAMPLE_SCHEMA,
        example_output=EXAMPLE_OUTPUT,
        schema_text=schema_text.strip(),
    )


def validate_output_json(output_text: str, schema_text: str) -> Tuple[bool, Optional[bool], str]:
    """
    尝试解析模型输出为 JSON，并可选做 schema 校验。
    返回 (json_parsed, schema_ok, reason)；schema_ok 为 None 表示跳过校验。
    """
    text = output_text.strip()
    # 去掉可能的代码块包装
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    # 粗略截取第一个 JSON 对象
    start = text.find("{")
    end = text.rfind("}")
    candidate = text[start : end + 1] if start != -1 and end != -1 and end > start else text
    candidate = candidate.strip()
    try:
        parsed_json = json.loads(candidate)
    except Exception as exc:  # noqa: BLE001
        return False, None, f"json parse failed: {exc}"

    if not JSONSCHEMA_AVAILABLE:
        return True, None, "json parsed (schema validation skipped: jsonschema not installed)"

    try:
        schema_obj = json.loads(schema_text)
    except Exception as exc:  # noqa: BLE001
        return True, None, f"json parsed, but schema parsing failed: {exc}"

    try:
        jsonschema.validate(parsed_json, schema_obj)
        return True, True, "json parsed and validated against schema"
    except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
        return True, False, f"schema validation failed: {exc.message}"
    except jsonschema.SchemaError as exc:  # type: ignore[attr-defined]
        return True, None, f"schema invalid, skip validation: {exc}"


# ============== 主流程 ==============

def run_generation(
    model_name_or_path: str,
    data_file: str,
    split: str,
    num_samples: int,
    max_new_tokens: int,
    device: str = "cuda",
    trust_remote_code: bool = True,
    temperature: float = 0.0,
    top_p: float = 1.0,
    torch_dtype: str = "auto",
    attn_implementation: str = None,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    mode: str = "prompt",
    log_file: str | None = None,
):
    # 1. 加载 JSONSchemaBench 本地 parquet 数据
    ds_dict = load_dataset(
        "parquet",
        data_files={split: data_file},
    )
    assert split in ds_dict, f"split {split!r} not found in data file; available: {list(ds_dict.keys())}"
    ds = ds_dict[split]

    print(f"Loaded dataset file: {data_file} [{split}]")
    print(f"num_rows = {len(ds)}")

    # 2. 加载模型 & tokenizer
    model, tokenizer, config = load_text_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    is_encoder_decoder = getattr(config, "is_encoder_decoder", False)

    # 2.1 创建约束生成器（可选）
    generator = ConstrainedGenerator(
        mode=mode,
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=trust_remote_code,
        model_name_or_path=model_name_or_path,
    )

    # 3. 遍历前 num_samples 个样本
    if num_samples == -1:
        n = len(ds)
    else:
        n = min(num_samples, len(ds))
    print(f"\n=== Start generation (first {n} samples) ===\n")

    correct = 0
    syntax_errors = 0
    schema_errors = 0
    schema_skipped = 0

    for idx in range(n):
        sample = ds[idx]
        schema_text = sample["json_schema"]
        schema_id = sample.get("unique_id", f"idx-{idx}")

        prompt = build_prompt(schema_text)

        sample_log = {
            "task": "jsonschemabench",
            "model_name_or_path": model_name_or_path,
            "data_file": data_file,
            "split": split,
            "mode": mode,
            "sample_id": schema_id,
            "schema_text": schema_text,
            "prompt_preview": prompt[:300] + "..." if len(prompt) > 300 else prompt,
        }

        # Print prompt preview for debugging (shortened)
        print("#" * 80)
        print(f"[Sample {idx}] unique_id = {schema_id}")
        print("- Prompt preview (truncated) -")
        print(textwrap.shorten(prompt, width=300, placeholder=" ..."))
        print("-" * 80)

        # 4. Generate（统一调用约束生成器）
        output_text = generate_with_generator(
            generator,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            constraint=schema_text,
            constraint_type="json",
        )

        print("Model-generated JSON:")
        print(output_text.strip())

        sample_log["raw_output"] = output_text

        is_parsed, schema_ok, reason = validate_output_json(output_text, schema_text)
        if not is_parsed:
            syntax_errors += 1
            status = "syntax error"
        elif schema_ok is False:
            schema_errors += 1
            status = "content error"
        else:
            correct += 1
            status = "passed" if schema_ok else "schema check skipped"
            if schema_ok is None:
                schema_skipped += 1
        sample_log["json_parsed"] = bool(is_parsed)
        sample_log["schema_ok"] = schema_ok
        sample_log["status"] = status
        sample_log["reason"] = reason

        print(f"[validation] {status} ({reason})")

        if log_file:
            append_jsonl_log(log_file, sample_log, verbose=False)

        print("\n")

    if n > 0:
        acc = correct / n if n else 0.0
        print("=" * 80)
        print(f"Valid outputs (parsed + schema): {correct}/{n} = {acc:.2%}")
        print(f"Syntax errors (invalid JSON): {syntax_errors}")
        print(f"Content errors (schema violations): {schema_errors}")
        if schema_skipped:
            print(f"Schema validation skipped: {schema_skipped} (missing jsonschema or invalid schema)")

        if log_file:
            summary_log = {
                "task": "jsonschemabench_summary",
                "model_name_or_path": model_name_or_path,
                "data_file": data_file,
                "split": split,
                "mode": mode,
                "num_samples": n,
                "correct": correct,
                "accuracy": f"{acc:.4f}",
                "syntax_errors": syntax_errors,
                "schema_errors": schema_errors,
                "schema_skipped": schema_skipped,
                "jsonschema_available": bool(JSONSCHEMA_AVAILABLE),
            }
            append_jsonl_log(log_file, summary_log, verbose=False)

        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run unconstrained generation on JSONSchemaBench with a transformers model.",
    )
    parser = add_common_model_args(parser)
    parser = add_common_gen_args(parser)
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Path to local JSONSchemaBench parquet file, e.g., datasets/JSONSchemaBench/Github_easy/test-00000-of-00001.parquet.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="Github_easy",
        help="If data_file is not provided, use datasets/JSONSchemaBench/{subset}/{split}-00000-of-00001.parquet.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split: train / val / test.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to generate from the dataset.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="prompt",
        choices=["prompt", "xgrammar", "guidance", "outlines", "llama_cpp"],
        help="生成模式：prompt（仅 prompt）、xgrammar、guidance、outlines、llama_cpp。",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="逐样本输出日志文件路径（jsonl）。不传则不写日志。",
    )
    return parser.parse_args()



def main():
    args = parse_args()
    trust_remote_code = not args.no_trust_remote_code

    data_file = args.data_file
    if data_file is None:
        data_file = f"datasets/JSONSchemaBench/{args.subset}/{args.split}-00000-of-00001.parquet"

    run_generation(
        model_name_or_path=args.model_name_or_path,
        data_file=data_file,
        split=args.split,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        trust_remote_code=trust_remote_code,
        temperature=args.temperature,
        top_p=args.top_p,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        mode=args.mode,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
