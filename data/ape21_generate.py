#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ape21_generate.py

在 Calc-ape210k 数据集上调用任意 transformers 模型进行生成：
- 从本地 parquet 数据加载中文数学应用题
- 为每个 problem 构造统一 prompt，让模型只输出数字答案

用法示例（在 datacode 目录下执行）：

python -m data.ape21_generate \
  --model_name_or_path ../Models/MiniCPM4-0.5B \
  --data_file ../datasets/Calc-ape210k/data/validation-00000-of-00001-b9c45633c2837e3b.parquet \
  --split validation \
  --num_samples 10 \
  --max_new_tokens 1024 \
  --device cuda \
  --temperature 0
"""

import argparse
import re
import textwrap

from datasets import load_dataset
from sympy import sympify

from common.cli import add_common_gen_args
from common.text_model_utils import (
    add_common_model_args,
    load_text_model_and_tokenizer,
)

from common.constrained_generation import ConstrainedGenerator
from common.cg_utils import load_template, generate_with_generator, append_jsonl_log


# ============== Prompt 模板 ==============

PROMPT_TEMPLATE = """You are a math assistant.
Follow the example: give the final equation in the form x=... inside <final>...</final>.

Example:
Problem: {example_problem}
Answer: <final>{example_equation}</final>

Now solve the new problem.
Problem:
{problem}

Equation inside <final>...</final>:
"""


def build_prompt(problem: str, example_problem: str, example_equation: str) -> str:
    """把 Calc-ape210k 的题目嵌入统一 prompt，并附带示例。"""
    return PROMPT_TEMPLATE.format(
        example_problem=example_problem.strip(),
        example_equation=example_equation.strip(),
        problem=problem.strip(),
    )


def pick_example(ds):
    """从数据集中挑一条示例题目和答案。"""
    if len(ds) == 0:
        return ("", "")
    sample = ds[0]
    example_problem = sample.get("question") or sample.get("question_chinese") or ""
    example_equation = str(sample.get("equation", "") or sample.get("result", ""))
    return example_problem, example_equation


def normalize_answer(text: str) -> str:
    """去除所有空白用于精确匹配。"""
    if text is None:
        return ""
    return re.sub(r"\s+", "", str(text))


FINAL_PATTERN = re.compile(r"<final\b[^>]*>(.*?)</final>", flags=re.IGNORECASE | re.DOTALL)


def extract_final(text: str) -> str:
    """截取最后一个 <final>...</final> 之间的内容；若无标记则返回空字符串。"""
    if text is None:
        return ""
    matches = FINAL_PATTERN.findall(str(text))
    return matches[-1] if matches else ""


def normalized_final(text: str) -> str:
    """提取 final 段并去空白。"""
    return normalize_answer(extract_final(text))


def parse_equation_value(text: str):
    """
    解析形如 x=... 的等式，返回 (数值, 错误信息)。
    错误信息为 None 表示解析成功。
    """
    def _clean(raw: str) -> str:
        # Normalize tags and common full-width symbols before parsing.
        cleaned = re.sub(r"<[^>]+>", " ", raw or "")
        return (
            cleaned.replace("＝", "=")
            .replace("ｘ", "x")
            .replace("Ｘ", "x")
        )

    raw_final = extract_final(text)
    if not raw_final or not str(raw_final).strip():
        return None, "missing <final> tag"

    cand = _clean(raw_final).strip()
    match = re.search(r"\bx\s*=\s*(.+)", cand, flags=re.IGNORECASE)
    if not match:
        return None, "missing '='"
    rhs = match.group(1).strip()
    if not rhs:
        return None, "missing rhs"
    rhs_line = rhs.splitlines()[0].strip()
    try:
        val = sympify(rhs_line)
        return float(val.evalf()), None
    except Exception as exc:  # noqa: BLE001
        return None, f"failed to parse rhs: {exc}"


def parse_numeric_value(text: str):
    """兼容旧逻辑，仅返回解析出的数值或 None。"""
    value, _ = parse_equation_value(text)
    return value


def answers_match(pred_text: str, gold_value) -> bool:
    """
    使用 Sympy 解析模型输出（等号右侧），与 gold result_float 数值对比。
    gold_value 为数据集的 result_float。
    """
    if gold_value is None:
        return False
    pred_num, parse_error = parse_equation_value(pred_text)
    try:
        gold_num = float(gold_value)
    except Exception:
        return False
    if parse_error or pred_num is None:
        return False
    return abs(pred_num - gold_num) < 1e-6


# ============== 模型加载相关 ==============


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
    # 1. 加载 Calc-ape210k 数据集（本地 parquet）
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

    constraint_content = r'<final>x = [+-]?(\d+\.?\d*|\.\d+)\s*</final>'
    constraint_type = "regex"

    # 3. 遍历前 num_samples 个样本
    n = min(num_samples, len(ds))
    print(f"\n=== Start generation (first {n} samples) ===\n")

    example_problem, example_equation = pick_example(ds)
    correct = 0
    total_with_gold = 0
    syntax_errors = 0
    result_errors = 0

    for idx in range(n):
        sample = ds[idx]
        # Prefer English question if available, otherwise fall back to Chinese.
        problem = sample.get("question") or sample.get("question_chinese")
        gold_answer = sample.get("result_float")
        sample_id = sample.get("id", f"idx-{idx}")

        prompt = build_prompt(problem, example_problem, example_equation)

        print("#" * 80)
        print(f"[Sample {idx}] id = {sample_id}")
        print("- Prompt preview (truncated) -")
        print(textwrap.shorten(prompt, width=300, placeholder=" ..."))
        print("-" * 80)

        sample_log = {
            "task": "ape21",
            "model_name_or_path": model_name_or_path,
            "data_file": data_file,
            "split": split,
            "mode": mode,
            "sample_id": sample_id,
            "gold_result_float": gold_answer,
            "prompt_preview": prompt[:300] + "..." if len(prompt) > 300 else prompt,
        }

        # 4. Generate（统一调用约束生成器）
        output_text = generate_with_generator(
            generator,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            constraint=constraint_content,
            constraint_type=constraint_type,
        )

        print("Model output:")
        print(output_text.strip())

        sample_log["raw_output"] = output_text
        if gold_answer is not None:
            try:
                gold_value = float(gold_answer)
            except Exception:
                gold_value = None
            if gold_value is None:
                print("[gold result_float] invalid, skip scoring for this sample")
                sample_log["evaluated"] = False
                sample_log["test_status"] = "skipped"
                sample_log["test_error"] = "invalid gold result_float"
                if log_file:
                    append_jsonl_log(log_file, sample_log, verbose=False)
                print("\n")
                continue

            total_with_gold += 1
            pred_value, parse_error = parse_equation_value(output_text)
            sample_log["parsed_value"] = pred_value
            sample_log["parse_error"] = parse_error

            if parse_error:
                syntax_errors += 1
                print(f"[parse status] syntax error ({parse_error})")
                print(f"[gold result_float] {gold_value}")
                sample_log["evaluated"] = True
                sample_log["test_status"] = "syntax_error"
                sample_log["test_error"] = str(parse_error)
            else:
                is_correct = abs(pred_value - gold_value) < 1e-6
                print(f"[parsed pred value] {pred_value}")
                print(f"[gold result_float] {gold_value}")
                if is_correct:
                    correct += 1
                    print("[match] correct")
                    sample_log["evaluated"] = True
                    sample_log["test_status"] = "correct"
                else:
                    result_errors += 1
                    print("[match] wrong (result error)")
                    sample_log["evaluated"] = True
                    sample_log["test_status"] = "wrong"
                    sample_log["test_error"] = f"pred={pred_value}, gold={gold_value}"

            if log_file:
                append_jsonl_log(log_file, sample_log, verbose=False)

        print("\n")

    if total_with_gold:
        acc = correct / total_with_gold
        print("=" * 80)
        print(f"Accuracy: {correct}/{total_with_gold} = {acc:.2%}")
        print(f"Syntax errors (invalid 'x=...'): {syntax_errors}")
        print(f"Result errors (parsed but mismatched): {result_errors}")

        if log_file:
            summary_log = {
                "task": "ape21_summary",
                "model_name_or_path": model_name_or_path,
                "data_file": data_file,
                "split": split,
                "mode": mode,
                "constraint_template": constraint_template,
                "num_samples": n,
                "evaluated": total_with_gold,
                "correct": correct,
                "accuracy": f"{acc:.4f}",
                "syntax_errors": syntax_errors,
                "result_errors": result_errors,
            }
            append_jsonl_log(log_file, summary_log, verbose=False)

        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run generation on Calc-ape210k dataset with a transformers model.",
    )
    parser = add_common_model_args(parser)
    parser = add_common_gen_args(parser)
    parser.set_defaults(max_new_tokens=128)
    parser.add_argument(
        "--data_file",
        type=str,
        default="datasets/Calc-ape210k/data/train-00000-of-00001-b9f022a8492442e4.parquet",
        help="Calc-ape210k data file (parquet).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split; must match the key in data_files. Default: train.",
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

    run_generation(
        model_name_or_path=args.model_name_or_path,
        data_file=args.data_file,
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
        mode=getattr(args, "mode", "prompt"),
        log_file=getattr(args, "log_file", None),
    )


if __name__ == "__main__":
    main()
