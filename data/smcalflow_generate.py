#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
smcalflow_generate.py

在 UDR_SMCalFlow 数据集上调用任意 transformers 模型进行生成：
- 从本地 parquet 数据加载 user_utterance
- 让模型生成对应的 lispress（语义解析结果）

用法示例（在 datacode 目录下执行）：

python -m data.smcalflow_generate \
  --model_name_or_path ../Models/MiniCPM4-0.5B \
  --data_file ../datasets/UDR_SMCalFlow/data/validation-00000-of-00001-4c17fe465006b08d.parquet \
  --split validation \
  --num_samples 10 \
  --max_new_tokens 512 \
  --device cuda \
  --temperature 0
"""

import argparse
import textwrap

from datasets import load_dataset

from dataflow.core.lispress import parse_lispress, render_compact, lispress_to_program
from dataflow.core.program import program_to_lispress

from common.cli import add_common_gen_args
from common.text_model_utils import (
    add_common_model_args,
    build_model_inputs as shared_build_model_inputs,
    generate as shared_generate,
    load_text_model_and_tokenizer as shared_load_text_model_and_tokenizer,
)


# ============== Prompt 模板 ==============

PROMPT_TEMPLATE = """You are a semantic parsing assistant.
Follow the example and output ONLY the Lispress program.

Example:
User: {example_user}
Lispress: {example_lispress}

Now parse the new utterance.
User: {utterance}

Lispress:
"""


def pick_example(ds):
    """从数据集中挑一条示例（user_utterance + lispress）。"""
    if len(ds) == 0:
        return ("", "")
    ex = ds[0]
    return ex.get("user_utterance", ""), ex.get("lispress", "")


def build_prompt(utterance: str, example_user: str, example_lispress: str) -> str:
    """把 user_utterance 填入 prompt，并包含一个示例。"""
    return PROMPT_TEMPLATE.format(
        example_user=example_user.strip(),
        example_lispress=(example_lispress or "").strip(),
        utterance=utterance.strip(),
    )


# ============== 文本后处理 ==============

def extract_first_balanced_parentheses(text: str):
    """查找第一段最外层平衡括号（含括号）。"""
    if not text:
        return None
    start = None
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            if depth == 0 and start is None:
                start = idx
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : idx + 1]
    return None


def normalize_lispress_with_dataflow(text: str) -> str:
    """
    使用 Dataflow Lispress 解析和规范化：
    1) 抽取首个平衡括号片段（若有），否则使用原文本
    2) parse_lispress -> lispress_to_program -> program_to_lispress
    3) render_compact 生成规范字符串
    """
    extracted = extract_first_balanced_parentheses(str(text)) if text is not None else None
    candidate = extracted if extracted is not None else ("" if text is None else str(text))
    lispress = parse_lispress(candidate)
    program, _ = lispress_to_program(lispress, indexicals={})
    return render_compact(program_to_lispress(program))


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
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
):
    # 1. 加载数据集
    ds_dict = load_dataset(
        "parquet",
        data_files={split: data_file},
    )
    assert split in ds_dict, f"split {split!r} not found in data file; available: {list(ds_dict.keys())}"
    ds = ds_dict[split]

    print(f"Loaded dataset file: {data_file} [{split}]")
    print(f"num_rows = {len(ds)}")

    # 2. 加载模型 & tokenizer
    model, tokenizer, config = shared_load_text_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    is_encoder_decoder = getattr(config, "is_encoder_decoder", False)

    # 3. 遍历前 num_samples 个样本
    n = min(num_samples, len(ds))
    print(f"\n=== Start generation (first {n} samples) ===\n")

    example_user, example_lispress = pick_example(ds)
    correct = 0
    syntax_errors = 0
    semantic_errors = 0

    for idx in range(n):
        sample = ds[idx]
        utterance = sample["user_utterance"]
        gold_lispress = sample.get("lispress")
        sample_id = sample.get("idx", f"idx-{idx}")

        prompt = build_prompt(utterance, example_user, example_lispress)

        print("#" * 80)
        print(f"[Sample {idx}] id = {sample_id}")
        print("- Prompt preview (truncated) -")
        print(textwrap.shorten(prompt, width=300, placeholder=" ..."))
        print("-" * 80)

        model_inputs = shared_build_model_inputs(
            tokenizer,
            prompt,
            is_encoder_decoder=is_encoder_decoder,
        )

        # 4. Generate
        output_text = shared_generate(
            model,
            tokenizer,
            model_inputs,
            is_encoder_decoder=is_encoder_decoder,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
        )

        print("Model output:")
        print(output_text.strip())
        if gold_lispress is not None:
            try:
                normalized_gold = normalize_lispress_with_dataflow(gold_lispress)
            except Exception as e:
                print("[gold parse error]")
                print(gold_lispress)
                print(e)
                print("[skip comparison due to gold parse error]")
                print("\n")
                continue

            try:
                normalized_output = normalize_lispress_with_dataflow(output_text)
                is_correct = normalized_output == normalized_gold
                if is_correct:
                    correct += 1
                else:
                    semantic_errors += 1
                print("[gold lispress]")
                print(gold_lispress)
                print("[normalized output]")
                print(normalized_output)
                print("[normalized gold]")
                print(normalized_gold)
                print(f"[match] {'correct' if is_correct else 'wrong'}")
            except Exception as e:
                syntax_errors += 1
                print("[parse error: model output cannot be parsed]")
                print(e)
        print("\n")

    if n > 0:
        acc = correct / n if n else 0.0
        print("=" * 80)
        print(f"Accuracy: {correct}/{n} = {acc:.2%}")
        print(f"Syntax errors (unparsable outputs): {syntax_errors}")
        print(f"Semantic errors (parsed but mismatch): {semantic_errors}")
        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Lispress generation on UDR_SMCalFlow with a transformers model.",
    )
    parser = add_common_model_args(parser)
    parser = add_common_gen_args(parser)
    parser.set_defaults(max_new_tokens=512)
    parser.add_argument(
        "--data_file",
        type=str,
        default="datasets/UDR_SMCalFlow/data/validation-00000-of-00001-4c17fe465006b08d.parquet",
        help="UDR_SMCalFlow data file (parquet).",
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
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
    )


if __name__ == "__main__":
    main()
