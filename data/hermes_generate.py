#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
hermes_generate.py

在 Hermes function calling (json_mode_singleturn) 上调用任意 transformers 模型进行生成：
- 从本地 Hermes function calling 数据文件加载样本（无需在线下载）
- 将 conversations 中的 system + user 组合成统一 prompt
- 使用 transformers 的模型直接生成 JSON / 函数调用输出

用法示例（在 datacode 目录下执行）：

python -m data.hermes_generate \
  --model_name_or_path ../Models/openPangu-Embedded-1B \
  --data_file ../datasets/hermes-function-calling-v1/json-mode-singleturn.json \
  --split train \
  --num_samples 300 \
  --max_new_tokens 512 \
  --device cuda \
  --temperature 0 | tee hermes#3_pg.txt
"""

import argparse
import textwrap

from datasets import load_dataset

from common.cli import add_common_gen_args
from common.text_model_utils import (
    add_common_model_args,
    build_model_inputs,
    generate,
    load_text_model_and_tokenizer,
)


# ============== Prompt 模板 ==============

PROMPT_TEMPLATE = """You are a function-calling assistant. Follow the example and reply with JSON that matches the schema in the system prompt.


Example:
{example_system}

User: {example_user}
Assistant:
{example_answer}

Now answer the new request. DO NOT output any line breaks.
{system_msg}

User: {user_msg}

Assistant:(without line breaks)"""


def extract_fields(sample):
    """获取 system/user/assistant 内容。"""
    conversations = sample.get("conversations", [])
    system_msg = conversations[0]["value"] if len(conversations) > 0 else ""
    user_msg = conversations[1]["value"] if len(conversations) > 1 else ""
    assistant_msg = conversations[2]["value"] if len(conversations) > 2 else None
    return system_msg.strip(), user_msg.strip(), assistant_msg


def pick_example(ds):
    """选取一条示例用于提示词。"""
    if len(ds) == 0:
        return ("", "", "")
    system_msg, user_msg, assistant_msg = extract_fields(ds[0])
    return system_msg, user_msg, assistant_msg or ""


def build_prompt(sample, example_system: str, example_user: str, example_answer: str) -> str:
    """从 Hermes 数据集的单条样本构造 prompt，包含示例。"""
    system_msg, user_msg, _ = extract_fields(sample)
    return PROMPT_TEMPLATE.format(
        example_system=example_system,
        example_user=example_user,
        example_answer=example_answer,
        system_msg=system_msg,
        user_msg=user_msg,
    )


def extract_first_json_field(text: str):
    """截取生成文本中的第一个 json 片段（第一对花括号，右括号取全文最后一个）。"""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}")
    if end == -1:
        return None
    if end < start:
        return None
    return text[start : end + 1]


def normalize_json_snippet(text: str):
    """去掉首尾空白，并将换行符统一替换为空格。"""
    if text is None:
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\n", " ").strip()


def strip_all_whitespace(text: str):
    """移除所有空白符用于宽松比对。"""
    if text is None:
        return None
    return "".join(text.split())


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
    # 1. 加载 Hermes 本地数据集
    ds_dict = load_dataset(
        "json",
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
    )
    is_encoder_decoder = getattr(config, "is_encoder_decoder", False)

    # 3. 遍历前 num_samples 个样本
    n = min(num_samples, len(ds))
    print(f"\n=== Start generation (first {n} samples) ===\n")

    example_system, example_user, example_answer = pick_example(ds)
    correct = 0

    for idx in range(n):
        sample = ds[idx]
        sample_id = sample.get("id", f"idx-{idx}")

        prompt = build_prompt(sample, example_system, example_user, example_answer)
        _, _, gold_answer = extract_fields(sample)

        print("#" * 80)
        print(f"[Sample {idx}] id = {sample_id}")
        print("- Prompt preview (truncated) -")
        print(textwrap.shorten(prompt, width=300, placeholder=" ..."))
        print("-" * 80)

        model_inputs = build_model_inputs(
            tokenizer,
            prompt,
            is_encoder_decoder=is_encoder_decoder,
        )

        # 4. Generate
        output_text = generate(
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

        extracted_json = extract_first_json_field(output_text)

        print("Model output:")
        print(output_text.strip())
        if gold_answer is not None:
            print("[gold answer]")
            gold_answer_raw = gold_answer.strip()
            print(gold_answer_raw)
            if extracted_json is None:
                is_correct = False
                print("[extracted json]")
                print("<not found>")
            else:
                extracted_json = normalize_json_snippet(extracted_json)
                print("[extracted json]")
                print(extracted_json)
                gold_answer_comp = strip_all_whitespace(gold_answer_raw)
                extracted_json_comp = strip_all_whitespace(extracted_json)
                is_correct = extracted_json_comp == gold_answer_comp
            correct += int(is_correct)
            print(f"[match] {'correct' if is_correct else 'wrong'}")
        print("\n")

    if n > 0:
        acc = correct / n if n else 0.0
        print("=" * 80)
        print(f"Accuracy: {correct}/{n} = {acc:.2%}")
        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run generation on Hermes function calling dataset with a transformers model.",
    )
    parser = add_common_model_args(parser)
    parser = add_common_gen_args(parser)
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Local Hermes data file (json/jsonl), e.g., datasets/hermes-function-calling-v1/json-mode-singleturn.json.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="json_mode_singleturn",
        help="Config name to infer default data_file, e.g., json_mode_singleturn / json_mode_agentic / func_calling.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split, typically train.",
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

    data_file = args.data_file
    if data_file is None:
        file_name = args.dataset_config.replace("_", "-")
        if not file_name.endswith(".json"):
            file_name = f"{file_name}.json"
        data_file = f"datasets/hermes-function-calling-v1/{file_name}"

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
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
    )


if __name__ == "__main__":
    main()
