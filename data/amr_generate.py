#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
amr_generate.py

在 amr-3-parsed 数据集上调用任意 transformers 模型进行生成：
- 从本地 parquet 数据加载 conversations（user -> assistant）
- 将 user 的 content 作为 prompt 输入模型，生成 AMR 文本

用法示例（在 datacode 目录下执行）：

python -m data.amr_generate \
  --model_name_or_path ../Models/openPangu-Embedded-1B \
  --data_file ../datasets/amr-3-parsed/data/validation-00000-of-00001.parquet \
  --split validation \
  --num_samples 1 \
  --max_new_tokens 256 \
  --device cuda \
  --temperature 0
"""

import argparse
import io
import re
import textwrap

try:
    import penman
    from penman import PenmanError
except ImportError:  # pragma: no cover - optional dependency
    penman = None
    PenmanError = Exception

try:
    import smatch
except ImportError:  # pragma: no cover - optional dependency
    smatch = None

from datasets import load_dataset

from common.cli import add_common_gen_args
from common.text_model_utils import (
    add_common_model_args,
    load_text_model_and_tokenizer,
)
from common.constrained_generation import ConstrainedGenerator
from common.cg_utils import load_template, generate_with_generator, append_jsonl_log


# ============== Prompt ==============

PROMPT_TEMPLATE = """You are an expert AMR parsing assistant.
Please generate the Abstract Meaning Representation (AMR) graph for the input sentence following strict Penman notation rules:

1. **Structure**: Every node must follow the format `(variable / concept :relation child)`.
2. **Root**: The graph must start with a single root parenthesis `(`.
3. **Relations**: All relations must start with a colon (e.g., `:ARG0`, `:time`, `:mod`) followed by a space.
4. **Variables**: Assign unique variable names (e.g., `p`, `p2`) to each concept.
5. **Output**: Output **ONLY** the raw AMR string. Do not use Markdown code blocks (```), and do not add explanations.

Example:
User: {example_user}
AMR:
{example_amr}

Now parse the new sentence.
User: {user_msg}

AMR:
"""


def extract_user_and_amr(sample):
    """从 conversations 中提取 user 问句和 gold AMR。"""
    conversations = sample.get("conversations", [])
    user_msg = ""
    gold_amr = None
    if isinstance(conversations, list):
        if len(conversations) > 0:
            user_msg = conversations[0].get("content", "")
        if len(conversations) > 1:
            gold_amr = conversations[1].get("content")
    return user_msg.strip(), gold_amr


def pick_example(ds):
    """从数据集中挑一条示例用于提示词。"""
    if len(ds) == 0:
        return ("", "")
    user_msg, gold_amr = extract_user_and_amr(ds[0])
    return user_msg, gold_amr or ""


def build_prompt(sample, example_user: str, example_amr: str) -> str:
    """构造带有示例的 prompt。"""
    user_msg, _ = extract_user_and_amr(sample)
    return PROMPT_TEMPLATE.format(
        example_user=example_user,
        example_amr=example_amr,
        user_msg=user_msg,
    )


# ============== Post-process ==============

SMATCH_EQ_THRESHOLD = 0.999


def _slice_last_top_level_paren_block(text: str) -> str:
    """截取最后一个最外层平衡括号的子串（含括号），无法匹配则返回原文。"""
    depth = 0
    start = None
    last_span = None
    for idx, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == ")":
            if depth == 0:
                # 未匹配的右括号，跳过
                continue
            depth -= 1
            if depth == 0 and start is not None:
                last_span = (start, idx + 1)
                start = None
    if last_span is not None:
        s, e = last_span
        return text[s:e]
    return text


def _normalize_whitespace(text: str) -> str:
    """将连续空格/换行压缩为单个空格。"""
    return re.sub(r"[ \t\r\n]+", " ", text).strip()


def postprocess_model_output(text: str) -> str:
    """模型输出先截取末尾的最外层括号对，再压缩空白。"""
    return _normalize_whitespace(_slice_last_top_level_paren_block(text))


def postprocess_gold_answer(text: str) -> str:
    """gold answer 仅做空白压缩。"""
    return _normalize_whitespace(text)


def _parse_amr(text: str):
    """使用 penman 解析 AMR，解析失败返回 None。"""
    if penman is None:
        return None
    try:
        return penman.decode(text)
    except (PenmanError, ValueError):
        return None
    except Exception:
        return None


def _smatch_f1(pred_graph, gold_graph) -> float | None:
    """计算 smatch F1 分数（需 penman+smatch），失败返回 None。"""
    if smatch is None or penman is None or pred_graph is None or gold_graph is None:
        return None
    try:
        pred_str = penman.encode(pred_graph, indent=None)
        gold_str = penman.encode(gold_graph, indent=None)
    except Exception:
        return None

    smatch.single_score = True
    smatch.pr_flag = False
    smatch.verbose = False
    smatch.veryVerbose = False

    try:
        scores = list(
            smatch.score_amr_pairs(
                io.StringIO(pred_str + "\n"),
                io.StringIO(gold_str + "\n"),
            )
        )
    except Exception:
        return None
    if not scores:
        return None
    _, _, f1 = scores[-1]
    return f1


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
    constraint_template: str | None = None,
    log_file: str | None = None,
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
    print(f"repetition_penalty = {repetition_penalty}")
    print(f"frequency_penalty  = {frequency_penalty}")

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
    constraint_content, constraint_type = load_template(constraint_template)

    # 3. 遍历前 num_samples 个样本
    if num_samples == -1:
        n = len(ds)
    else:
        n = min(num_samples, len(ds))
    print(f"\n=== Start generation (first {n} samples) ===\n")

    example_user, example_amr = pick_example(ds)
    correct = 0
    evaluated = 0
    syntax_errors = 0
    semantic_errors = 0
    smatch_sum = 0.0
    smatch_count = 0
    smatch_missing = 0
    parser_ready = penman is not None
    smatch_ready = parser_ready and smatch is not None

    if not parser_ready:
        print("Warning: penman not installed; syntax/semantic validation skipped (fall back to string exact match).")
    elif not smatch_ready:
        print("Warning: smatch not installed; semantic equivalence will fall back to string exact match.")

    for idx in range(n):
        sample = ds[idx]
        prompt = build_prompt(sample, example_user, example_amr)
        _, gold_answer = extract_user_and_amr(sample)
        sample_id = sample.get("id", f"idx-{idx}")


        sample_log = {
            "task": "amr",
            "model_name_or_path": model_name_or_path,
            "data_file": data_file,
            "split": split,
            "mode": mode,
            "constraint_template": constraint_template,
            "sample_id": sample_id,
            "prompt_preview": prompt[:300] + "..." if len(prompt) > 300 else prompt,
            "gold_answer": gold_answer,
        }

        print("#" * 80)
        print(f"[Sample {idx}] id = {sample_id}")
        print("- Prompt preview (truncated) -")
        print(textwrap.shorten(prompt, width=300, placeholder=" ..."))
        print("-" * 80)

        # 4. 生成（统一调用约束生成器）
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
        if gold_answer is not None:
            evaluated += 1
            processed_output = postprocess_model_output(output_text)
            processed_gold = postprocess_gold_answer(gold_answer)
            pred_graph = _parse_amr(processed_output)
            gold_graph = _parse_amr(processed_gold)
            smatch_score = _smatch_f1(pred_graph, gold_graph)

            if smatch_ready:
                smatch_count += 1
                if smatch_score is None:
                    smatch_missing += 1
                else:
                    smatch_sum += smatch_score

            is_correct = False
            if parser_ready and pred_graph is None:
                syntax_errors += 1
            else:
                if smatch_score is not None:
                    is_correct = smatch_score >= SMATCH_EQ_THRESHOLD
                    if pred_graph is not None and smatch_score < SMATCH_EQ_THRESHOLD:
                        semantic_errors += 1
                elif not smatch_ready or not parser_ready or gold_graph is None:
                    is_correct = processed_output == processed_gold

            sample_log["is_correct"] = bool(is_correct)
            sample_log["parsed_pred"] = pred_graph is not None
            sample_log["smatch_score"] = smatch_score

            correct += int(is_correct)
            print("[gold answer]")
            print(gold_answer)
            print("[post-processed]")
            print(f"model: {processed_output}")
            print(f"gold : {processed_gold}")
            print(
                "[parser] pred: {pred}, gold: {gold}, smatch: {smatch}".format(
                    pred="ok" if pred_graph else "fail",
                    gold="ok" if gold_graph else "fail",
                    smatch=f"{smatch_score:.4f}" if smatch_score is not None else "n/a",
                )
            )
            print(f"[match] {'correct' if is_correct else 'wrong'}")

        # 写入逐样本日志（jsonl）
        if log_file:
            sample_log["raw_output"] = output_text
            sample_log["post_processed_output"] = postprocess_model_output(output_text)
            if gold_answer is not None:
                sample_log["post_processed_gold"] = postprocess_gold_answer(gold_answer)
            append_jsonl_log(log_file, sample_log, verbose=False)
        print("\n")

    if evaluated > 0:
        acc = correct / evaluated
        print("=" * 80)
        print(f"Checked samples (with gold): {evaluated}")
        print(f"Accuracy: {correct}/{evaluated} = {acc:.2%}")


        if log_file:
            summary_log = {
                "task": "amr_summary",
                "model_name_or_path": model_name_or_path,
                "data_file": data_file,
                "split": split,
                "mode": mode,
                "constraint_template": constraint_template,
                "num_samples": n,
                "evaluated": evaluated,
                "correct": correct,
                "accuracy": f"{acc:.4f}",
                "syntax_errors": syntax_errors,
                "semantic_errors": semantic_errors,
                "parser_ready": bool(parser_ready),
                "smatch_ready": bool(smatch_ready),
            }
            append_jsonl_log(log_file, summary_log, verbose=False)

        if parser_ready:
            print(f"Syntax errors: {syntax_errors}")
            if smatch_ready:
                print(f"Semantic errors: {semantic_errors}")
                if smatch_count > 0:
                    avg_smatch = smatch_sum / smatch_count
                    print(f"Average Smatch (missing as 0): {avg_smatch:.4f}")
                    if smatch_missing > 0:
                        print(f"Smatch missing (treated as 0): {smatch_missing}")
        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AMR generation on amr-3-parsed with a transformers model.",
    )
    parser = add_common_model_args(parser)
    parser = add_common_gen_args(parser)
    parser.add_argument(
        "--data_file",
        type=str,
        default="datasets/amr-3-parsed/data/validation-00000-of-00001.parquet",
        help="amr-3-parsed data file (parquet).",
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
        default=-1,
        help="Number of samples to generate from the dataset.",
    )
    parser.set_defaults(max_new_tokens=512)
    parser.add_argument(
        "--mode",
        type=str,
        default="prompt",
        choices=["prompt", "xgrammar", "guidance", "outlines", "llama_cpp"],
        help="生成模式：prompt（仅 prompt）、xgrammar、guidance、outlines、llama_cpp。",
    )
    parser.add_argument(
        "--constraint_template",
        type=str,
        default=None,
        help="约束模板路径：.json(JSON Schema)/.cfg(CFG)/.ebnf/.regex 等。",
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
        mode=args.mode,
        constraint_template=args.constraint_template,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
