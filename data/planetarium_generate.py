#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
planetarium_generate.py

在 Planetarium 数据集上调用任意 transformers 模型进行生成：
- 从本地 parquet 数据加载自然语言描述（初始状态 + 目标）
- 为每条描述构造 prompt，要求输出对应的 PDDL problem
- 使用 transformers 的模型直接生成（无约束）

用法示例（在 datacode 目录下执行）：

python -m data.planetarium_generate \
  --model_name_or_path ../Models/MiniCPM4-0.5B \
  --data_file ../datasets/planetarium/data/test-00000-of-00001.parquet \
  --split test \
  --num_samples 300 \
  --max_new_tokens 1024 \
  --device cuda \
  --temperature 0 | tee planetarium#3_cpm05b.txt
"""

import argparse
import textwrap

from datasets import load_dataset

from common.cli import add_common_gen_args, add_common_model_args
from common.text_model_utils import load_text_model_and_tokenizer
from common.constrained_generation import ConstrainedGenerator
from common.cg_utils import load_template, generate_with_generator, append_jsonl_log

try:
    from tarski.io import PDDLReader
    from tarski.syntax import land
except ImportError:
    PDDLReader = None
    land = None

# ============== Prompt 模板 ==============

PROMPT_TEMPLATE = """You are a planning assistant.
Follow the example and write the corresponding PDDL problem definition.

Example:
Description: {example_nl}
PDDL:
{example_pddl}

Now generate PDDL for the new description.
Problem description:
{nl}

PDDL:
"""


def pick_example(ds):
    """从数据集中挑一个示例（自然语言 + PDDL）。"""
    if len(ds) == 0:
        return ("", "")
    ex = ds[0]
    return ex.get("natural_language", ""), ex.get("problem_pddl", "")


def build_prompt(nl: str, example_nl: str, example_pddl: str) -> str:
    """把 Planetarium 的自然语言描述嵌入统一 prompt，并加入示例。"""
    example_pddl_preview = textwrap.shorten(
        (example_pddl or "").strip(),
        width=600,
        placeholder=" ...",
    )
    return PROMPT_TEMPLATE.format(
        example_nl=example_nl.strip(),
        example_pddl=example_pddl_preview,
        nl=nl.strip(),
    )


# ============== 文本后处理 ==============

def extract_last_balanced_parentheses_span(text: str):
    """提取最后一对最外层且平衡的括号片段，避免前缀括号干扰。"""
    if not text:
        return None
    last_span = None
    stack_depth = 0
    current_start = None
    for idx, ch in enumerate(text):
        if ch == "(":
            if stack_depth == 0:
                current_start = idx
            stack_depth += 1
        elif ch == ")":
            if stack_depth == 0:
                continue  # 忽略不匹配的右括号
            stack_depth -= 1
            if stack_depth == 0 and current_start is not None:
                last_span = (current_start, idx)
                current_start = None
    if last_span is None:
        return None
    start, end = last_span
    return text[start : end + 1]


def extract_pddl_block(text: str) -> str:
    """保留括号包裹的 PDDL 片段，去除前后前缀文本。"""
    if text is None:
        return ""
    extracted = extract_last_balanced_parentheses_span(str(text))
    return extracted.strip() if extracted is not None else str(text).strip()


# ============== PDDL 解析与等价判断 ==============

class PDDLParseError(ValueError):
    """PDDL 解析错误。"""


DOMAIN_PDDL_TEXT = {
    "blocksworld": """(define (domain blocksworld)
  (:requirements :strips)
  (:predicates
    (on ?x ?y)
    (on-table ?x)
    (clear ?x)
    (holding ?x)
    (arm-empty)
  )
)""",
    "gripper": """(define (domain gripper)
  (:requirements :strips)
  (:predicates
    (at ?b ?r)
    (at-robby ?r)
    (carry ?g ?b)
    (free ?g)
    (ball ?b)
    (gripper ?g)
    (room ?r)
  )
)""",
    "floor-tile": """(define (domain floor-tile)
  (:requirements :typing)
  (:types robot tile color)
  (:predicates
    (robot-at ?r - robot ?t - tile)
    (robot-has ?r - robot ?c - color)
    (available-color ?r - robot ?c - color)
    (painted ?t - tile ?c - color)
    (right ?t1 - tile ?t2 - tile)
    (up ?t1 - tile ?t2 - tile)
  )
)""",
}

def require_tarski():
    if PDDLReader is None or land is None:
        raise RuntimeError("tarski is not installed. Please install with: pip install tarski")


def parse_problem_with_tarski(domain_name: str, pddl_text: str):
    require_tarski()
    if not pddl_text or not str(pddl_text).strip():
        raise PDDLParseError("empty input")
    domain_key = (domain_name or "").strip().lower()
    if domain_key not in DOMAIN_PDDL_TEXT:
        raise PDDLParseError(f"unknown domain: {domain_name!r}")
    domain_text = DOMAIN_PDDL_TEXT[domain_key]
    reader = PDDLReader(raise_on_error=True)
    try:
        reader.parse_domain_string(domain_text)
        return reader.parse_instance_string(pddl_text)
    except Exception as exc:
        raise PDDLParseError(str(exc)) from exc


def canonicalize_pddl_problem(pddl_text: str, domain_name: str):
    problem = parse_problem_with_tarski(domain_name, pddl_text)
    objects = sorted((c.symbol, c.sort.name) for c in problem.language.constants())
    init_atoms = sorted(str(a) for a in problem.init.as_atoms())
    goal = problem.goal
    if goal is None:
        raise PDDLParseError("missing :goal")
    if getattr(goal, "connective", None) == land:
        goal_repr = ("and", sorted(str(g) for g in goal.subformulas))
    else:
        goal_repr = ("expr", str(goal))
    return (objects, init_atoms, goal_repr)


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
    # 1. 加载 Planetarium 数据集（本地 parquet）
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
    constraint_content, constraint_type = load_template(constraint_template)

    # 3. 遍历前 num_samples 个样本
    if num_samples == -1:
        n = len(ds)
    else:
        n = min(num_samples, len(ds))
    print(f"\n=== Start generation (first {n} samples) ===\n")

    example_nl, example_pddl = pick_example(ds)
    correct = 0
    syntax_error_count = 0
    semantic_error_count = 0

    for idx in range(n):
        sample = ds[idx]
        nl = sample["natural_language"]
        gold_pddl = sample.get("problem_pddl")
        domain_name = sample.get("domain", "")
        sample_id = sample.get("id", f"idx-{idx}")

        prompt = build_prompt(nl, example_nl, example_pddl)

        sample_log = {
            "task": "planetarium",
            "model_name_or_path": model_name_or_path,
            "data_file": data_file,
            "split": split,
            "mode": mode,
            "constraint_template": constraint_template,
            "sample_id": sample_id,
            "domain": domain_name,
            "nl": nl,
            "prompt_preview": prompt[:300] + "..." if len(prompt) > 300 else prompt,
            "gold_pddl": gold_pddl,
        }

        print("#" * 80)
        print(f"[Sample {idx}] id = {sample_id}")
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
            constraint=constraint_content,
            constraint_type=constraint_type,
        )

        print("Model output:")
        print(output_text.strip())

        sample_log["raw_output"] = output_text
        sample_log["pred_pddl"] = extract_pddl_block(output_text)

        if gold_pddl is not None:
            print("Gold answer：")
            print(str(gold_pddl).strip())
            pred_block = extract_pddl_block(output_text)
            gold_block = extract_pddl_block(gold_pddl)
            sample_log["gold_pddl_block"] = gold_block

            try:
                pred_canonical = canonicalize_pddl_problem(pred_block, domain_name)
                sample_log["parsed_pred"] = True
            except PDDLParseError as exc:
                syntax_error_count += 1
                sample_log["parsed_pred"] = False
                sample_log["parse_error_pred"] = str(exc)
                sample_log["is_correct"] = False
                sample_log["result_label"] = "syntax error"
                print(f"[parse] pred syntax error: {exc}")
                if log_file:
                    append_jsonl_log(log_file, sample_log, verbose=False)
                print("\n")
                continue

            try:
                gold_canonical = canonicalize_pddl_problem(gold_block, domain_name)
                sample_log["parsed_gold"] = True
            except PDDLParseError as exc:
                syntax_error_count += 1
                sample_log["parsed_gold"] = False
                sample_log["parse_error_gold"] = str(exc)
                sample_log["is_correct"] = False
                sample_log["result_label"] = "gold syntax error"
                print(f"[parse] gold syntax error: {exc}")
                if log_file:
                    append_jsonl_log(log_file, sample_log, verbose=False)
                print("\n")
                continue

            is_correct = pred_canonical == gold_canonical
            correct += int(is_correct)
            if not is_correct:
                semantic_error_count += 1
            sample_log["is_correct"] = bool(is_correct)
            sample_log["result_label"] = "correct" if is_correct else "semantic error"
            print(f"[semantic match] {'correct' if is_correct else 'wrong'}")

        # 写入逐样本日志（jsonl）
        if log_file:
            append_jsonl_log(log_file, sample_log, verbose=False)

        print("\n")

    if n > 0:
        acc = correct / n if n else 0.0
        print("=" * 80)
        print(f"Accuracy: {correct}/{n} = {acc:.2%}")
        print(f"Syntax errors: {syntax_error_count}")
        print(f"Semantic errors: {semantic_error_count}")

        if log_file:
            summary_log = {
                "task": "planetarium_summary",
                "model_name_or_path": model_name_or_path,
                "data_file": data_file,
                "split": split,
                "mode": mode,
                "constraint_template": constraint_template,
                "num_samples": n,
                "correct": correct,
                "accuracy": f"{acc:.4f}",
                "syntax_errors": syntax_error_count,
                "semantic_errors": semantic_error_count,
            }
            append_jsonl_log(log_file, summary_log, verbose=False)

        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run generation on Planetarium dataset with a transformers model.",
    )
    parser = add_common_model_args(parser)
    parser = add_common_gen_args(parser)
    parser.add_argument(
        "--data_file",
        type=str,
        default="datasets/planetarium/data/test-00000-of-00001.parquet",
        help="Planetarium data file (parquet).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split; must match the key in data_files. Default: test.",
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
