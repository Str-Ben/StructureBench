#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
bigmath_generate.py

在 Big-Math 数据集上调用任意 transformers 模型进行生成：
- 从本地 parquet 数据（SynthLabsAI/Big-Math-RL-Verified 导出）加载题目
- 为每个 problem 构造统一 prompt，让模型只输出最终答案
- 使用 transformers 的模型直接生成答案（无约束）

用法示例（在 datacode 目录下执行）：

python -m data.bigmath_generate \
  --model_name_or_path ../Models/Qwen3-0.6B \
  --data_file ../datasets/Big-Math-RL-Verified/data/train-00000-of-00001.parquet \
  --split train \
  --num_samples 10 \
  --max_new_tokens 512 \
  --device cuda \
  --temperature 0
"""

import argparse
import re
import textwrap
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple

from datasets import load_dataset

from common.cli import add_common_gen_args
from common.text_model_utils import (
    add_common_model_args,
    build_model_inputs,
    generate,
    load_text_model_and_tokenizer,
)

_LATEX_IMPORT_ERROR = None
try:
    import sympy as sp
    from sympy.parsing.latex import parse_latex
except Exception as exc:
    sp = None
    parse_latex = None
    _LATEX_IMPORT_ERROR = str(exc)


# ============== Prompt 模板 ==============

PROMPT_TEMPLATE = """You are a math expert.
Follow the example: first give a brief reasoning, then put the final answer inside <final>...</final> (no extra text after </final>).

Example:
Problem: {example_problem}
Reasoning: The quadratic inequality is solved by finding its roots and checking the interval; applying the condition narrows a to the required range.
Answer: <final>{example_answer}</final>

Now solve the new problem.
Problem:
{problem}

Reasoning and Answer:/no_think
"""


def build_prompt(problem: str, example_problem: str, example_answer: str) -> str:
    """把 Big-Math 的 problem 文本嵌入统一 prompt，并加入示例。"""
    return PROMPT_TEMPLATE.format(
        example_problem=example_problem.strip(),
        example_answer=example_answer.strip(),
        problem=problem.strip(),
    )


def pick_example(ds):
    """取一条示例用于 prompt。"""
    if len(ds) == 0:
        return ("", "")
    ex = ds[0]
    return ex.get("problem", ""), str(ex.get("answer", ""))


FINAL_PATTERN = re.compile(r"<final>(.*?)</final>", flags=re.IGNORECASE | re.DOTALL)
UNSUPPORTED_TOKEN_RE = re.compile(
    r"\\text\{or\}|\\pm|\\begin\{cases\}|\\cases|\\begin\{array\}|\\\{|"
    r"\\leqslant|\\geqslant|\\leq|\\geq|\\le|\\ge|<|>|\\lt|\\gt|\\neq"
)
INTERVAL_BOUND_RE = re.compile(r"^[\[(].*,.*[\])]$")


def extract_final(text: str) -> str:
    """提取 <final>...</final> 之间的内容；若未找到则返回原文。"""
    if text is None:
        return ""
    match = FINAL_PATTERN.search(text)
    return match.group(1) if match else text


def extract_final_with_flag(text: str) -> Tuple[str, bool]:
    """提取 <final>...</final>，并返回是否成功匹配。"""
    if text is None:
        return "", False
    match = FINAL_PATTERN.search(text)
    return (match.group(1), True) if match else (text, False)


def normalized_final(text: str) -> str:
    """提取 final 段并去除空白。"""
    return normalize_answer(extract_final(text))


def normalize_answer(text: str) -> str:
    """去除所有空白用于精确匹配。"""
    if text is None:
        return ""
    return re.sub(r"\s+", "", text)


def _strip_math_wrappers(text: str) -> str:
    s = text.strip()
    if s.startswith("$$") and s.endswith("$$") and len(s) > 3:
        return s[2:-2].strip()
    if s.startswith("$") and s.endswith("$") and len(s) > 1:
        return s[1:-1].strip()
    return s


def _cleanup_latex(text: str) -> str:
    s = _strip_math_wrappers(text)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\displaystyle", "")
    s = re.sub(r"\\(?:,|;|!|quad|qquad)", "", s)
    return re.sub(r"\s+", "", s)


def _prepare_latex(text: str) -> str:
    if text is None:
        return ""
    return _cleanup_latex(str(text))


def _contains_unsupported(text: str) -> bool:
    return bool(UNSUPPORTED_TOKEN_RE.search(text))


def _is_interval_candidate(text: str) -> bool:
    if "\\cup" in text:
        return True
    if re.search(r"\\in(?!fty)", text):
        return True
    return bool(INTERVAL_BOUND_RE.match(text))


def _classify_answer(text: str) -> Optional[str]:
    if not text:
        return None
    if _contains_unsupported(text):
        return None
    if _is_interval_candidate(text):
        return "interval"
    if text.count("=") == 1:
        return "equation"
    if text.count("=") > 1:
        return None
    return "expression"


def gold_is_evaluable(text: str) -> Tuple[bool, str]:
    """判断 gold 是否支持并可解析；不可解析则跳过该样本。"""
    gold = _prepare_latex(text)
    if not gold:
        return False, "empty answer"
    gold_type = _classify_answer(gold)
    if gold_type is None:
        return False, "unsupported format"
    if gold_type == "expression":
        expr, err = _try_parse_latex_expr(gold)
        return (expr is not None), (err or "")
    if gold_type == "equation":
        if gold.count("=") != 1:
            return False, "invalid equation format"
        lhs, rhs = gold.split("=", 1)
        lhs_expr, lhs_err = _try_parse_latex_expr(lhs)
        if lhs_expr is None:
            return False, lhs_err or "equation lhs parse error"
        rhs_expr, rhs_err = _try_parse_latex_expr(rhs)
        if rhs_expr is None:
            return False, rhs_err or "equation rhs parse error"
        return True, ""
    if gold_type == "interval":
        parsed, err = _parse_interval_set(gold)
        return (parsed is not None), (err or "")
    return False, "unknown format"


def _try_parse_latex_expr(latex_str: str) -> Tuple[Optional["sp.Expr"], Optional[str]]:
    if parse_latex is None or sp is None:
        reason = _LATEX_IMPORT_ERROR or "sympy parse_latex unavailable"
        return None, f"latex parser unavailable: {reason}"
    try:
        return parse_latex(latex_str), None
    except Exception as exc:
        return None, f"parse error: {exc}"


def _simplify_expr(expr: "sp.Expr") -> "sp.Expr":
    expr = sp.expand(expr)
    expr = sp.together(expr)
    expr = sp.cancel(expr)
    expr = sp.trigsimp(expr)
    return sp.simplify(expr)


def _is_zero_expr(expr: "sp.Expr") -> bool:
    if expr == 0:
        return True
    return expr.is_zero is True


def _is_nonzero_constant(expr: "sp.Expr") -> bool:
    if expr is None:
        return False
    if expr.free_symbols:
        return False
    if expr.is_number is False:
        return False
    if expr.is_zero is True:
        return False
    if expr.is_finite is False:
        return False
    return True


def _parse_interval_endpoint(text: str) -> Tuple[Optional["sp.Expr"], Optional[str]]:
    if sp is None:
        return None, "sympy unavailable"
    if text in ("\\infty", "+\\infty", "infty", "+infty"):
        return sp.oo, None
    if text in ("-\\infty", "-infty"):
        return -sp.oo, None
    return _try_parse_latex_expr(text)


def _parse_single_interval(text: str) -> Tuple[Optional["sp.Set"], Optional[str]]:
    if sp is None:
        return None, "sympy unavailable"
    if not text:
        return None, "empty interval text"
    if text[0] not in "[(" or text[-1] not in "])":
        return None, "not an interval literal"
    body = text[1:-1]
    if "," not in body:
        return None, "interval missing comma"
    left_text, right_text = body.split(",", 1)
    left_expr, left_err = _parse_interval_endpoint(left_text)
    if left_expr is None:
        return None, f"left endpoint parse failed: {left_err}"
    right_expr, right_err = _parse_interval_endpoint(right_text)
    if right_expr is None:
        return None, f"right endpoint parse failed: {right_err}"
    left_open = text[0] == "("
    right_open = text[-1] == ")"
    return sp.Interval(left_expr, right_expr, left_open=left_open, right_open=right_open), None


def _parse_interval_set(text: str) -> Tuple[Optional["sp.Set"], Optional[str]]:
    if sp is None:
        return None, "sympy unavailable"
    s = text
    if "\\in" in s:
        parts = s.split("\\in", 1)
        s = parts[1] if len(parts) > 1 else s
    if not s:
        return None, "empty interval after in"
    pieces = re.split(r"\\cup", s)
    intervals = []
    for piece in pieces:
        piece = piece.strip()
        interval, err = _parse_single_interval(piece)
        if interval is None:
            return None, err
        intervals.append(interval)
    if len(intervals) == 1:
        return intervals[0], None
    return sp.Union(*intervals), None


def _equiv_expressions(pred: str, gold: str) -> Tuple[bool, Optional[str]]:
    pred_expr, pred_err = _try_parse_latex_expr(pred)
    if pred_expr is None:
        return False, f"pred parse error: {pred_err}"
    gold_expr, gold_err = _try_parse_latex_expr(gold)
    if gold_expr is None:
        return False, f"gold parse error: {gold_err}"
    diff = _simplify_expr(pred_expr - gold_expr)
    return _is_zero_expr(diff), None


def _equiv_equations(pred: str, gold: str) -> Tuple[bool, Optional[str]]:
    pred_lhs, pred_rhs = pred.split("=", 1)
    gold_lhs, gold_rhs = gold.split("=", 1)
    pred_lhs_expr, pred_err = _try_parse_latex_expr(pred_lhs)
    if pred_lhs_expr is None:
        return False, f"pred parse error: {pred_err}"
    pred_rhs_expr, pred_err = _try_parse_latex_expr(pred_rhs)
    if pred_rhs_expr is None:
        return False, f"pred parse error: {pred_err}"
    gold_lhs_expr, gold_err = _try_parse_latex_expr(gold_lhs)
    if gold_lhs_expr is None:
        return False, f"gold parse error: {gold_err}"
    gold_rhs_expr, gold_err = _try_parse_latex_expr(gold_rhs)
    if gold_rhs_expr is None:
        return False, f"gold parse error: {gold_err}"
    f1 = _simplify_expr(pred_lhs_expr - pred_rhs_expr)
    f2 = _simplify_expr(gold_lhs_expr - gold_rhs_expr)
    if _is_zero_expr(f1) and _is_zero_expr(f2):
        return True, None
    if _is_zero_expr(f1) or _is_zero_expr(f2):
        return False, None
    if _is_zero_expr(_simplify_expr(f1 - f2)):
        return True, None
    ratio = _simplify_expr(f1 / f2)
    return _is_nonzero_constant(ratio), None


def _equiv_intervals(pred: str, gold: str) -> Tuple[bool, Optional[str]]:
    pred_set, pred_err = _parse_interval_set(pred)
    if pred_set is None:
        return False, f"pred parse error: {pred_err}"
    gold_set, gold_err = _parse_interval_set(gold)
    if gold_set is None:
        return False, f"gold parse error: {gold_err}"
    if pred_set == gold_set:
        return True, None
    try:
        eq = pred_set.equals(gold_set)
    except Exception:
        eq = None
    return bool(eq), None


def compare_answers(pred_text: str, gold_text: str) -> dict:
    pred_raw, pred_has_final = extract_final_with_flag(pred_text)
    if not pred_has_final:
        return {
            "equivalent": False,
            "reason": "missing <final> answer",
            "pred_parse_error": True,
            "gold_parse_error": False,
        }
    pred = _prepare_latex(pred_raw)
    gold = _prepare_latex(extract_final(gold_text))
    if not pred or not gold:
        return {
            "equivalent": False,
            "reason": "empty answer",
            "pred_parse_error": True,
            "gold_parse_error": True,
        }
    pred_type = _classify_answer(pred)
    gold_type = _classify_answer(gold)
    if pred_type is None or gold_type is None:
        return {
            "equivalent": False,
            "reason": "unsupported format",
            "pred_parse_error": pred_type is None,
            "gold_parse_error": gold_type is None,
        }
    if pred_type != gold_type:
        return {
            "equivalent": False,
            "reason": f"type mismatch: {pred_type} vs {gold_type}",
            "pred_parse_error": False,
            "gold_parse_error": False,
        }
    pred_parse_error = False
    gold_parse_error = False
    if pred_type == "expression":
        equivalent, err = _equiv_expressions(pred, gold)
    elif pred_type == "equation":
        equivalent, err = _equiv_equations(pred, gold)
    elif pred_type == "interval":
        equivalent, err = _equiv_intervals(pred, gold)
    else:
        return {"equivalent": False, "skipped": True, "reason": "unknown type"}
    if err:
        err_lower = err.lower()
        if "pred parse error" in err_lower:
            pred_parse_error = True
        if "gold parse error" in err_lower:
            gold_parse_error = True
        if "latex parser unavailable" in err_lower or "sympy unavailable" in err_lower:
            pred_parse_error = True
            gold_parse_error = True
        return {
            "equivalent": False,
            "reason": err,
            "pred_parse_error": pred_parse_error,
            "gold_parse_error": gold_parse_error,
        }
    return {
        "equivalent": equivalent,
        "reason": "",
        "pred_parse_error": pred_parse_error,
        "gold_parse_error": gold_parse_error,
    }


def answers_match(pred_text: str, gold_text: str) -> bool:
    """兼容旧接口：无法判等时返回 False。"""
    result = compare_answers(pred_text, gold_text)
    return result["equivalent"]


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
):
    # 1. 加载 Big-Math 数据集（本地 parquet）
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

    # 3. 遍历可解析样本，最多生成 num_samples 条
    target = min(num_samples, len(ds))
    print(f"\n=== Start generation (up to {target} parseable samples) ===\n")

    example_problem, example_answer = pick_example(ds)
    correct = 0
    evaluated = 0
    syntax_errors = 0
    mismatches = 0
    generated = 0

    for idx in range(len(ds)):
        if generated >= target:
            break
        sample = ds[idx]
        problem = sample["problem"]
        gold_answer = sample.get("answer")
        sample_id = sample.get("__index_level_0__", f"idx-{idx}")
        if gold_answer is None:
            continue
        gold_ok, gold_reason = gold_is_evaluable(str(gold_answer))
        if not gold_ok:
            print(f"[Sample {idx}] id = {sample_id} skipped: {gold_reason}")
            continue

        prompt = build_prompt(problem, example_problem, example_answer)

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

        print("Model output:")
        print(output_text.strip())
        if gold_answer is not None:
            result = compare_answers(output_text, str(gold_answer))
            evaluated += 1
            if result.get("pred_parse_error"):
                syntax_errors += 1
                print(f"[match] syntax error: {result['reason']}")
            else:
                is_correct = bool(result["equivalent"])
                correct += int(is_correct)
                if not is_correct:
                    mismatches += 1
                print(f"[match] {'correct' if is_correct else 'wrong'}")
        print("\n")
        generated += 1

    if generated > 0:
        acc = correct / evaluated if evaluated else 0.0
        print("=" * 80)
        print(f"Accuracy (evaluated): {correct}/{evaluated} = {acc:.2%}")
        print(f"Syntax errors (pred): {syntax_errors}/{generated}")
        print(f"Wrong but syntactically valid: {mismatches}/{generated}")
        print(f"Generated: {generated}/{target}")
        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run generation on Big-Math dataset with a transformers model.",
    )
    parser = add_common_model_args(parser)
    parser = add_common_gen_args(parser)
    parser.set_defaults(max_new_tokens=512)
    parser.add_argument(
        "--data_file",
        type=str,
        default="datasets/Big-Math-RL-Verified/data/train-00000-of-00001.parquet",
        help="Big-Math data file (parquet).",
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
    )


if __name__ == "__main__":
    main()
