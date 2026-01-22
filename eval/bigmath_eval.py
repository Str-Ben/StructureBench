#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Big-Math evaluation runner.

Workflow:
1) Run data.text_models_generate.py to produce JSONL predictions.
2) Run this script with --pred_file and dataset args to score outputs.
3) The script loads dataset rows, aligns by sample_id, writes per-sample results,
   and prints a summary consistent with the original generate script.

Example:
python -m eval.bigmath_eval \
  --pred_file outputs/bigmath.jsonl \
  --output_file eval/bigmath_eval.jsonl \
  --data_file datasets/Big-Math-RL-Verified/data/train-00000-of-00001.parquet \
  --split train
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from datasets import load_dataset

from eval.eval_logging import append_eval_log
_LATEX_IMPORT_ERROR = None
try:
    import sympy as sp
    from sympy.parsing.latex import parse_latex
except Exception as exc:
    sp = None
    parse_latex = None
    _LATEX_IMPORT_ERROR = str(exc)


FINAL_PATTERN = re.compile(r"<final>(.*?)</final>", flags=re.IGNORECASE | re.DOTALL)
UNSUPPORTED_TOKEN_RE = re.compile(
    r"\\text\{or\}|\\pm|\\begin\{cases\}|\\cases|\\begin\{array\}|\\\{|"
    r"\\leqslant|\\geqslant|\\leq|\\geq|\\le|\\ge|<|>|\\lt|\\gt|\\neq"
)
INTERVAL_BOUND_RE = re.compile(r"^[\[(].*,.*[\])]$")


def _json_default(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _get_field(item: Any, key: str, default=None):
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def _index_predictions(records: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], int]:
    pred_map: Dict[str, Dict[str, Any]] = {}
    duplicates = 0
    for idx, record in enumerate(records):
        sample_id = _get_field(record, "sample_id", None)
        if sample_id is None:
            sample_id = f"idx-{idx}"
        sample_id = str(sample_id)
        if sample_id in pred_map:
            duplicates += 1
            continue
        pred_map[sample_id] = record
    return pred_map, duplicates


def _resolve_sample_id(sample: Mapping[str, Any], idx: int) -> str:
    return str(sample.get("__index_level_0__", f"idx-{idx}"))


def extract_final(text: str) -> str:
    if text is None:
        return ""
    match = FINAL_PATTERN.search(text)
    return match.group(1) if match else text


def extract_final_with_flag(text: str) -> Tuple[str, bool]:
    if text is None:
        return "", False
    match = FINAL_PATTERN.search(text)
    return (match.group(1), True) if match else (text, False)


def normalized_final(text: str) -> str:
    return normalize_answer(extract_final(text))


def normalize_answer(text: str) -> str:
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
    result = compare_answers(pred_text, gold_text)
    return result["equivalent"]


def run_evaluation(args) -> None:
    ds_dict = load_dataset("parquet", data_files={args.split: args.data_file})
    if args.split not in ds_dict:
        raise ValueError(f"split {args.split!r} not found in data file; available: {list(ds_dict.keys())}")
    ds = ds_dict[args.split]

    pred_map, duplicates = _index_predictions(_read_jsonl(args.pred_file))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append_output else "w"

    target = len(ds) if not args.num_samples or args.num_samples <= 0 else min(args.num_samples, len(ds))
    correct = 0
    evaluated = 0
    syntax_errors = 0
    mismatches = 0
    generated = 0
    missing_pred = 0

    with output_path.open(mode, encoding="utf-8") as fout:
        for idx in range(len(ds)):
            if generated >= target:
                break
            sample = ds[idx]
            gold_answer = sample.get("answer")
            sample_id = _resolve_sample_id(sample, idx)
            if gold_answer is None:
                continue
            gold_ok, gold_reason = gold_is_evaluable(str(gold_answer))
            if not gold_ok:
                if args.verbose:
                    print(f"[Sample {idx}] id={sample_id} skipped: {gold_reason}")
                continue

            pred_record = pred_map.get(sample_id)
            if pred_record is None:
                missing_pred += 1
                record = {
                    "dataset": "bigmath",
                    "sample_id": sample_id,
                    "status": "missing_pred",
                }
                fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                if args.verbose:
                    print(f"[Sample {idx}] id={sample_id} status=missing_pred")
                generated += 1
                continue

            output_text = str(_get_field(pred_record, "raw_output", "") or "")
            result = compare_answers(output_text, str(gold_answer))
            evaluated += 1
            if result.get("pred_parse_error"):
                syntax_errors += 1
                status = "syntax error"
            else:
                is_correct = bool(result["equivalent"])
                correct += int(is_correct)
                if not is_correct:
                    mismatches += 1
                status = "correct" if is_correct else "wrong"

            record = {
                "dataset": "bigmath",
                "sample_id": sample_id,
                "status": status,
                "reason": result.get("reason", ""),
                "pred_parse_error": result.get("pred_parse_error"),
                "gold_parse_error": result.get("gold_parse_error"),
            }
            fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

            if args.verbose:
                print(f"[Sample {idx}] id={sample_id} status={status}")
            generated += 1

    accuracy = correct / evaluated if evaluated else None
    semantic_errors = mismatches
    if generated > 0:
        acc = accuracy or 0.0
        print("=" * 80)
        print(f"Accuracy (evaluated): {correct}/{evaluated} = {acc:.2%}")
        print(f"Syntax errors (pred): {syntax_errors}/{generated}")
        print(f"Wrong but syntactically valid: {mismatches}/{generated}")
        print(f"Generated: {generated}/{target}")
        if missing_pred:
            print(f"Missing predictions: {missing_pred}/{generated}")
        if duplicates:
            print(f"Prediction duplicates skipped: {duplicates}")
        print("=" * 80)

    append_eval_log(
        dataset="bigmath",
        pred_file=args.pred_file,
        output_file=args.output_file,
        accuracy=accuracy,
        syntax_errors=syntax_errors,
        semantic_errors=semantic_errors,
        total_samples=generated,
        evaluated_samples=evaluated,
        missing_pred=missing_pred,
        duplicates=duplicates,
        eval_script=Path(__file__).name,
        extra={
            "data_file": args.data_file,
            "split": args.split,
            "num_samples": args.num_samples,
            "generated": generated,
            "target": target,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Big-Math predictions with LaTeX parsing logic.",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        required=True,
        help="JSONL file produced by text_models_generate.py.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to write per-sample evaluation JSONL.",
    )
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
        default=0,
        help="Number of parseable samples to evaluate; 0 means all.",
    )
    parser.add_argument(
        "--append_output",
        action="store_true",
        help="Append to output_file instead of overwriting.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample evaluation status.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
