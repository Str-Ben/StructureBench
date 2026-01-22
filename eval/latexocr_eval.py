#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LaTeX OCR evaluation runner.

Workflow:
1) Run data.vl_models_generate.py to produce JSONL predictions.
2) Run this script with --pred_file and dataset args to score outputs.
3) The script loads dataset rows, aligns by sample_id, writes per-sample results,
   and prints a summary consistent with the original generate script.

Example:
python -m eval.latexocr_eval \
  --pred_file outputs/latexocr.jsonl \
  --output_file eval/latexocr_eval.jsonl \
  --data_file datasets/LaTeX_OCR/small/validation-00000-of-00001.parquet \
  --split validation
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

ENV_PATTERN = re.compile(r"\\(begin|end)\s*\{([^}]*)\}")

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
    return str(sample.get("id", f"idx-{idx}"))


def _extract_between(text: str, start: str, end: str) -> Optional[str]:
    start_idx = text.find(start)
    end_idx = text.rfind(end)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx + len(start):
        return text[start_idx + len(start) : end_idx]
    return None


def _extract_math_block(text: Any) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    for start, end in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        block = _extract_between(s, start, end)
        if block is not None:
            return block
    return s


def _cleanup_latex(text: str) -> str:
    s = text.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\displaystyle", "")
    s = re.sub(r"\\(?:,|;|!|quad|qquad)", "", s)
    return s


def _normalize_latex_text(text: str) -> str:
    return re.sub(r"\s+", "", _cleanup_latex(text))


def _normalize_latex(text: Any) -> str:
    return _normalize_latex_text(_extract_math_block(text))


def _check_env_balance(text: str) -> Tuple[bool, str]:
    stack = []
    for match in ENV_PATTERN.finditer(text):
        kind = match.group(1)
        name = match.group(2).strip()
        if kind == "begin":
            stack.append(name)
            continue
        if not stack:
            return False, f"unexpected \\end{{{name}}}"
        if stack[-1] != name:
            return False, f"mismatched \\end{{{name}}}"
        stack.pop()
    if stack:
        return False, f"unclosed \\begin{{{stack[-1]}}}"
    return True, ""


def _consume_left_right(text: str, idx: int) -> Tuple[int, bool]:
    for cmd in ("\\left", "\\right"):
        if not text.startswith(cmd, idx):
            continue
        next_idx = idx + len(cmd)
        if next_idx < len(text) and text[next_idx].isalpha():
            return idx, False
        while next_idx < len(text) and text[next_idx].isspace():
            next_idx += 1
        if next_idx >= len(text):
            return next_idx, True
        if text[next_idx] == "\\":
            next_idx += 1
            if next_idx < len(text) and text[next_idx].isalpha():
                while next_idx < len(text) and text[next_idx].isalpha():
                    next_idx += 1
            else:
                next_idx += 1
        else:
            next_idx += 1
        return next_idx, True
    return idx, False


def _check_bracket_balance(text: str) -> Tuple[bool, str]:
    pairs = {")": "(", "]": "[", "}": "{"}
    openers = set(pairs.values())
    stack = []
    idx = 0
    while idx < len(text):
        idx_next, consumed = _consume_left_right(text, idx)
        if consumed:
            idx = idx_next
            continue
        ch = text[idx]
        if ch == "\\":
            idx += 1
            if idx < len(text):
                idx += 1
            continue
        if ch in openers:
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False, "unbalanced brackets"
            stack.pop()
        idx += 1
    if stack:
        return False, "unbalanced brackets"
    return True, ""


def _basic_format_check(text: str) -> Tuple[bool, str]:
    if not text or not text.strip():
        return False, "empty output"
    ok, err = _check_env_balance(text)
    if not ok:
        return False, err
    ok, err = _check_bracket_balance(text)
    if not ok:
        return False, err
    return True, ""


def _try_parse_latex_expr(latex_str: str) -> Tuple[Optional["sp.Expr"], Optional[str]]:
    if parse_latex is None or sp is None:
        reason = _LATEX_IMPORT_ERROR or "sympy parse_latex unavailable"
        return None, f"latex parser unavailable: {reason}"
    if not latex_str or not latex_str.strip():
        return None, "empty latex"
    try:
        return parse_latex(latex_str), None
    except Exception as exc:
        return None, f"parse error: {exc}"


def _simplify_expr(expr: "sp.Expr") -> "sp.Expr":
    return sp.simplify(expr)


def _is_zero_expr(expr: "sp.Expr") -> Optional[bool]:
    if expr == 0:
        return True
    if expr.is_zero is True:
        return True
    if expr.is_zero is False:
        return False
    return None


def _equiv_sympy(pred_text: str, gold_text: str) -> Tuple[Optional[bool], bool, str]:
    pred_expr, pred_err = _try_parse_latex_expr(pred_text)
    if pred_expr is None:
        return None, False, pred_err or "pred parse error"
    gold_expr, gold_err = _try_parse_latex_expr(gold_text)
    if gold_expr is None:
        return None, False, gold_err or "gold parse error"
    diff = None
    try:
        diff = _simplify_expr(pred_expr - gold_expr)
    except Exception:
        diff = None
    if diff is not None:
        zero = _is_zero_expr(diff)
        if zero is True:
            return True, True, ""
        if zero is False:
            return False, True, ""
    try:
        eq = pred_expr.equals(gold_expr)
    except Exception as exc:
        return None, False, f"equiv error: {exc}"
    if eq is None:
        return None, False, "equiv undecidable"
    return bool(eq), True, ""


def run_evaluation(args) -> None:
    ds_dict = load_dataset("parquet", data_files={args.split: args.data_file})
    if args.split not in ds_dict:
        raise ValueError(f"split {args.split!r} not found in data file; available: {list(ds_dict.keys())}")
    ds = ds_dict[args.split]

    pred_map, duplicates = _index_predictions(_read_jsonl(args.pred_file))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append_output else "w"

    n = len(ds) if not args.num_samples or args.num_samples <= 0 else min(args.num_samples, len(ds))
    correct = 0
    evaluated = 0
    syntax_errors = 0
    semantic_errors = 0
    missing_pred = 0

    with output_path.open(mode, encoding="utf-8") as fout:
        for idx in range(n):
            sample = ds[idx]
            gold_latex = sample.get("text")
            sample_id = _resolve_sample_id(sample, idx)
            pred_record = pred_map.get(sample_id)
            if pred_record is None:
                missing_pred += 1
                record = {
                    "dataset": "latexocr",
                    "sample_id": sample_id,
                    "status": "missing_pred",
                }
                fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                if args.verbose:
                    print(f"[Sample {idx}] id={sample_id} status=missing_pred")
                continue

            output_text = str(_get_field(pred_record, "raw_output", "") or "")  
            if gold_latex is not None:
                evaluated += 1
                pred_block = _extract_math_block(output_text)
                gold_block = _extract_math_block(gold_latex)
                pred_norm = _normalize_latex_text(pred_block)
                gold_norm = _normalize_latex_text(gold_block)
                string_match = pred_norm == gold_norm
                format_ok, format_error = _basic_format_check(pred_block)
                parser_ok = False
                parser_error = ""
                sympy_equivalent = None
                if not format_ok:
                    syntax_errors += 1
                    status = "syntax error"
                    is_correct = False
                else:
                    pred_parse = _cleanup_latex(pred_block)
                    gold_parse = _cleanup_latex(gold_block)
                    sympy_equivalent, parser_ok, parser_error = _equiv_sympy(
                        pred_parse, gold_parse
                    )
                    if parser_ok:
                        is_correct = bool(sympy_equivalent)
                    else:
                        is_correct = string_match
                    correct += int(is_correct)
                    if not is_correct:
                        semantic_errors += 1
                    status = "correct" if is_correct else "semantic error"
                if not format_ok and not parser_error:
                    parser_error = format_error
            else:
                status = "no_gold"
                pred_norm = ""
                gold_norm = ""
                string_match = None
                format_ok = None
                format_error = ""
                parser_ok = None
                parser_error = ""
                sympy_equivalent = None

            record = {
                "dataset": "latexocr",
                "sample_id": sample_id,
                "status": status,
                "pred_norm": pred_norm,
                "gold_norm": gold_norm,
                "string_match": string_match,
                "format_ok": format_ok,
                "format_error": format_error,
                "parser_ok": parser_ok,
                "parser_error": parser_error,
                "sympy_equivalent": sympy_equivalent,
            }
            fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

            if args.verbose:
                print(f"[Sample {idx}] id={sample_id} status={status}")

    accuracy = correct / evaluated if evaluated else None
    if n > 0:
        acc = accuracy or 0.0
        print("=" * 80)
        print(f"Accuracy: {correct}/{evaluated} = {acc:.2%}")
        print(f"Syntax errors: {syntax_errors}")
        print(f"Semantic errors: {semantic_errors}")
        if missing_pred:
            print(f"Missing predictions: {missing_pred}/{n}")
        if duplicates:
            print(f"Prediction duplicates skipped: {duplicates}")
        print("=" * 80)

    append_eval_log(
        dataset="latexocr",
        pred_file=args.pred_file,
        output_file=args.output_file,
        accuracy=accuracy,
        syntax_errors=syntax_errors,
        semantic_errors=semantic_errors,
        total_samples=n,
        evaluated_samples=evaluated,
        missing_pred=missing_pred,
        duplicates=duplicates,
        eval_script=Path(__file__).name,
        extra={
            "data_file": args.data_file,
            "split": args.split,
            "num_samples": args.num_samples,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LaTeX OCR predictions against reference LaTeX strings.",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        required=True,
        help="JSONL file produced by vl_models_generate.py.",
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
        default="datasets/LaTeX_OCR/small/validation-00000-of-00001.parquet",
        help="LaTeX_OCR data file (parquet).",
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
        default=0,
        help="Number of samples to evaluate; 0 means all.",
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
