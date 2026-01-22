#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
APE21 evaluation runner.

Workflow:
1) Run data.text_models_generate.py to produce JSONL predictions.
2) Run this script with --pred_file and dataset args to score outputs.
3) The script loads dataset rows, aligns by sample_id, writes per-sample results,
   and prints a summary consistent with the original generate script.

Example:
python -m eval.ape21_eval \
  --pred_file outputs/ape21.jsonl \
  --output_file eval/ape21_eval.jsonl \
  --data_file datasets/Calc-ape210k/data/train-00000-of-00001-b9f022a8492442e4.parquet \
  --split train
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

from datasets import load_dataset
from sympy import sympify

from eval.eval_logging import append_eval_log

FINAL_PATTERN = re.compile(r"<final\b[^>]*>(.*?)</final>", flags=re.IGNORECASE | re.DOTALL)


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


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", "", str(text))


def extract_final(text: str) -> str:
    if text is None:
        return ""
    matches = FINAL_PATTERN.findall(str(text))
    return matches[-1] if matches else ""


def normalized_final(text: str) -> str:
    return normalize_answer(extract_final(text))


def parse_equation_value(text: str):
    """
    Parse an equation in the form x=... and return (value, error_message).
    """

    def _clean(raw: str) -> str:
        cleaned = re.sub(r"<[^>]+>", " ", raw or "")
        return (
            cleaned.replace("\uFF1D", "=")
            .replace("\uFF58", "x")
            .replace("\uFF38", "x")
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
    value, _ = parse_equation_value(text)
    return value


def answers_match(pred_text: str, gold_value) -> bool:
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
    total_with_gold = 0
    syntax_errors = 0
    result_errors = 0
    missing_pred = 0

    with output_path.open(mode, encoding="utf-8") as fout:
        for idx in range(n):
            sample = ds[idx]
            gold_answer = sample.get("result_float")
            sample_id = _resolve_sample_id(sample, idx)
            pred_record = pred_map.get(sample_id)

            if pred_record is None:
                missing_pred += 1
                record = {
                    "dataset": "ape21",
                    "sample_id": sample_id,
                    "status": "missing_pred",
                }
                fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                if args.verbose:
                    print(f"[Sample {idx}] id={sample_id} status=missing_pred")
                continue

            output_text = str(_get_field(pred_record, "raw_output", "") or "")

            if gold_answer is None:
                record = {
                    "dataset": "ape21",
                    "sample_id": sample_id,
                    "status": "no_gold",
                }
                fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                if args.verbose:
                    print(f"[Sample {idx}] id={sample_id} status=no_gold")
                continue

            try:
                gold_value = float(gold_answer)
            except Exception:
                gold_value = None
            if gold_value is None:
                record = {
                    "dataset": "ape21",
                    "sample_id": sample_id,
                    "status": "gold_invalid",
                }
                fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                if args.verbose:
                    print(f"[Sample {idx}] id={sample_id} status=gold_invalid")
                continue

            total_with_gold += 1
            pred_value, parse_error = parse_equation_value(output_text)
            if parse_error:
                syntax_errors += 1
                status = "syntax error"
            else:
                is_correct = abs(pred_value - gold_value) < 1e-6
                if is_correct:
                    correct += 1
                    status = "correct"
                else:
                    result_errors += 1
                    status = "wrong"

            record = {
                "dataset": "ape21",
                "sample_id": sample_id,
                "status": status,
                "pred_value": pred_value,
                "gold_value": gold_value,
                "parse_error": parse_error,
            }
            fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

            if args.verbose:
                print(f"[Sample {idx}] id={sample_id} status={status}")

    accuracy = correct / total_with_gold if total_with_gold else None
    semantic_errors = result_errors
    if total_with_gold:
        acc = accuracy
        print("=" * 80)
        print(f"Accuracy: {correct}/{total_with_gold} = {acc:.2%}")
        print(f"Syntax errors (invalid 'x=...'): {syntax_errors}")
        print(f"Result errors (parsed but mismatched): {result_errors}")
        if missing_pred:
            print(f"Missing predictions: {missing_pred}/{n}")
        if duplicates:
            print(f"Prediction duplicates skipped: {duplicates}")
        print("=" * 80)

    append_eval_log(
        dataset="ape21",
        pred_file=args.pred_file,
        output_file=args.output_file,
        accuracy=accuracy,
        syntax_errors=syntax_errors,
        semantic_errors=semantic_errors,
        total_samples=n,
        evaluated_samples=total_with_gold,
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
        description="Evaluate Calc-ape210k predictions with numeric parsing.",
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
