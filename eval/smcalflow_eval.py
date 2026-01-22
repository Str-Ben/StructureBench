#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SMCalFlow evaluation runner.

Workflow:
1) Run data.text_models_generate.py to produce JSONL predictions.
2) Run this script with --pred_file and dataset args to score outputs.
3) The script loads dataset rows, aligns by sample_id, writes per-sample results,
   and prints a summary consistent with the original generate script.

Example:
python -m eval.smcalflow_eval \
  --pred_file outputs/smcalflow.jsonl \
  --output_file eval/smcalflow_eval.jsonl \
  --data_file datasets/UDR_SMCalFlow/data/validation-00000-of-00001-4c17fe465006b08d.parquet \
  --split validation
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

from datasets import load_dataset

from eval.eval_logging import append_eval_log
from dataflow.core.lispress import parse_lispress, render_compact, lispress_to_program
from dataflow.core.program import program_to_lispress


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
    return str(sample.get("idx", f"idx-{idx}"))


def extract_first_balanced_parentheses(text: str):
    """Return the first balanced parenthesis span (inclusive)."""
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
    Normalize Lispress with Dataflow:
    1) Extract the first balanced parenthesis span if present.
    2) parse_lispress -> lispress_to_program -> program_to_lispress.
    3) render_compact for canonical comparison.
    """
    extracted = extract_first_balanced_parentheses(str(text)) if text is not None else None
    candidate = extracted if extracted is not None else ("" if text is None else str(text))
    lispress = parse_lispress(candidate)
    program, _ = lispress_to_program(lispress, indexicals={})
    return render_compact(program_to_lispress(program))


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
    syntax_errors = 0
    semantic_errors = 0
    evaluated = 0
    missing_pred = 0

    with output_path.open(mode, encoding="utf-8") as fout:
        for idx in range(n):
            sample = ds[idx]
            gold_lispress = sample.get("lispress")
            sample_id = _resolve_sample_id(sample, idx)
            pred_record = pred_map.get(sample_id)
            if pred_record is None:
                missing_pred += 1
                record = {
                    "dataset": "smcalflow",
                    "sample_id": sample_id,
                    "status": "missing_pred",
                }
                fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                if args.verbose:
                    print(f"[Sample {idx}] id={sample_id} status=missing_pred")
                continue

            output_text = str(_get_field(pred_record, "raw_output", "") or "")
            if gold_lispress is not None:
                evaluated += 1
                try:
                    normalized_gold = normalize_lispress_with_dataflow(gold_lispress)
                except Exception as exc:
                    record = {
                        "dataset": "smcalflow",
                        "sample_id": sample_id,
                        "status": "gold_parse_error",
                        "error": str(exc),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                    if args.verbose:
                        print(f"[Sample {idx}] id={sample_id} status=gold_parse_error")
                    continue

                try:
                    normalized_output = normalize_lispress_with_dataflow(output_text)
                    is_correct = normalized_output == normalized_gold
                    if is_correct:
                        correct += 1
                        status = "correct"
                    else:
                        semantic_errors += 1
                        status = "wrong"
                    record = {
                        "dataset": "smcalflow",
                        "sample_id": sample_id,
                        "status": status,
                        "normalized_output": normalized_output,
                        "normalized_gold": normalized_gold,
                    }
                except Exception as exc:
                    syntax_errors += 1
                    record = {
                        "dataset": "smcalflow",
                        "sample_id": sample_id,
                        "status": "parse_error",
                        "error": str(exc),
                    }
            else:
                record = {
                    "dataset": "smcalflow",
                    "sample_id": sample_id,
                    "status": "no_gold",
                }

            fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

            if args.verbose:
                print(f"[Sample {idx}] id={sample_id} status={record['status']}")

    accuracy = correct / evaluated if evaluated else None
    if n > 0:
        acc = accuracy or 0.0
        print("=" * 80)
        print(f"Accuracy: {correct}/{evaluated} = {acc:.2%}")
        print(f"Syntax errors (unparsable outputs): {syntax_errors}")
        print(f"Semantic errors (parsed but mismatch): {semantic_errors}")
        if missing_pred:
            print(f"Missing predictions: {missing_pred}/{n}")
        if duplicates:
            print(f"Prediction duplicates skipped: {duplicates}")
        print("=" * 80)

    append_eval_log(
        dataset="smcalflow",
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
        description="Evaluate SMCalFlow Lispress predictions with Dataflow normalization.",
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
