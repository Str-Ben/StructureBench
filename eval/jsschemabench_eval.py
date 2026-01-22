#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JSONSchemaBench evaluation runner.

Workflow:
1) Run data.text_models_generate.py to produce JSONL predictions.
2) Run this script with --pred_file and dataset args to score outputs.
3) The script loads dataset rows, aligns by sample_id, writes per-sample results,
   and prints a summary consistent with the original generate script.

Example:
python -m eval.jsschemabench_eval \
  --pred_file outputs/jsschema.jsonl \
  --output_file eval/jsschema_eval.jsonl \
  --data_file datasets/JSONSchemaBench/Github_easy/test-00000-of-00001.parquet \
  --split test
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from datasets import load_dataset

from eval.eval_logging import append_eval_log
try:
    import jsonschema

    JSONSCHEMA_AVAILABLE = True
except Exception:  # noqa: BLE001
    JSONSCHEMA_AVAILABLE = False
    jsonschema = None


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
    return str(sample.get("unique_id", f"idx-{idx}"))


def validate_output_json(output_text: str, schema_text: str) -> Tuple[bool, Optional[bool], str]:
    """
    Parse model output as JSON and optionally validate against the schema.
    Returns (json_parsed, schema_ok, reason); schema_ok=None means validation skipped.
    """
    text = output_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    start = text.find("{")
    end = text.rfind("}")
    candidate = text[start : end + 1] if start != -1 and end != -1 and end > start else text
    candidate = candidate.strip()
    try:
        parsed_json = json.loads(candidate)
    except Exception as exc:  # noqa: BLE001
        return False, None, f"json parse failed: {exc}"

    if not JSONSCHEMA_AVAILABLE:
        return True, None, "json parsed (schema validation skipped: jsonschema not installed)"

    try:
        schema_obj = json.loads(schema_text)
    except Exception as exc:  # noqa: BLE001
        return True, None, f"json parsed, but schema parsing failed: {exc}"

    try:
        jsonschema.validate(parsed_json, schema_obj)
        return True, True, "json parsed and validated against schema"
    except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
        return True, False, f"schema validation failed: {exc.message}"
    except jsonschema.SchemaError as exc:  # type: ignore[attr-defined]
        return True, None, f"schema invalid, skip validation: {exc}"


def run_evaluation(args) -> None:
    if args.data_file is None:
        data_file = f"datasets/JSONSchemaBench/{args.subset}/{args.split}-00000-of-00001.parquet"
    else:
        data_file = args.data_file

    ds_dict = load_dataset("parquet", data_files={args.split: data_file})
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
    schema_errors = 0
    schema_skipped = 0
    missing_pred = 0

    with output_path.open(mode, encoding="utf-8") as fout:
        for idx in range(n):
            sample = ds[idx]
            schema_text = sample["json_schema"]
            sample_id = _resolve_sample_id(sample, idx)
            pred_record = pred_map.get(sample_id)
            if pred_record is None:
                missing_pred += 1
                record = {
                    "dataset": "jsschemabench",
                    "sample_id": sample_id,
                    "status": "missing_pred",
                }
                fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                if args.verbose:
                    print(f"[Sample {idx}] id={sample_id} status=missing_pred")
                continue

            output_text = str(_get_field(pred_record, "raw_output", "") or "")
            is_parsed, schema_ok, reason = validate_output_json(output_text, schema_text)
            if not is_parsed:
                syntax_errors += 1
                status = "syntax error"
            elif schema_ok is False:
                schema_errors += 1
                status = "content error"
            else:
                correct += 1
                status = "passed" if schema_ok else "schema check skipped"
                if schema_ok is None:
                    schema_skipped += 1

            record = {
                "dataset": "jsschemabench",
                "sample_id": sample_id,
                "status": status,
                "reason": reason,
            }
            fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

            if args.verbose:
                print(f"[Sample {idx}] id={sample_id} status={status} ({reason})")

    accuracy = correct / n if n else None
    semantic_errors = schema_errors
    if n > 0:
        acc = accuracy or 0.0
        print("=" * 80)
        print(f"Valid outputs (parsed + schema): {correct}/{n} = {acc:.2%}")
        print(f"Syntax errors (invalid JSON): {syntax_errors}")
        print(f"Content errors (schema violations): {schema_errors}")
        if schema_skipped:
            print(f"Schema validation skipped: {schema_skipped} (missing jsonschema or invalid schema)")
        if missing_pred:
            print(f"Missing predictions: {missing_pred}/{n}")
        if duplicates:
            print(f"Prediction duplicates skipped: {duplicates}")
        print("=" * 80)

    append_eval_log(
        dataset="jsschemabench",
        pred_file=args.pred_file,
        output_file=args.output_file,
        accuracy=accuracy,
        syntax_errors=syntax_errors,
        semantic_errors=semantic_errors,
        total_samples=n,
        evaluated_samples=n,
        missing_pred=missing_pred,
        duplicates=duplicates,
        eval_script=Path(__file__).name,
        extra={
            "data_file": data_file,
            "subset": args.subset,
            "split": args.split,
            "num_samples": args.num_samples,
            "schema_skipped": schema_skipped,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate JSONSchemaBench predictions against JSON schemas.",
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
        default=None,
        help="Path to JSONSchemaBench parquet file.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="Github_easy",
        help="Subset name when data_file is not provided.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split: train / val / test.",
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
