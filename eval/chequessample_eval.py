#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ChequeSample evaluation runner.

Workflow:
1) Run data.vl_models_generate.py to produce JSONL predictions.
2) Run this script with --pred_file and dataset args to score outputs.
3) The script loads dataset rows, aligns by sample_id, writes per-sample results,
   and prints a summary consistent with the original generate script.

Example:
python -m eval.chequessample_eval \
  --pred_file outputs/cheques.jsonl \
  --output_file eval/cheques_eval.jsonl \
  --data_file datasets/cheques_sample_data/data/validation-00000-of-00001-4b3af28127dd79d2.parquet \
  --split validation
"""

import argparse
import ast
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


CHEQUE_JSON_SCHEMA = {
    "type": "object",
    "required": ["gt_parse"],
    "properties": {
        "gt_parse": {
            "type": "object",
            "required": ["cheque_details"],
            "properties": {
                "cheque_details": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "minProperties": 1,
                        "maxProperties": 1,
                        "additionalProperties": {"type": "string"},
                    },
                }
            },
            "additionalProperties": False,
        }
    },
    "additionalProperties": False,
}


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


def _basic_schema_check(obj: Any) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "root is not an object"
    gt_parse = obj.get("gt_parse")
    if not isinstance(gt_parse, dict):
        return False, "gt_parse missing or not an object"
    details = gt_parse.get("cheque_details")
    if not isinstance(details, list) or not details:
        return False, "cheque_details missing or not a non-empty list"
    for i, item in enumerate(details):
        if not isinstance(item, dict) or not item:
            return False, f"cheque_details[{i}] is not an object with one field"
        if len(item) != 1:
            return False, f"cheque_details[{i}] should contain exactly one field"
        value = next(iter(item.values()))
        if not isinstance(value, str):
            return False, f"cheque_details[{i}] value is not a string"
    return True, "basic structure passed"


def _extract_json_candidate(output_text: str) -> str:
    text = output_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    start = text.find("{")
    end = text.rfind("}")
    candidate = text[start : end + 1] if start != -1 and end != -1 and end > start else text
    return candidate.strip()


_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\uFF02": '"',
        "\uFF07": "'",
        ".": "/",
    }
)


def _normalize_text(text: str) -> str:
    normalized = text.translate(_QUOTE_TRANSLATION)
    return " ".join(normalized.split())


def _parse_json_text(text: str) -> Tuple[bool, Optional[Any], str]:
    candidate = _extract_json_candidate(text)
    try:
        return True, json.loads(candidate), "json parsed"
    except Exception as exc:  # noqa: BLE001
        json_error = exc

    try:
        parsed = ast.literal_eval(candidate)
    except Exception as exc:  # noqa: BLE001
        return False, None, f"json parse failed: {json_error}; literal_eval failed: {exc}"

    if isinstance(parsed, str):
        try:
            return True, json.loads(parsed), "json parsed (from literal string)"
        except Exception as exc:  # noqa: BLE001
            return False, None, f"literal_eval produced string but json parse failed: {exc}"
    return True, parsed, "literal_eval parsed"


def validate_output_json(output_text: str) -> Tuple[bool, bool, Optional[dict], str]:
    parsed, parsed_json, reason = _parse_json_text(output_text)
    if not parsed:
        return False, False, None, reason

    if JSONSCHEMA_AVAILABLE:
        try:
            jsonschema.validate(parsed_json, CHEQUE_JSON_SCHEMA)
            return True, True, parsed_json, "json parsed and validated against schema"
        except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            return True, False, parsed_json, f"schema validation failed: {exc.message}"
        except jsonschema.SchemaError as exc:  # type: ignore[attr-defined]
            ok, reason = _basic_schema_check(parsed_json)
            return True, ok, parsed_json, f"schema invalid ({exc}), basic check: {reason}"

    ok, reason = _basic_schema_check(parsed_json)
    return True, ok, parsed_json, f"json parsed (basic schema check): {reason}"


def _flatten_cheque_details(obj: dict, normalize: bool = False) -> Optional[Dict[str, str]]:
    try:
        details = obj["gt_parse"]["cheque_details"]
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(details, list):
        return None
    flattened: Dict[str, str] = {}
    for item in details:
        if not isinstance(item, dict) or len(item) != 1:
            return None
        key, value = next(iter(item.items()))
        if not isinstance(value, str):
            return None
        key_text = str(key).strip()
        value_text = value.strip()
        if normalize:
            key_text = _normalize_text(key_text)
            value_text = _normalize_text(value_text)
            if key_text == "cheque_date":
                value_text = value_text.replace(".", "/")
        flattened[key_text] = value_text
    return flattened


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
    missing_pred = 0

    with output_path.open(mode, encoding="utf-8") as fout:
        for idx in range(n):
            sample = ds[idx]
            gold_json = sample.get("ground_truth")
            sample_id = _resolve_sample_id(sample, idx)
            pred_record = pred_map.get(sample_id)

            if pred_record is None:
                missing_pred += 1
                record = {
                    "dataset": "chequessample",
                    "sample_id": sample_id,
                    "status": "missing_pred",
                }
                fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                if args.verbose:
                    print(f"[Sample {idx}] id={sample_id} status=missing_pred")
                continue

            output_text = str(_get_field(pred_record, "raw_output", "") or "")
            parsed, schema_ok, parsed_obj, reason = validate_output_json(output_text)
            if not parsed or not schema_ok or parsed_obj is None:
                syntax_errors += 1
                status = "syntax error (parse/schema)"
            else:
                if isinstance(gold_json, dict):
                    gold_parsed, gold_obj, gold_reason = True, gold_json, "gold already dict"
                else:
                    gold_parsed, gold_obj, gold_reason = _parse_json_text(str(gold_json))
                if not gold_parsed or not isinstance(gold_obj, dict):
                    semantic_errors += 1
                    status = f"semantic error (gold parse failed: {gold_reason})"
                else:
                    gold_fields = _flatten_cheque_details(gold_obj, normalize=True)
                    pred_fields = _flatten_cheque_details(parsed_obj, normalize=True)
                    semantic_match = (
                        gold_fields is not None
                        and pred_fields is not None
                        and gold_fields == pred_fields
                    )
                    if semantic_match:
                        correct += 1
                        status = "correct"
                    else:
                        semantic_errors += 1
                        status = "semantic error (content mismatch)"

            record = {
                "dataset": "chequessample",
                "sample_id": sample_id,
                "status": status,
                "reason": reason,
            }
            fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

            if args.verbose:
                print(f"[Sample {idx}] id={sample_id} status={status} ({reason})")

    accuracy = correct / n if n else None
    if n > 0:
        acc = accuracy or 0.0
        print("=" * 80)
        print(f"Accuracy: {correct}/{n} = {acc:.2%}")
        print(f"Syntax errors (parse/schema): {syntax_errors}")
        print(f"Semantic errors (schema ok but mismatched content): {semantic_errors}")
        if missing_pred:
            print(f"Missing predictions: {missing_pred}/{n}")
        if duplicates:
            print(f"Prediction duplicates skipped: {duplicates}")
        print("=" * 80)

    append_eval_log(
        dataset="chequessample",
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
            "data_file": args.data_file,
            "split": args.split,
            "num_samples": args.num_samples,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate cheque OCR JSON outputs against structured ground truth.",
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
        default="datasets/cheques_sample_data/data/validation-00000-of-00001-4b3af28127dd79d2.parquet",
        help="cheques_sample_data data file (parquet).",
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
