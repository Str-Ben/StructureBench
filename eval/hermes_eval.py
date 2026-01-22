#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hermes function calling evaluation runner.

Workflow:
1) Run data.text_models_generate.py to produce JSONL predictions.
2) Run this script with --pred_file and dataset args to score outputs.
3) The script loads dataset rows, aligns by sample_id, writes per-sample results,
   and prints a summary consistent with the original generate script.

Example:
python -m eval.hermes_eval \
  --pred_file outputs/hermes.jsonl \
  --output_file eval/hermes_eval.jsonl \
  --data_file datasets/hermes-function-calling-v1/json-mode-singleturn.json \
  --split train
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

from datasets import load_dataset

from eval.eval_logging import append_eval_log

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


def extract_fields(sample):
    conversations = sample.get("conversations", [])
    system_msg = conversations[0]["value"] if len(conversations) > 0 else ""
    user_msg = conversations[1]["value"] if len(conversations) > 1 else ""
    assistant_msg = conversations[2]["value"] if len(conversations) > 2 else None
    return system_msg.strip(), user_msg.strip(), assistant_msg


def extract_first_json_field(text: str):
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
    if text is None:
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\n", " ").strip()


def strip_all_whitespace(text: str):
    if text is None:
        return None
    return "".join(text.split())


def run_evaluation(args) -> None:
    data_file = args.data_file
    if data_file is None:
        file_name = args.dataset_config.replace("_", "-")
        if not file_name.endswith(".json"):
            file_name = f"{file_name}.json"
        data_file = f"datasets/hermes-function-calling-v1/{file_name}"

    ds_dict = load_dataset("json", data_files={args.split: data_file})
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
            sample_id = _resolve_sample_id(sample, idx)
            pred_record = pred_map.get(sample_id)
            if pred_record is None:
                missing_pred += 1
                record = {
                    "dataset": "hermes",
                    "sample_id": sample_id,
                    "status": "missing_pred",
                }
                fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                if args.verbose:
                    print(f"[Sample {idx}] id={sample_id} status=missing_pred")
                continue

            output_text = str(_get_field(pred_record, "raw_output", "") or "")
            extracted_json = extract_first_json_field(output_text)
            _, _, gold_answer = extract_fields(sample)

            if gold_answer is not None:
                evaluated += 1
                gold_answer_raw = gold_answer.strip()
                if extracted_json is None:
                    is_correct = False
                    syntax_errors += 1
                else:
                    extracted_json = normalize_json_snippet(extracted_json)
                    gold_answer_comp = strip_all_whitespace(gold_answer_raw)
                    extracted_json_comp = strip_all_whitespace(extracted_json)
                    is_correct = extracted_json_comp == gold_answer_comp
                    if not is_correct:
                        semantic_errors += 1
                correct += int(is_correct)
                status = "correct" if is_correct else ("syntax error" if extracted_json is None else "semantic error")
            else:
                status = "no_gold"

            record = {
                "dataset": "hermes",
                "sample_id": sample_id,
                "status": status,
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
        dataset="hermes",
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
            "data_file": data_file,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "num_samples": args.num_samples,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Hermes function calling JSON outputs.",
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
        help="Local Hermes data file (json/jsonl).",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="json_mode_singleturn",
        help="Config name to infer default data_file.",
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
