#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AMR evaluation runner.

Workflow:
1) Run data.text_models_generate.py to produce JSONL predictions.
2) Run this script with --pred_file and dataset args to score outputs.
3) The script loads dataset rows, aligns by sample_id, writes per-sample results,
   and prints a summary consistent with the original generate script.

Example:
python -m eval.amr_eval \
  --pred_file outputs/amr.jsonl \
  --output_file eval/amr_eval.jsonl \
  --data_file datasets/amr-3-parsed/data/validation-00000-of-00001.parquet \
  --split validation
"""

import argparse
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

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

from eval.eval_logging import append_eval_log


SMATCH_EQ_THRESHOLD = 0.999


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


def extract_user_and_amr(sample):
    conversations = sample.get("conversations", [])
    user_msg = ""
    gold_amr = None
    if isinstance(conversations, list):
        if len(conversations) > 0:
            user_msg = conversations[0].get("content", "")
        if len(conversations) > 1:
            gold_amr = conversations[1].get("content")
    return user_msg.strip(), gold_amr


def _slice_last_top_level_paren_block(text: str) -> str:
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
    return re.sub(r"[ \t\r\n]+", " ", text).strip()


def postprocess_model_output(text: str) -> str:
    return _normalize_whitespace(_slice_last_top_level_paren_block(text))


def postprocess_gold_answer(text: str) -> str:
    return _normalize_whitespace(text)


def _parse_amr(text: str):
    if penman is None:
        return None
    try:
        return penman.decode(text)
    except (PenmanError, ValueError):
        return None
    except Exception:
        return None


def _smatch_f1(pred_graph, gold_graph) -> float | None:
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
    smatch_sum = 0.0
    smatch_count = 0
    smatch_missing = 0
    missing_pred = 0
    parser_ready = penman is not None
    smatch_ready = parser_ready and smatch is not None

    if not parser_ready:
        print("Warning: penman not installed; syntax/semantic validation skipped (fall back to string exact match).")
    elif not smatch_ready:
        print("Warning: smatch not installed; semantic equivalence will fall back to string exact match.")

    with output_path.open(mode, encoding="utf-8") as fout:
        for idx in range(n):
            sample = ds[idx]
            _, gold_answer = extract_user_and_amr(sample)
            sample_id = _resolve_sample_id(sample, idx)
            pred_record = pred_map.get(sample_id)

            if pred_record is None:
                missing_pred += 1
                record = {
                    "dataset": "amr",
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
                    "dataset": "amr",
                    "sample_id": sample_id,
                    "status": "no_gold",
                }
                fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                if args.verbose:
                    print(f"[Sample {idx}] id={sample_id} status=no_gold")
                continue

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
            status = "wrong"
            if parser_ready and pred_graph is None:
                syntax_errors += 1
                status = "syntax error"
            else:
                if smatch_score is not None:
                    is_correct = smatch_score >= SMATCH_EQ_THRESHOLD
                    if pred_graph is not None and smatch_score < SMATCH_EQ_THRESHOLD:
                        semantic_errors += 1
                elif not smatch_ready or not parser_ready or gold_graph is None:
                    is_correct = processed_output == processed_gold
                status = "correct" if is_correct else "wrong"

            correct += int(is_correct)
            record = {
                "dataset": "amr",
                "sample_id": sample_id,
                "status": status,
                "processed_output": processed_output,
                "processed_gold": processed_gold,
                "smatch": smatch_score,
            }
            fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

            if args.verbose:
                smatch_preview = f"{smatch_score:.4f}" if smatch_score is not None else "n/a"
                print(f"[Sample {idx}] id={sample_id} status={status} smatch={smatch_preview}")

    accuracy = correct / evaluated if evaluated else None
    if evaluated > 0:
        acc = accuracy
        print("=" * 80)
        print(f"Checked samples (with gold): {evaluated}")
        print(f"Accuracy: {correct}/{evaluated} = {acc:.2%}")
        if parser_ready:
            print(f"Syntax errors: {syntax_errors}")
            if smatch_ready:
                print(f"Semantic errors: {semantic_errors}")
                if smatch_count > 0:
                    avg_smatch = smatch_sum / smatch_count
                    print(f"Average Smatch (missing as 0): {avg_smatch:.4f}")
                    if smatch_missing > 0:
                        print(f"Smatch missing (treated as 0): {smatch_missing}")
        if missing_pred:
            print(f"Missing predictions: {missing_pred}/{n}")
        if duplicates:
            print(f"Prediction duplicates skipped: {duplicates}")
        print("=" * 80)

    append_eval_log(
        dataset="amr",
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
        description="Evaluate AMR predictions on amr-3-parsed.",
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
