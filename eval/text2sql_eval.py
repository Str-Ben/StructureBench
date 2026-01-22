#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text2SQL evaluation runner.

Workflow:
1) Run data.text_models_generate.py to produce JSONL predictions.
2) Run this script with --pred_file and dataset args to score outputs.
3) The script loads dataset rows, aligns by sample_id, writes per-sample results,
   and prints a summary consistent with the original generate script.

Example:
python -m eval.text2sql_eval \
  --pred_file outputs/text2sql.jsonl \
  --output_file eval/text2sql_eval.jsonl \
  --data_file datasets/synthetic_text_to_sql/synthetic_text_to_sql_test.snappy.parquet \
  --split test \
  --sql_parser auto \
  --sql_dialect sqlite
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

from datasets import load_dataset

from eval.eval_logging import append_eval_log

FINAL_TAG_TOKEN_RE = re.compile(r"</?final>", re.IGNORECASE)
SQL_CODE_FENCE_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
SQL_KEYWORD_RE = re.compile(
    r"\b(select|with|insert|update|delete|create|drop|alter)\b",
    re.IGNORECASE,
)
SQL_START_RE = re.compile(
    r"^\s*(select|with|insert|update|delete|create|drop|alter)\b",
    re.IGNORECASE,
)

try:
    import sqlglot
    from sqlglot import parse_one

    SQLGLOT_AVAILABLE = True
except Exception:
    SQLGLOT_AVAILABLE = False
    parse_one = None
    sqlglot = None


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


def resolve_sql_parser(sql_parser: str) -> str:
    if sql_parser == "auto":
        return "sqlglot" if SQLGLOT_AVAILABLE else "none"
    if sql_parser == "sqlglot" and not SQLGLOT_AVAILABLE:
        print("Warning: sqlglot not installed, fallback to string normalization.")
        return "none"
    return sql_parser


def extract_sql_from_output(text: str) -> str:
    if text is None:
        return ""
    s = text.strip()
    if not s:
        return ""

    depth = 0
    start = None
    last_span = None
    for match in FINAL_TAG_TOKEN_RE.finditer(s):
        token = match.group(0).lower()
        if token == "<final>":
            if depth == 0:
                start = match.end()
            depth += 1
        elif token == "</final>":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                last_span = (start, match.start())
                start = None

    if last_span:
        return s[last_span[0] : last_span[1]].strip()

    last_fence = None
    for match in SQL_CODE_FENCE_RE.finditer(s):
        last_fence = match
    if last_fence:
        return last_fence.group(1).strip()

    last_keyword = None
    for match in SQL_KEYWORD_RE.finditer(s):
        last_keyword = match
    if last_keyword:
        return s[last_keyword.start() :].strip()

    return ""


def normalize_sql_basic(sql: str) -> str:
    if sql is None:
        return ""
    cleaned = sql.strip().rstrip(";")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


def normalize_sql(sql: str, sql_parser: str = "auto", sql_dialect: str = None) -> str:
    cleaned = normalize_sql_basic(sql)
    if not cleaned:
        return ""

    if sql_parser == "auto":
        sql_parser = "sqlglot" if SQLGLOT_AVAILABLE else "none"
    if sql_parser == "sqlglot" and parse_one is None:
        sql_parser = "none"

    if sql_parser == "none":
        return cleaned

    if sql_parser == "sqlglot" and parse_one is not None:
        try:
            expr = parse_one(cleaned, read=sql_dialect) if sql_dialect else parse_one(cleaned)
            expr = expr.normalize()
            canonical = expr.sql(dialect=sql_dialect) if sql_dialect else expr.sql()
            return normalize_sql_basic(canonical)
        except Exception:
            return cleaned

    return cleaned


def normalize_for_match(
    pred_text: str,
    gold_text: str,
    sql_parser: str = "auto",
    sql_dialect: str = None,
):
    pred_sql = extract_sql_from_output(pred_text)
    if gold_text is None:
        return pred_sql, "", ""
    pred_norm = normalize_sql(pred_sql, sql_parser=sql_parser, sql_dialect=sql_dialect)
    gold_norm = normalize_sql(str(gold_text), sql_parser=sql_parser, sql_dialect=sql_dialect)
    return pred_sql, pred_norm, gold_norm


def check_sql_syntax(sql: str, sql_parser: str, sql_dialect: str = None):
    if not sql:
        return False
    if sql_parser == "sqlglot" and parse_one is not None:
        try:
            if sql_dialect:
                parse_one(sql, read=sql_dialect)
            else:
                parse_one(sql)
            return True
        except Exception:
            return False
    return SQL_START_RE.match(sql) is not None


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
    resolved_parser = resolve_sql_parser(args.sql_parser)
    correct = 0
    total_with_gold = 0
    syntax_errors = 0
    semantic_errors = 0
    missing_pred = 0

    with output_path.open(mode, encoding="utf-8") as fout:
        for idx in range(n):
            sample = ds[idx]
            gold_sql = sample.get("sql")
            sample_id = _resolve_sample_id(sample, idx)
            pred_record = pred_map.get(sample_id)

            if pred_record is None:
                missing_pred += 1
                record = {
                    "dataset": "text2sql",
                    "sample_id": sample_id,
                    "status": "missing_pred",
                }
                fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                if args.verbose:
                    print(f"[Sample {idx}] id={sample_id} status=missing_pred")
                continue

            output_text = str(_get_field(pred_record, "raw_output", "") or "")
            pred_sql, pred_norm, gold_norm = normalize_for_match(
                output_text,
                gold_sql,
                sql_parser=resolved_parser,
                sql_dialect=args.sql_dialect,
            )
            syntax_ok = check_sql_syntax(pred_sql, resolved_parser, sql_dialect=args.sql_dialect)
            if not syntax_ok:
                syntax_errors += 1
            is_correct = False
            if gold_sql is not None:
                total_with_gold += 1
                is_correct = pred_norm == gold_norm
                if not syntax_ok:
                    is_correct = False
                if syntax_ok and is_correct:
                    correct += 1
                elif syntax_ok:
                    semantic_errors += 1

            if not syntax_ok:
                result_label = "syntax error"
            elif is_correct:
                result_label = "correct"
            else:
                result_label = "semantic error"

            record = {
                "dataset": "text2sql",
                "sample_id": sample_id,
                "status": result_label,
                "pred_sql": pred_sql,
                "pred_norm": pred_norm,
                "gold_norm": gold_norm,
            }
            fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

            if args.verbose:
                print(f"[Sample {idx}] id={sample_id} status={result_label}")

    accuracy = correct / total_with_gold if total_with_gold else None
    if total_with_gold:
        acc = accuracy
        print("=" * 80)
        print(f"Accuracy: {correct}/{total_with_gold} = {acc:.2%}")
        print(f"Syntax errors: {syntax_errors}")
        print(f"Semantic errors (syntax ok but not equivalent): {semantic_errors}")
        if missing_pred:
            print(f"Missing predictions: {missing_pred}/{n}")
        if duplicates:
            print(f"Prediction duplicates skipped: {duplicates}")
        print("=" * 80)

    append_eval_log(
        dataset="text2sql",
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
            "sql_parser": resolved_parser,
            "sql_dialect": args.sql_dialect,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Text2SQL predictions with SQL normalization.",
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
        default="datasets/synthetic_text_to_sql/synthetic_text_to_sql_test.snappy.parquet",
        help="synthetic_text_to_sql data file (parquet).",
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
        default=0,
        help="Number of samples to evaluate; 0 means all.",
    )
    parser.add_argument(
        "--sql_parser",
        type=str,
        default="auto",
        choices=["auto", "none", "sqlglot"],
        help="SQL normalization: auto=prefer sqlglot if available; none=string normalization only.",
    )
    parser.add_argument(
        "--sql_dialect",
        type=str,
        default=None,
        help="Optional sqlglot dialect, e.g., 'sqlite'/'postgres'.",
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
