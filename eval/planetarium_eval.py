#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Planetarium evaluation runner.

Workflow:
1) Run data.text_models_generate.py to produce JSONL predictions.
2) Run this script with --pred_file and dataset args to score outputs.
3) The script loads dataset rows, aligns by sample_id, writes per-sample results,
   and prints a summary consistent with the original generate script.

Example:
python -m eval.planetarium_eval \
  --pred_file outputs/planetarium.jsonl \
  --output_file eval/planetarium_eval.jsonl \
  --data_file datasets/planetarium/data/test-00000-of-00001.parquet \
  --split test
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

from datasets import load_dataset

from eval.eval_logging import append_eval_log
try:
    from tarski.io import PDDLReader
    from tarski.syntax import land
except ImportError:
    PDDLReader = None
    land = None


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


def extract_last_balanced_parentheses_span(text: str):
    if not text:
        return None
    last_span = None
    stack_depth = 0
    current_start = None
    for idx, ch in enumerate(text):
        if ch == "(":
            if stack_depth == 0:
                current_start = idx
            stack_depth += 1
        elif ch == ")":
            if stack_depth == 0:
                continue
            stack_depth -= 1
            if stack_depth == 0 and current_start is not None:
                last_span = (current_start, idx)
                current_start = None
    if last_span is None:
        return None
    start, end = last_span
    return text[start : end + 1]


def extract_balanced_parentheses_span_from(text: str, start_idx: int):
    if start_idx < 0 or start_idx >= len(text):
        return None
    depth = 0
    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0:
                return text[start_idx : idx + 1]
    return None


def extract_top_level_parentheses_blocks(text: str):
    blocks = []
    depth = 0
    current_start = None
    for idx, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                current_start = idx
            depth += 1
        elif ch == ")":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and current_start is not None:
                blocks.append(text[current_start : idx + 1])
                current_start = None
    return blocks


def _strip_pddl_markers(text: str) -> str:
    if not text:
        return ""
    lines = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        if stripped.lower().startswith("pddl:"):
            remainder = stripped[5:].strip()
            if remainder:
                lines.append(remainder)
            continue
        lines.append(line)
    return "\n".join(lines)


def _normalize_requirements_block(text: str) -> str:
    def repl(match: re.Match) -> str:
        body = match.group(1)
        tokens = re.split(r"\s+", body.strip())
        fixed = []
        for tok in tokens:
            if not tok:
                continue
            fixed.append(tok if tok.startswith(":") else f":{tok}")
        return f"(:requirements {' '.join(fixed)})"

    return re.sub(r"\(:requirements\s+([^)]+)\)", repl, text, flags=re.IGNORECASE)


def _normalize_pddl_text(text: str) -> str:
    cleaned = _strip_pddl_markers(text)
    cleaned = cleaned.replace("...", "").replace("…", "")
    cleaned = _normalize_requirements_block(cleaned)
    return cleaned


def _find_block(blocks, prefix: str) -> str:
    for block in blocks:
        if block.lstrip().lower().startswith(prefix):
            return block.strip()
    return ""


def _wrap_pddl_problem(blocks, domain_name: str) -> str:
    domain_block = _find_block(blocks, "(:domain")
    if not domain_block and domain_name:
        domain_block = f"(:domain {domain_name})"

    requirements_block = _find_block(blocks, "(:requirements")
    if not requirements_block:
        default_requirements = {
            "blocksworld": ":strips",
            "gripper": ":strips",
            "floor-tile": ":typing",
        }
        req = default_requirements.get((domain_name or "").strip().lower())
        if req:
            requirements_block = f"(:requirements {req})"

    objects_block = _find_block(blocks, "(:objects")
    init_block = _find_block(blocks, "(:init")
    goal_block = _find_block(blocks, "(:goal")

    parts = [domain_block, requirements_block, objects_block, init_block, goal_block]
    body = "\n  ".join([p for p in parts if p])
    if not body:
        return ""
    return f"(define (problem predicted_problem)\n  {body}\n)"


def extract_pddl_block(text: str, domain_name: str = "") -> str:
    if text is None:
        return ""
    cleaned = _normalize_pddl_text(str(text))
    define_match = re.search(r"\(define\b", cleaned, flags=re.IGNORECASE)
    if define_match:
        extracted = extract_balanced_parentheses_span_from(cleaned, define_match.start())
        if extracted:
            return extracted.strip()

    blocks = extract_top_level_parentheses_blocks(cleaned)
    wrapped = _wrap_pddl_problem(blocks, domain_name)
    if wrapped:
        return wrapped.strip()

    extracted = extract_last_balanced_parentheses_span(cleaned)
    return extracted.strip() if extracted is not None else cleaned.strip()


class PDDLParseError(ValueError):
    """PDDL parse error."""


DOMAIN_PDDL_TEXT = {
    "blocksworld": """(define (domain blocksworld)
  (:requirements :strips)
  (:predicates
    (on ?x ?y)
    (on-table ?x)
    (clear ?x)
    (holding ?x)
    (arm-empty)
  )
)""",
    "gripper": """(define (domain gripper)
  (:requirements :strips)
  (:predicates
    (at ?b ?r)
    (at-robby ?r)
    (carry ?g ?b)
    (free ?g)
    (ball ?b)
    (gripper ?g)
    (room ?r)
  )
)""",
    "floor-tile": """(define (domain floor-tile)
  (:requirements :typing)
  (:types robot tile color)
  (:predicates
    (robot-at ?r - robot ?t - tile)
    (robot-has ?r - robot ?c - color)
    (available-color ?r - robot ?c - color)
    (painted ?t - tile ?c - color)
    (right ?t1 - tile ?t2 - tile)
    (up ?t1 - tile ?t2 - tile)
  )
)""",
}


def require_tarski():
    if PDDLReader is None or land is None:
        raise RuntimeError("tarski is not installed. Please install with: pip install tarski")


def parse_problem_with_tarski(domain_name: str, pddl_text: str):
    require_tarski()
    if not pddl_text or not str(pddl_text).strip():
        raise PDDLParseError("empty input")
    domain_key = (domain_name or "").strip().lower()
    if domain_key not in DOMAIN_PDDL_TEXT:
        raise PDDLParseError(f"unknown domain: {domain_name!r}")
    domain_text = DOMAIN_PDDL_TEXT[domain_key]
    reader = PDDLReader(raise_on_error=True)
    try:
        reader.parse_domain_string(domain_text)
        return reader.parse_instance_string(pddl_text)
    except Exception as exc:
        raise PDDLParseError(str(exc)) from exc


def canonicalize_pddl_problem(pddl_text: str, domain_name: str):
    problem = parse_problem_with_tarski(domain_name, pddl_text)
    objects = sorted((c.symbol, c.sort.name) for c in problem.language.constants())
    init_atoms = sorted(str(a) for a in problem.init.as_atoms())
    goal = problem.goal
    if goal is None:
        raise PDDLParseError("missing :goal")
    if getattr(goal, "connective", None) == land:
        goal_repr = ("and", sorted(str(g) for g in goal.subformulas))
    else:
        goal_repr = ("expr", str(goal))
    return (objects, init_atoms, goal_repr)


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
    syntax_error_count = 0
    semantic_error_count = 0
    missing_pred = 0

    with output_path.open(mode, encoding="utf-8") as fout:
        for idx in range(n):
            sample = ds[idx]
            gold_pddl = sample.get("problem_pddl")
            domain_name = sample.get("domain", "")
            sample_id = _resolve_sample_id(sample, idx)
            pred_record = pred_map.get(sample_id)

            if pred_record is None:
                missing_pred += 1
                record = {
                    "dataset": "planetarium",
                    "sample_id": sample_id,
                    "status": "missing_pred",
                }
                fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                if args.verbose:
                    print(f"[Sample {idx}] id={sample_id} status=missing_pred")
                continue

            output_text = str(_get_field(pred_record, "raw_output", "") or "")

            if gold_pddl is not None:
                pred_block = extract_pddl_block(output_text, domain_name=domain_name)
                gold_block = extract_pddl_block(gold_pddl, domain_name=domain_name)
                try:
                    pred_canonical = canonicalize_pddl_problem(pred_block, domain_name)
                except PDDLParseError as exc:
                    syntax_error_count += 1
                    record = {
                        "dataset": "planetarium",
                        "sample_id": sample_id,
                        "status": "syntax error",
                        "error": f"pred syntax error: {exc}",
                    }
                    fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                    if args.verbose:
                        print(f"[Sample {idx}] id={sample_id} status=syntax error")
                    continue
                try:
                    gold_canonical = canonicalize_pddl_problem(gold_block, domain_name)
                except PDDLParseError as exc:
                    syntax_error_count += 1
                    record = {
                        "dataset": "planetarium",
                        "sample_id": sample_id,
                        "status": "syntax error",
                        "error": f"gold syntax error: {exc}",
                    }
                    fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                    if args.verbose:
                        print(f"[Sample {idx}] id={sample_id} status=syntax error")
                    continue
                is_correct = pred_canonical == gold_canonical
                correct += int(is_correct)
                if not is_correct:
                    semantic_error_count += 1
                status = "correct" if is_correct else "semantic error"
                record = {
                    "dataset": "planetarium",
                    "sample_id": sample_id,
                    "status": status,
                }
            else:
                record = {
                    "dataset": "planetarium",
                    "sample_id": sample_id,
                    "status": "no_gold",
                }

            fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            if args.verbose:
                print(f"[Sample {idx}] id={sample_id} status={record['status']}")

    accuracy = correct / n if n else None
    if n > 0:
        acc = accuracy or 0.0
        print("=" * 80)
        print(f"Accuracy: {correct}/{n} = {acc:.2%}")
        print(f"Syntax errors: {syntax_error_count}")
        print(f"Semantic errors: {semantic_error_count}")
        if missing_pred:
            print(f"Missing predictions: {missing_pred}/{n}")
        if duplicates:
            print(f"Prediction duplicates skipped: {duplicates}")
        print("=" * 80)

    append_eval_log(
        dataset="planetarium",
        pred_file=args.pred_file,
        output_file=args.output_file,
        accuracy=accuracy,
        syntax_errors=syntax_error_count,
        semantic_errors=semantic_error_count,
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
        description="Evaluate Planetarium PDDL outputs with Tarski parsing.",
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
        default="datasets/planetarium/data/test-00000-of-00001.parquet",
        help="Planetarium data file (parquet).",
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
