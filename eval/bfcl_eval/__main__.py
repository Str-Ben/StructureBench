#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BFCL v4 single-turn evaluation runner for StructureBench (prompt-mode).
- Uses local eval/bfcl_eval package (migrated from Gorilla repo).
- Prints per-sample statuses to stdout; does NOT write per-sample JSONL.
- Writes summary JSON (default eval/bfcl_eval_summary.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


from . import utils as bfcl_utils
from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.model_handler.parser.json_parser import parse_json_function_call
from .eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.constants import model_config as bfcl_model_config
from bfcl_eval.constants.model_config import ModelConfig

DEFAULT_DATA_ROOT = Path(r"D:\StructureBench\datasets\bfcl_eval\data")

# BFCL scope (single-turn, no irrelevance/relevance/multi-turn)
SCOPE_CATEGORIES = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
]


def _configure_bfcl_paths(data_root: Path) -> None:
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"BFCL data root not found: {data_root}")

    bfcl_utils.PROMPT_PATH = data_root
    bfcl_utils.POSSIBLE_ANSWER_PATH = data_root / "possible_answer"
    bfcl_utils.MEMORY_PREREQ_CONVERSATION_PATH = data_root / "memory_prereq_conversation"
    bfcl_utils.MULTI_TURN_FUNC_DOC_PATH = data_root / "multi_turn_func_doc"
    bfcl_utils.FORMAT_SENSITIVITY_IDS_PATH = (
        data_root / f"{bfcl_utils.VERSION_PREFIX}_format_sensitivity.json"
    )


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
        sample_id = record.get("sample_id")
        if sample_id is None:
            sample_id = f"idx-{idx}"
        sample_id = str(sample_id)
        if sample_id in pred_map:
            duplicates += 1
            continue
        pred_map[sample_id] = record
    return pred_map, duplicates


def _read_first_pred_record(pred_file: str) -> Dict[str, Any]:
    path = Path(pred_file)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return {}
    return {}


def _normalize_categories(test_categories: Optional[Sequence[str]]) -> List[str]:
    if not test_categories:
        return list(SCOPE_CATEGORIES)

    if isinstance(test_categories, str):
        raw = [s.strip() for s in test_categories.replace(",", " ").split() if s.strip()]
    else:
        raw = [str(s).strip() for s in test_categories if str(s).strip()]

    if not raw:
        return list(SCOPE_CATEGORIES)

    try:
        expanded = bfcl_utils.parse_test_category_argument(raw)
    except Exception:
        expanded = raw

    seen = set()
    filtered: List[str] = []
    for name in expanded:
        if name in SCOPE_CATEGORIES and name not in seen:
            filtered.append(name)
            seen.add(name)
    return filtered or list(SCOPE_CATEGORIES)


def _map_language_token(token: str):
    token = token.lower()
    if "javascript" in token or token == "js":
        return ReturnFormat.JAVASCRIPT
    if "java" in token:
        return ReturnFormat.JAVA
    if "python" in token or token in {"py", "py3"}:
        return ReturnFormat.PYTHON
    return None


def _resolve_return_format(entry: Mapping[str, Any], test_category: str):
    for key in ("return_format", "language", "lang"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            mapped = _map_language_token(value.strip())
            if mapped is not None:
                return mapped

    if bfcl_utils.is_java(test_category):
        return ReturnFormat.JAVA
    if bfcl_utils.is_js(test_category):
        return ReturnFormat.JAVASCRIPT
    return ReturnFormat.PYTHON


def _return_format_to_language(return_format):
    if return_format.name == "JAVA":
        return Language.JAVA
    if return_format.name == "JAVASCRIPT":
        return Language.JAVASCRIPT
    return Language.PYTHON


def _ensure_model_config(model_name: Optional[str], underscore_to_dot: bool) -> str:
    name = model_name or "unknown_model"
    key = name.replace("_", "/")
    if key not in bfcl_model_config.MODEL_CONFIG_MAPPING:
        bfcl_model_config.MODEL_CONFIG_MAPPING[key] = ModelConfig(
            model_name=name,
            display_name=name,
            url="",
            org="local",
            license="",
            model_handler=None,
            is_fc_model=False,
            underscore_to_dot=underscore_to_dot,
        )
    return name


def _init_stats() -> Dict[str, int]:
    return {
        "total_samples": 0,
        "evaluated_samples": 0,
        "missing_pred": 0,
        "correct": 0,
        "syntax_errors": 0,
        "semantic_errors": 0,
    }


def _compute_accuracy(stats: Dict[str, int]) -> Optional[float]:
    evaluated = stats["evaluated_samples"]
    if not evaluated:
        return None
    return stats["correct"] / evaluated


def _build_reason(error: Optional[List[str]], status: str) -> Optional[str]:
    if status == "ok":
        return "passed"
    if error:
        return "; ".join(str(e) for e in error)
    if status == "missing_pred":
        return "missing prediction"
    if status == "missing_gold":
        return "missing ground truth"
    return None


def _strip_think_blocks(text: str) -> Tuple[str, str]:
    """
    Remove <think>...</think> blocks. If a closing tag is missing, return status "truncated_think".
    """
    if text is None:
        return "", "ok"
    s = str(text)
    start = s.find("<think>")
    if start == -1:
        return s, "ok"

    parts = []
    pos = 0
    while True:
        start = s.find("<think>", pos)
        if start == -1:
            parts.append(s[pos:])
            return "".join(parts), "ok"
        end = s.find("</think>", start + len("<think>"))
        if end == -1:
            return "", "truncated_think"
        parts.append(s[pos:start])
        pos = end + len("</think>")


def _stringify_json_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _coerce_calls_to_strings(
    decoded: List[Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Dict[str, Any]]]:
    coerced: List[Dict[str, Dict[str, Any]]] = []
    for call in decoded:
        new_call: Dict[str, Dict[str, Any]] = {}
        for func_name, params in call.items():
            new_params = {k: _stringify_json_value(v) for k, v in params.items()}
            new_call[func_name] = new_params
        coerced.append(new_call)
    return coerced


def run_evaluation(args) -> None:
    _configure_bfcl_paths(Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT)

    pred_map, duplicates = _index_predictions(_read_jsonl(args.pred_file))
    first_pred = _read_first_pred_record(args.pred_file)
    model_name = first_pred.get("model_name_or_path") or first_pred.get("model") or "unknown_model"
    model_name = _ensure_model_config(model_name, args.underscore_to_dot)

    categories = _normalize_categories(args.test_categories)
    per_category_limit = args.num_samples if args.num_samples and args.num_samples > 0 else None

    overall = _init_stats()
    per_category: Dict[str, Dict[str, int]] = {}

    for test_category in categories:
        entries = bfcl_utils.load_dataset_entry(test_category)
        if per_category_limit:
            entries = entries[: per_category_limit]
        ground_truth_entries = bfcl_utils.load_ground_truth_entry(test_category)
        ground_truth_map = {
            str(entry.get("id")): entry.get("ground_truth")
            for entry in ground_truth_entries
        }

        stats = _init_stats()
        per_category[test_category] = stats

        for entry in entries:
            sample_id = str(entry.get("id"))
            stats["total_samples"] += 1
            overall["total_samples"] += 1

            pred_record = pred_map.get(sample_id)
            if pred_record is None:
                stats["missing_pred"] += 1
                overall["missing_pred"] += 1
                status = "missing_pred"
                reason = "missing prediction"
                if args.verbose:
                    print(f"[{test_category}] id={sample_id} status={status} reason={reason}")
                continue

            stats["evaluated_samples"] += 1
            overall["evaluated_samples"] += 1
            prediction_text = pred_record.get("raw_output") or ""
            if "<think>" in prediction_text:
                cleaned, think_status = _strip_think_blocks(prediction_text)
                if think_status == "truncated_think":
                    status = "syntax_error"
                    error_type = "truncated_think"
                    error = ["Truncated <think> block."]
                    stats["syntax_errors"] += 1
                    overall["syntax_errors"] += 1
                    reason = _build_reason(error, status)
                    msg = f"[{test_category}] id={sample_id} status={status}"
                    if reason:
                        msg += f" reason={reason}"
                    if error_type:
                        msg += f" error_type={error_type}"
                    if args.verbose:
                        print(msg)
                    continue
                prediction_text = cleaned

            error_type = None
            error = None
            status = "ok"

            try:
                decoded = parse_json_function_call(str(prediction_text))
            except Exception as exc:
                status = "syntax_error"
                error_type = "json_parser:decoder_failed"
                error = [f"Invalid JSON. Failed to parse output. {str(exc)}"]
                stats["syntax_errors"] += 1
                overall["syntax_errors"] += 1
            else:
                if not bfcl_utils.is_function_calling_format_output(decoded):
                    status = "syntax_error"
                    error_type = "json_parser:wrong_output_format"
                    error = ["Did not output in the specified JSON format."]
                    stats["syntax_errors"] += 1
                    overall["syntax_errors"] += 1
                else:
                    return_format = _resolve_return_format(entry, test_category)
                    language = _return_format_to_language(return_format)
                    if language in (Language.JAVA, Language.JAVASCRIPT):
                        decoded = _coerce_calls_to_strings(decoded)
                    possible_answer = ground_truth_map.get(sample_id)
                    if possible_answer is None:
                        status = "missing_gold"
                        error_type = "missing_gold"
                        error = ["Missing ground truth for sample."]
                    else:
                        checker_result = ast_checker(
                            entry.get("function", []),
                            decoded,
                            possible_answer,
                            language,
                            test_category,
                            model_name,
                        )
                        if not checker_result.get("valid", False):
                            status = "semantic_error"
                            error_type = checker_result.get("error_type")
                            error = checker_result.get("error")
                            stats["semantic_errors"] += 1
                            overall["semantic_errors"] += 1
                        else:
                            stats["correct"] += 1
                            overall["correct"] += 1

            reason = _build_reason(error, status)
            msg = f"[{test_category}] id={sample_id} status={status}"
            if reason:
                msg += f" reason={reason}"
            if error_type:
                msg += f" error_type={error_type}"
            if args.verbose:
                print(msg)

    overall_accuracy = _compute_accuracy(overall)
    per_category_summary: Dict[str, Dict[str, Any]] = {}
    for test_category, stats in per_category.items():
        per_category_summary[test_category] = {
            **stats,
            "accuracy": _compute_accuracy(stats),
        }

    summary = {
        "dataset": "bfcl",
        "data_root": args.data_root,
        "pred_file": args.pred_file,
        "num_samples_per_category": args.num_samples,
        "categories": categories,
        "overall": {
            **overall,
            "accuracy": overall_accuracy,
            "duplicates": duplicates,
        },
        "by_category": per_category_summary,
    }

    summary_path = (
        Path(args.summary_file)
        if args.summary_file
        else Path("eval") / "bfcl_eval_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("BFCL single-turn evaluation summary")
    for test_category in categories:
        stats = per_category_summary[test_category]
        acc = stats.get("accuracy")
        acc_text = "N/A" if acc is None else f"{acc:.2%}"
        print(
            f"{test_category}: "
            f"correct={stats['correct']} "
            f"evaluated={stats['evaluated_samples']} "
            f"missing={stats['missing_pred']} "
            f"syntax={stats['syntax_errors']} "
            f"semantic={stats['semantic_errors']} "
            f"accuracy={acc_text}"
        )
    overall_text = "N/A" if overall_accuracy is None else f"{overall_accuracy:.2%}"
    print("-" * 80)
    print(
        f"Overall: correct={overall['correct']} "
        f"evaluated={overall['evaluated_samples']} "
        f"missing={overall['missing_pred']} "
        f"syntax={overall['syntax_errors']} "
        f"semantic={overall['semantic_errors']} "
        f"accuracy={overall_text}"
    )
    if duplicates:
        print(f"Prediction duplicates skipped: {duplicates}")
    print("=" * 80)



def parse_args():
    parser = argparse.ArgumentParser(
        description="BFCL v4 single-turn evaluation runner (prompt-only, no per-sample output).",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        required=True,
        help="JSONL file produced by data.text_models_generate.py.",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default=None,
        help="Optional path to write summary JSON (defaults to eval/bfcl_eval_summary.json).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Root directory of BFCL data (contains BFCL_v4_*.json).",
    )
    parser.add_argument(
        "--test_categories",
        type=str,
        default=None,
        help="Optional category list or group names; full scope is always enforced.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Number of samples per category to evaluate; 0 means all.",
    )
    parser.add_argument(
        "--underscore_to_dot",
        action="store_true",
        help="Treat underscores in function names as dots for checker compatibility.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print per-sample evaluation status (default on).",
    )
    parser.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Disable per-sample evaluation prints.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
