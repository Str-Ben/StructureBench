#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generic evaluation entrypoint.
- Reads JSONL outputs from text_models_generate.py / vl_models_generate.py.
- Loads dataset records and aligns by sample_id.
- Dispatches to dataset-specific evaluation hooks (to be implemented).
"""

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None

try:
    from eval.folio_eval import evaluate_sample as evaluate_folio_sample, load_folio_dataset
except Exception:  # pragma: no cover - optional dependency
    evaluate_folio_sample = None
    load_folio_dataset = None
from data import prompts as prompt_lib

from eval.eval_logging import append_eval_log

def _json_default(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _parse_json_arg(raw: Optional[str], arg_name: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - CLI validation
        raise argparse.ArgumentTypeError(f"Invalid JSON for {arg_name}: {exc}") from exc


def _get_field(item: Any, key: str, default=None):
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _infer_data_format(data_file: str) -> str:
    suffix = Path(data_file).suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix in {".csv", ".tsv"}:
        return "csv"
    return "parquet"


def _resolve_data_file(
    dataset: str,
    data_file: Optional[str],
    dataset_config: Optional[str],
    subset: Optional[str],
    split: Optional[str],
) -> str:
    if data_file:
        return data_file
    if hasattr(prompt_lib, "_resolve_data_file"):
        return prompt_lib._resolve_data_file(
            dataset,
            data_file,
            dataset_config=dataset_config,
            subset=subset,
            split=split,
        )
    spec = getattr(prompt_lib, "DATASET_SPECS", {}).get(dataset)
    if spec and getattr(spec, "default_data_file", None):
        return spec.default_data_file
    raise ValueError(
        f"data_file is required for dataset {dataset!r}. "
        "Pass --data_file or configure a default path."
    )


def _resolve_split(dataset: str, split: Optional[str]) -> str:
    if hasattr(prompt_lib, "_resolve_split"):
        return prompt_lib._resolve_split(dataset, split)
    spec = getattr(prompt_lib, "DATASET_SPECS", {}).get(dataset)
    return split or (spec.default_split if spec else "test")


def _resolve_id_field(dataset: str, id_field: Optional[str]) -> Optional[str]:
    if id_field:
        return id_field
    spec = getattr(prompt_lib, "DATASET_SPECS", {}).get(dataset)
    return getattr(spec, "id_field", None) if spec else None


def _resolve_sample_id(sample: Any, idx: int, id_field: Optional[str]) -> str:
    if id_field:
        value = _get_field(sample, id_field, None)
        if value is not None:
            return str(value)
    for candidate in ("id", "idx", "uid", "guid", "task_id", "instance_id", "unique_id"):
        value = _get_field(sample, candidate, None)
        if value is not None:
            return str(value)
    return f"idx-{idx}"


def _invoke_callable(func, **candidate_kwargs):
    sig = inspect.signature(func)
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    call_kwargs: Dict[str, Any] = {}
    for key, value in candidate_kwargs.items():
        if key in sig.parameters or accepts_var_kw:
            call_kwargs[key] = value
    return func(**call_kwargs)


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


def load_generic_dataset(
    dataset: str,
    data_file: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    data_format: Optional[str] = None,
    dataset_config: Optional[str] = None,
    subset: Optional[str] = None,
    **_: Any,
) -> Iterable[Any]:
    if load_dataset is None:
        raise RuntimeError("datasets is required to load dataset records.")
    split = _resolve_split(dataset, split)
    data_file = _resolve_data_file(dataset, data_file, dataset_config, subset, split)
    data_format = data_format or _infer_data_format(data_file)

    ds_dict = load_dataset(data_format, data_files={split: data_file})
    if split not in ds_dict:
        raise ValueError(f"split {split!r} not found in data file; available: {list(ds_dict.keys())}")
    ds = ds_dict[split]

    limit = len(ds) if not num_samples or num_samples <= 0 else min(num_samples, len(ds))
    for idx in range(limit):
        yield ds[idx]


def _default_evaluate_sample(*_, **__) -> Dict[str, Any]:
    return {"status": "not_implemented"}


@dataclass
class EvalSpec:
    dataset: str
    load_dataset: Optional[Any] = None
    evaluate_sample: Optional[Any] = None
    sample_id_field: Optional[str] = None


EVAL_SPECS: Dict[str, EvalSpec] = {}
if evaluate_folio_sample and load_folio_dataset:
    EVAL_SPECS["folio"] = EvalSpec(
        dataset="folio",
        load_dataset=load_folio_dataset,
        evaluate_sample=evaluate_folio_sample,
        sample_id_field="example_id",
    )


def _resolve_eval_spec(dataset: str) -> EvalSpec:
    spec = EVAL_SPECS.get(dataset)
    if spec is None:
        spec = EvalSpec(dataset=dataset)
    if spec.load_dataset is None:
        spec.load_dataset = load_generic_dataset
    if spec.evaluate_sample is None:
        spec.evaluate_sample = _default_evaluate_sample
    if spec.sample_id_field is None:
        spec.sample_id_field = _resolve_id_field(dataset, None)
    return spec


@dataclass
class EvalContext:
    dataset: str
    data_file: Optional[str]
    split: str
    pred_file: str
    output_file: str
    eval_kwargs: Dict[str, Any]


def run_evaluation(args):
    dataset_kwargs = _parse_json_arg(args.dataset_kwargs, "--dataset_kwargs")
    eval_kwargs = _parse_json_arg(args.eval_kwargs, "--eval_kwargs")

    eval_spec = _resolve_eval_spec(args.dataset)
    split = _resolve_split(args.dataset, args.split)
    data_file = args.data_file
    if data_file is None and eval_spec.load_dataset == load_generic_dataset:
        data_file = _resolve_data_file(
            args.dataset,
            args.data_file,
            dataset_config=args.dataset_config,
            subset=args.subset,
            split=split,
        )
    ctx = EvalContext(
        dataset=args.dataset,
        data_file=data_file,
        split=split,
        pred_file=args.pred_file,
        output_file=args.output_file,
        eval_kwargs=eval_kwargs,
    )

    pred_map, duplicates = _index_predictions(_read_jsonl(args.pred_file))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append_output else "w"

    stats = {
        "total": 0,
        "with_pred": 0,
        "missing_pred": 0,
        "ok": 0,
        "not_implemented": 0,
        "error": 0,
        "syntax_errors": 0,
        "semantic_errors": 0,
        "duplicates": duplicates,
    }

    dataset_iter = _invoke_callable(
        eval_spec.load_dataset,
        dataset=args.dataset,
        data_file=data_file,
        split=split,
        num_samples=args.num_samples,
        data_format=args.data_format,
        dataset_config=args.dataset_config,
        subset=args.subset,
        **dataset_kwargs,
    )

    with output_path.open(mode, encoding="utf-8") as fout:
        resolved_id_field = _resolve_id_field(args.dataset, args.id_field)
        for idx, sample in enumerate(dataset_iter):
            stats["total"] += 1
            sample_id = _resolve_sample_id(sample, idx, resolved_id_field)
            pred_record = pred_map.get(sample_id)

            if pred_record is None:
                stats["missing_pred"] += 1
                record = {
                    "dataset": args.dataset,
                    "sample_id": sample_id,
                    "status": "missing_pred",
                }
                fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                if args.verbose:
                    print(f"[Sample {idx}] id={sample_id} status=missing_pred")
                continue

            stats["with_pred"] += 1
            prediction_text = _get_field(pred_record, "raw_output", None)

            try:
                eval_result = _invoke_callable(
                    eval_spec.evaluate_sample,
                    sample=sample,
                    prediction=prediction_text,
                    pred_record=pred_record,
                    sample_id=sample_id,
                    context=ctx,
                    **eval_kwargs,
                )
                if not isinstance(eval_result, Mapping):
                    eval_result = {}
            except NotImplementedError as exc:
                eval_result = {"status": "not_implemented", "error": str(exc)}
            except Exception as exc:
                eval_result = {"status": "error", "error": str(exc)}

            status = eval_result.get("status", "ok")
            if status == "ok":
                stats["ok"] += 1
            elif status == "not_implemented":
                stats["not_implemented"] += 1
            elif status == "error":
                stats["error"] += 1
            elif status == "syntax error":
                stats["syntax_errors"] += 1
            else:
                stats["semantic_errors"] += 1

            record = {
                "dataset": args.dataset,
                "sample_id": sample_id,
                "status": status,
            }
            if args.store_prediction:
                record["prediction"] = prediction_text
            if args.store_pred_record:
                record["pred_record"] = pred_record
            for key, value in eval_result.items():
                if key in ("dataset", "sample_id"):
                    continue
                record[key] = value

            fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

            if args.verbose:
                print(f"[Sample {idx}] id={sample_id} status={status}")

    if args.summary_file:
        summary_path = Path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "dataset": args.dataset,
            "data_file": data_file,
            "split": split,
            "pred_file": args.pred_file,
            "output_file": args.output_file,
            "stats": stats,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total samples: {stats['total']}")
    print(f"Predictions found: {stats['with_pred']} (missing {stats['missing_pred']})")
    print(
        f"Status: ok={stats['ok']} syntax={stats['syntax_errors']} semantic={stats['semantic_errors']} "
        f"not_implemented={stats['not_implemented']} error={stats['error']}"
    )
    if stats["duplicates"]:
        print(f"Prediction duplicates skipped: {stats['duplicates']}")

    accuracy = stats["ok"] / stats["with_pred"] if stats["with_pred"] else None
    append_eval_log(
        dataset=args.dataset,
        pred_file=args.pred_file,
        output_file=args.output_file,
        accuracy=accuracy,
        syntax_errors=stats["syntax_errors"],
        semantic_errors=stats["semantic_errors"],
        total_samples=stats["total"],
        evaluated_samples=stats["with_pred"],
        missing_pred=stats["missing_pred"],
        duplicates=stats["duplicates"],
        eval_script=Path(__file__).name,
        extra={
            "data_file": data_file,
            "split": split,
            "num_samples": args.num_samples,
            "data_format": args.data_format,
            "dataset_config": args.dataset_config,
            "subset": args.subset,
            "not_implemented": stats["not_implemented"],
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic evaluation runner for StructureBench datasets.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name for evaluation.",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        required=True,
        help="JSONL file produced by text_models_generate.py or vl_models_generate.py.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to write per-sample evaluation JSONL.",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default=None,
        help="Optional path to write summary JSON.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Optional dataset file path; defaults follow prompts.py.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split forwarded to loader.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Number of samples to evaluate; 0 means all.",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default=None,
        help="Optional dataset format override (parquet/json/csv).",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Optional dataset config for datasets with multiple files.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional dataset subset (e.g., JSONSchemaBench).",
    )
    parser.add_argument(
        "--id_field",
        type=str,
        default=None,
        help="Optional field name for sample_id in dataset records.",
    )
    parser.add_argument(
        "--dataset_kwargs",
        type=str,
        default=None,
        help="JSON string of extra kwargs for dataset loader.",
    )
    parser.add_argument(
        "--eval_kwargs",
        type=str,
        default=None,
        help="JSON string of extra kwargs for evaluation logic.",
    )
    parser.add_argument(
        "--append_output",
        action="store_true",
        help="Append to output_file instead of overwriting.",
    )
    parser.add_argument(
        "--store_prediction",
        action="store_true",
        help="Store prediction text in the output JSONL.",
    )
    parser.add_argument(
        "--store_pred_record",
        action="store_true",
        help="Store full prediction record in the output JSONL.",
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
