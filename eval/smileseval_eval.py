#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SMILES-eval evaluation runner.

Workflow:
1) Run data.text_models_generate.py to produce JSONL predictions.
2) Run this script with --pred_file and dataset args to score outputs.
3) The script loads dataset rows, aligns by sample_id, writes per-sample results,
   and prints a summary consistent with the original generate script.

Example:
python -m eval.smileseval_eval \
  --pred_file outputs/smileseval.jsonl \
  --output_file eval/smileseval_eval.jsonl \
  --data_file datasets/smiles-eval/data/test-00000-of-00001.parquet \
  --split test
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

from datasets import load_dataset
import partialsmiles
from rdkit import Chem

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
    return str(sample.get("instance_id", f"idx-{idx}"))


def smiles_valid(output: str) -> bool:
    if not output:
        return False
    try:
        return Chem.MolFromSmiles(output) is not None
    except Exception:
        return False


def are_smiles_equivalent(smiles_a: str, smiles_b: str, isomeric: bool = True):
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return False, True
    canon_smiles_a = Chem.MolToSmiles(mol_a, isomericSmiles=isomeric)
    canon_smiles_b = Chem.MolToSmiles(mol_b, isomericSmiles=isomeric)
    return canon_smiles_a == canon_smiles_b, False


def extract_final_smiles(output_text: str) -> str:
    if not output_text:
        return ""
    match = re.search(r"<final>(.+?)</final>", output_text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    for line in reversed(output_text.splitlines()):
        line = line.strip()
        if not line:
            continue
        tagged = re.search(r"(?i)smiles\s*[:=]\s*(.+)", line)
        if tagged:
            candidate = tagged.group(1).strip()
            if candidate:
                return candidate
        if " " not in line:
            return line
    return output_text.strip()


def canonicalize_smiles(smiles: str) -> str:
    if not smiles:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return ""
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return ""


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
    comparison_errors = 0
    missing_pred = 0

    with output_path.open(mode, encoding="utf-8") as fout:
        for idx in range(n):
            sample = ds[idx]
            gold_smiles = sample.get("output")
            sample_id = _resolve_sample_id(sample, idx)
            pred_record = pred_map.get(sample_id)

            if pred_record is None:
                missing_pred += 1
                record = {
                    "dataset": "smileseval",
                    "sample_id": sample_id,
                    "status": "missing_pred",
                }
                fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                if args.verbose:
                    print(f"[Sample {idx}] id={sample_id} status=missing_pred")
                continue

            output_text = str(_get_field(pred_record, "raw_output", "") or "")

            if gold_smiles is not None:
                final_smiles = extract_final_smiles(output_text)
                gold_str = str(gold_smiles).strip()
                compiler_output = ""
                parse_ok = True
                if not final_smiles:
                    parse_ok = False
                    compiler_output = "Empty SMILES"
                else:
                    try:
                        partialsmiles.ParseSmiles(final_smiles, partial=False)
                    except partialsmiles.SMILESSyntaxError as e:
                        parse_ok = False
                        compiler_output = f"Invalid Syntax: {e}"
                    except partialsmiles.ValenceError as e:
                        parse_ok = False
                        compiler_output = f"Invalid Valence: {e}"
                    except partialsmiles.KekulizationFailure as e:
                        parse_ok = False
                        compiler_output = f"Kekulization Failure: {e}"

                rdkit_ok = smiles_valid(final_smiles)
                syntax_ok = parse_ok and rdkit_ok
                if parse_ok and not rdkit_ok and not compiler_output:
                    compiler_output = "Otherwise Invalid SMILES"

                valid_ok = syntax_ok
                passed_tests = False
                if not valid_ok:
                    syntax_errors += 1
                else:
                    equivalent, error = are_smiles_equivalent(final_smiles, gold_str)
                    passed_tests = equivalent and not error
                    if error:
                        comparison_errors += 1
                        compiler_output = (
                            "SMILES equivalence check failed. "
                            "Ensure both prediction and reference are parseable."
                        )
                    elif not passed_tests:
                        semantic_errors += 1
                        compiler_output = (
                            "SMILES is valid but does not match reference.\n"
                            f"Extracted: {final_smiles}\nReference: {gold_str}"
                        )

                correct += int(passed_tests)
                status = "correct" if passed_tests else ("syntax error" if not valid_ok else "wrong")
                record = {
                    "dataset": "smileseval",
                    "sample_id": sample_id,
                    "status": status,
                    "syntax_ok": syntax_ok,
                    "final_smiles": final_smiles,
                    "gold_smiles": gold_str,
                    "details": compiler_output,
                    "pred_canonical": canonicalize_smiles(final_smiles),
                    "gold_canonical": canonicalize_smiles(gold_str),
                }
            else:
                record = {
                    "dataset": "smileseval",
                    "sample_id": sample_id,
                    "status": "no_gold",
                }

            fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            if args.verbose:
                print(f"[Sample {idx}] id={sample_id} status={record['status']}")

    accuracy = correct / n if n else None
    if n > 0:
        acc = accuracy or 0.0
        syntax_rate = syntax_errors / n if n else 0.0
        semantic_rate = semantic_errors / n if n else 0.0
        comparison_rate = comparison_errors / n if n else 0.0
        print("=" * 80)
        print(f"Accuracy: {correct}/{n} = {acc:.2%}")
        print(f"Syntax errors: {syntax_errors}/{n} = {syntax_rate:.2%}")
        print(f"Semantic errors: {semantic_errors}/{n} = {semantic_rate:.2%}")
        if comparison_errors:
            print(f"Comparison errors: {comparison_errors}/{n} = {comparison_rate:.2%}")
        if missing_pred:
            print(f"Missing predictions: {missing_pred}/{n}")
        if duplicates:
            print(f"Prediction duplicates skipped: {duplicates}")
        print("=" * 80)

    append_eval_log(
        dataset="smileseval",
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
            "comparison_errors": comparison_errors,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SMILES-eval predictions against reference SMILES.",
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
        default="datasets/smiles-eval/data/test-00000-of-00001.parquet",
        help="SMILES-eval data file (parquet).",
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
