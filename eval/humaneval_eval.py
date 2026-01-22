#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HumanEval evaluation runner.

Workflow:
1) Run data.text_models_generate.py to produce JSONL predictions.
2) Run this script with --pred_file and dataset args to score outputs.
3) The script loads dataset rows, aligns by sample_id, executes tests when possible,
   and prints a summary consistent with the original generate script.

Example:
python -m eval.humaneval_eval \
  --pred_file outputs/humaneval.jsonl \
  --output_file eval/humaneval_eval.jsonl \
  --data_file datasets/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet \
  --split test
"""

import argparse
import json
import re
import signal
import textwrap
import traceback
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
    return str(sample.get("task_id", f"idx-{idx}"))


def cleanup_generated_code(code: str) -> str:
    if code is None:
        return ""
    text = str(code)
    stripped = text.lstrip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.rstrip()


def strip_think_content(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def extract_first_code_block(text: str) -> str:
    if not text or "```" not in text:
        return ""
    match = re.search(r"```(?:\s*[\w.-]+)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    after = text.split("```", 1)[1]
    block_lines = after.splitlines()
    if block_lines:
        first_line = block_lines[0].strip()
        looks_like_lang = (
            len(block_lines) > 1
            and first_line
            and len(first_line.split()) == 1
            and not first_line.startswith(("def ", "class ", "from ", "import "))
        )
        if looks_like_lang:
            block_lines = block_lines[1:]
    return "\n".join(block_lines).strip()


def postprocess_generated_code(raw_output: str) -> str:
    if raw_output is None:
        return ""
    stripped = strip_think_content(str(raw_output)).strip()
    if "```" not in stripped:
        return stripped
    if stripped.startswith("```"):
        return cleanup_generated_code(stripped)
    block = extract_first_code_block(stripped)
    return block or cleanup_generated_code(stripped)


def build_executable_script(task_prompt: str, generated_code: str, test_code: str, entry_point: str) -> str:
    entry_literal = repr(entry_point)
    driver = textwrap.dedent(
        f"""
        if 'check' not in globals():
            raise RuntimeError('check function not defined')
        _entry_point = {entry_literal}
        if _entry_point not in globals():
            raise NameError(f"entry point {{_entry_point!r}} not found in generated code")
        check(globals()[_entry_point])
        """
    ).strip()
    parts = [
        str(task_prompt or "").rstrip(),
        cleanup_generated_code(generated_code),
        str(test_code or "").rstrip(),
        driver,
    ]
    return "\n\n".join(part for part in parts if part)


def exec_with_timeout(script: str, timeout_sec: float):
    timeout = timeout_sec if timeout_sec and timeout_sec > 0 else 1.0

    def _handler(signum, frame):
        raise TimeoutError(f"Execution exceeded {timeout} seconds")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    exec_globals = {"__builtins__": __builtins__, "__name__": "__main__"}
    try:
        exec(script, exec_globals, exec_globals)
        return "passed", "passed"
    except TimeoutError as exc:
        return "runtime_error", str(exc)
    except (SyntaxError, IndentationError):
        return "compile_error", traceback.format_exc()
    except Exception:
        return "runtime_error", traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


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
    compile_errors = 0
    runtime_errors = 0
    evaluated = 0
    missing_pred = 0

    with output_path.open(mode, encoding="utf-8") as fout:
        for idx in range(n):
            sample = ds[idx]
            task = sample["prompt"]
            test_code = sample.get("test")
            entry_point = sample.get("entry_point")
            sample_id = _resolve_sample_id(sample, idx)
            pred_record = pred_map.get(sample_id)

            if pred_record is None:
                missing_pred += 1
                record = {
                    "dataset": "humaneval",
                    "sample_id": sample_id,
                    "status": "missing_pred",
                }
                fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                if args.verbose:
                    print(f"[Sample {idx}] id={sample_id} status=missing_pred")
                continue

            output_text = str(_get_field(pred_record, "raw_output", "") or "")
            cleaned_output = postprocess_generated_code(output_text)

            if test_code and entry_point:
                evaluated += 1
                script = build_executable_script(task, cleaned_output, test_code, entry_point)
                status, err_msg = exec_with_timeout(script, args.eval_timeout)
                if status == "passed":
                    correct += 1
                elif status == "compile_error":
                    compile_errors += 1
                else:
                    runtime_errors += 1
                record = {
                    "dataset": "humaneval",
                    "sample_id": sample_id,
                    "status": status,
                    "error": None if status == "passed" else err_msg,
                }
            else:
                record = {
                    "dataset": "humaneval",
                    "sample_id": sample_id,
                    "status": "missing_test",
                }

            fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

            if args.verbose:
                print(f"[Sample {idx}] id={sample_id} status={record['status']}")

    accuracy = correct / evaluated if evaluated else None
    syntax_errors = compile_errors
    semantic_errors = runtime_errors
    if evaluated:
        acc = accuracy
        print("=" * 80)
        print(f"AC: {correct}")
        print(f"CE: {compile_errors}")
        print(f"WA: {runtime_errors}")
        print(f"Accuracy: {correct}/{evaluated} = {acc:.2%} (by executing tests)")
        if missing_pred:
            print(f"Missing predictions: {missing_pred}/{n}")
        if duplicates:
            print(f"Prediction duplicates skipped: {duplicates}")
        print("=" * 80)

    append_eval_log(
        dataset="humaneval",
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
            "eval_timeout": args.eval_timeout,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HumanEval predictions by executing tests.",
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
        default="datasets/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet",
        help="HumanEval data file (parquet).",
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
        "--eval_timeout",
        type=float,
        default=5.0,
        help="Per-sample test execution timeout (seconds).",
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
