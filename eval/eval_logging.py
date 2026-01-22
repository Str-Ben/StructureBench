#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shared evaluation logging helper.
Appends one JSONL record per eval run into result/eval_results.jsonl.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


LOG_FILE_NAME = "eval_results.jsonl"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _log_path() -> Path:
    return _repo_root() / "result" / LOG_FILE_NAME


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


def _extract_gen_params(pred_record: Dict[str, Any]) -> Dict[str, Any]:
    gen_params = pred_record.get("gen_params", {}) if isinstance(pred_record, dict) else {}
    if not isinstance(gen_params, dict):
        gen_params = {}
    return {
        "model_name_or_path": pred_record.get("model_name_or_path"),
        "temperature": gen_params.get("temperature"),
        "max_new_tokens": gen_params.get("max_new_tokens"),
        "top_p": gen_params.get("top_p"),
        "repetition_penalty": gen_params.get("repetition_penalty"),
        "frequency_penalty": gen_params.get("frequency_penalty"),
        "num_samples": gen_params.get("num_samples"),
        "think": gen_params.get("think"),
        "gen_params": gen_params if gen_params else None,
    }


def append_eval_log(
    *,
    dataset: str,
    pred_file: str,
    output_file: str,
    accuracy: Optional[float],
    syntax_errors: int,
    semantic_errors: int,
    total_samples: int,
    evaluated_samples: int,
    missing_pred: int = 0,
    duplicates: int = 0,
    eval_script: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": dataset,
        "pred_file": str(pred_file),
        "output_file": str(output_file),
        "accuracy": accuracy,
        "syntax_errors": int(syntax_errors),
        "semantic_errors": int(semantic_errors),
        "total_samples": int(total_samples),
        "evaluated_samples": int(evaluated_samples),
        "missing_pred": int(missing_pred),
        "duplicates": int(duplicates),
    }
    if eval_script:
        record["eval_script"] = eval_script

    pred_record = _read_first_pred_record(pred_file)
    record.update(_extract_gen_params(pred_record))

    if extra:
        record.update(extra)

    log_path = _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError:
        # Best-effort logging: avoid breaking eval runs if the log cannot be written.
        pass
