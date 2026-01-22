#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BFCL prompt loader for StructureBench.
Builds prompt-only inputs by reusing BFCL's system prompt formatting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_DATA_ROOT = Path(r"D:\StructureBench\datasets\bfcl_eval\data")

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


def _ensure_bfcl_import(data_root: Optional[Path]):
    try:
        from eval.bfcl_eval import utils as bfcl_utils
        from eval.bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model
    except Exception as exc:
        raise RuntimeError(
            "bfcl_eval is not available. Ensure eval/bfcl_eval exists under the project root."
        ) from exc

    if data_root is not None:
        _configure_bfcl_paths(bfcl_utils, data_root)

    return bfcl_utils, system_prompt_pre_processing_chat_model


def _configure_bfcl_paths(bfcl_utils, data_root: Path) -> None:
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


def _normalize_categories(bfcl_utils, test_categories: Optional[Sequence[str]]) -> List[str]:
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


def _flatten_messages(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for message in messages:
        content = message.get("content", "")
        if content is None:
            continue
        text = str(content).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def load_bfcl_prompts(
    dataset: str = "bfcl",
    data_root: Optional[str] = None,
    test_categories: Optional[Sequence[str]] = None,
    max_per_category: Optional[int] = None,
    num_samples: Optional[int] = None,
    **_: Any,
) -> Iterable[Any]:
    bfcl_utils, system_prompt_pre_processing_chat_model = _ensure_bfcl_import(
        Path(data_root) if data_root else DEFAULT_DATA_ROOT
    )

    categories = _normalize_categories(bfcl_utils, test_categories)
    per_category_limit = max_per_category
    if per_category_limit is None or per_category_limit <= 0:
        if num_samples and num_samples > 0:
            per_category_limit = num_samples
        else:
            per_category_limit = None

    from data.prompts import PromptItem

    for test_category in categories:
        entries = bfcl_utils.load_dataset_entry(test_category)
        if per_category_limit:
            entries = entries[: per_category_limit]

        for entry in entries:
            sample_id = str(entry.get("id"))
            question = entry.get("question", [])
            first_turn = question[0] if isinstance(question, list) and question else []
            if not isinstance(first_turn, list):
                first_turn = []
            messages = [dict(msg) for msg in first_turn]
            if not messages:
                messages = [{"role": "user", "content": ""}]
            messages = system_prompt_pre_processing_chat_model(
                messages, entry.get("function", []), sample_id
            )
            prompt_text = _flatten_messages(messages)
            yield PromptItem(
                sample_id=sample_id,
                prompt_text=prompt_text,
                meta={"test_category": test_category},
            )
