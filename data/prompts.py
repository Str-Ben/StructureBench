#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prompt loaders for dataset-driven generation.
This module centralizes dataset selection and yields prompt items for generators.
"""

from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from datasets import load_dataset

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

from data.bfcl_prompts import load_bfcl_prompts


@dataclass
class PromptItem:
    sample_id: str
    prompt_text: str
    vision_inputs: Optional[Any] = None
    messages: Optional[List[Dict[str, Any]]] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    default_data_file: Optional[str] = None
    default_split: str = "test"
    id_field: Optional[str] = "id"
    data_format: Optional[str] = None
    is_vl: bool = False
    prompt_template: Optional[str] = None


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "bfcl": DatasetSpec(
        name="bfcl",
        default_split="test",
    ),
    "amr": DatasetSpec(
        name="amr",
        default_data_file="datasets/amr-3-parsed/data/validation-00000-of-00001.parquet",
        default_split="validation",
    ),
    "ape21": DatasetSpec(
        name="ape21",
        default_data_file="datasets/Calc-ape210k/data/train-00000-of-00001-b9f022a8492442e4.parquet",
        default_split="train",
    ),
    "bigmath": DatasetSpec(
        name="bigmath",
        default_data_file="datasets/Big-Math-RL-Verified/data/train-00000-of-00001.parquet",
        default_split="train",
    ),
    "chequessample": DatasetSpec(
        name="chequessample",
        default_data_file="datasets/cheques_sample_data/data/validation-00000-of-00001-4b3af28127dd79d2.parquet",
        default_split="validation",
        is_vl=True,
    ),
    "hermes": DatasetSpec(
        name="hermes",
        default_split="train",
        data_format="json",
    ),
    "folio": DatasetSpec(
        name="folio",
        default_data_file="datasets/FOLIO/folio_v2_validation.jsonl",
        default_split="validation",
        id_field="example_id",
    ),
    "humaneval": DatasetSpec(
        name="humaneval",
        default_data_file="datasets/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet",
        default_split="test",
    ),
    "jsschemabench": DatasetSpec(
        name="jsschemabench",
        default_split="test",
    ),
    "latexocr": DatasetSpec(
        name="latexocr",
        default_data_file="datasets/LaTeX_OCR/small/validation-00000-of-00001.parquet",
        default_split="validation",
        is_vl=True,
    ),
    "planetarium": DatasetSpec(
        name="planetarium",
        default_data_file="datasets/planetarium/data/test-00000-of-00001.parquet",
        default_split="test",
    ),
    "smcalflow": DatasetSpec(
        name="smcalflow",
        default_data_file="datasets/UDR_SMCalFlow/data/validation-00000-of-00001-4c17fe465006b08d.parquet",
        default_split="validation",
    ),
    "smileseval": DatasetSpec(
        name="smileseval",
        default_data_file="datasets/smiles-eval/data/test-00000-of-00001.parquet",
        default_split="test",
    ),
    "text2sql": DatasetSpec(
        name="text2sql",
        default_data_file="datasets/synthetic_text_to_sql/synthetic_text_to_sql_test.snappy.parquet",
        default_split="test",
    ),
}


def list_datasets() -> List[str]:
    return sorted(DATASET_SPECS.keys())


def _normalize_dataset_name(dataset: str) -> str:
    return (dataset or "").strip().lower()


def get_prompt_loader(dataset: str):
    """Return a loader callable for the dataset."""
    dataset_key = _normalize_dataset_name(dataset)
    loader = PROMPT_LOADERS.get(dataset_key)
    if loader is not None:
        return loader
    return lambda **kwargs: load_prompts(dataset=dataset_key, **kwargs)


def _infer_data_format(data_file: str) -> str:
    suffix = Path(data_file).suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix in {".csv", ".tsv"}:
        return "csv"
    return "parquet"


def _resolve_sample_id(sample: Mapping[str, Any], idx: int, id_field: Optional[str]) -> str:
    if id_field and id_field in sample:
        return str(sample[id_field])
    for candidate in ("id", "idx", "uid", "guid"):
        if candidate in sample:
            return str(sample[candidate])
    return f"idx-{idx}"


def _build_meta(sample: Mapping[str, Any], meta_fields: Optional[Sequence[str]]) -> Optional[Dict[str, Any]]:
    if not meta_fields:
        return None
    meta: Dict[str, Any] = {}
    for field in meta_fields:
        if field in sample:
            meta[field] = sample[field]
    return meta or None


def _build_prompt_text(
    dataset: str,
    sample_id: str,
    sample: Mapping[str, Any],
    prompt_template: Optional[str],
    text_field: Optional[str],
) -> str:
    template = prompt_template or DEFAULT_PROMPT_TEMPLATE
    text_value = ""
    if text_field and text_field in sample:
        text_value = str(sample[text_field])
    try:
        return template.format(dataset=dataset, sample_id=sample_id, text=text_value)
    except Exception:
        return DEFAULT_PROMPT_TEMPLATE.format(dataset=dataset)


def _resolve_data_file(
    dataset: str,
    data_file: Optional[str],
    dataset_config: Optional[str] = None,
    subset: Optional[str] = None,
    split: Optional[str] = None,
) -> str:
    spec = DATASET_SPECS.get(dataset)
    if data_file:
        return data_file
    if dataset == "hermes":
        config = dataset_config or "json_mode_singleturn"
        file_name = config.replace("_", "-")
        if not file_name.endswith(".json"):
            file_name = f"{file_name}.json"
        return f"datasets/hermes-function-calling-v1/{file_name}"
    if dataset == "jsschemabench":
        subset_name = subset or "Github_easy"
        split_name = split or "test"
        return f"datasets/JSONSchemaBench/{subset_name}/{split_name}-00000-of-00001.parquet"
    if spec and spec.default_data_file:
        return spec.default_data_file
    raise ValueError(
        f"data_file is required for dataset {dataset!r}. "
        "Pass --data_file or configure a default path."
    )


def _resolve_split(dataset: str, split: Optional[str]) -> str:
    spec = DATASET_SPECS.get(dataset)
    return split or (spec.default_split if spec else "test")


def _require_pil():
    if Image is None:
        raise RuntimeError("PIL is required for image prompts; please install Pillow.")


def _load_image_from_bytes(image_field: Dict[str, Any]):
    _require_pil()
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    img_bytes = image_field["bytes"]
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


DEFAULT_PROMPT_TEMPLATE = "[TODO:{dataset}] Fill in dataset-specific prompt here."


# =========================
# AMR
# =========================

AMR_PROMPT_TEMPLATE = """You are an expert AMR parsing assistant.
Please generate the Abstract Meaning Representation (AMR) graph for the input sentence following strict Penman notation rules:

1. **Structure**: Every node must follow the format `(variable / concept :relation child)`.
2. **Root**: The graph must start with a single root parenthesis `(`.
3. **Relations**: All relations must start with a colon (e.g., `:ARG0`, `:time`, `:mod`) followed by a space.
4. **Variables**: Assign unique variable names (e.g., `p`, `p2`) to each concept.
5. **Output**: Output **ONLY** the raw AMR string. Do not use Markdown code blocks (```), and do not add explanations.

Example:
User: {example_user}
AMR:
{example_amr}

Now parse the new sentence.
User: {user_msg}

AMR:
"""


def _amr_extract_user_and_amr(sample: Mapping[str, Any]):
    conversations = sample.get("conversations", [])
    user_msg = ""
    gold_amr = None
    if isinstance(conversations, list):
        if len(conversations) > 0:
            user_msg = conversations[0].get("content", "")
        if len(conversations) > 1:
            gold_amr = conversations[1].get("content")
    return user_msg.strip(), gold_amr


def _amr_pick_example(ds) -> Tuple[str, str]:
    if len(ds) == 0:
        return ("", "")
    user_msg, gold_amr = _amr_extract_user_and_amr(ds[0])
    return user_msg, gold_amr or ""


def _amr_build_prompt(sample: Mapping[str, Any], example_user: str, example_amr: str) -> str:
    user_msg, _ = _amr_extract_user_and_amr(sample)
    return AMR_PROMPT_TEMPLATE.format(
        example_user=example_user,
        example_amr=example_amr,
        user_msg=user_msg,
    )


def load_amr_prompts(
    data_file: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    **_: Any,
) -> Iterable[PromptItem]:
    dataset = "amr"
    data_file = _resolve_data_file(dataset, data_file)
    split = _resolve_split(dataset, split)
    ds = load_dataset("parquet", data_files={split: data_file})[split]
    example_user, example_amr = _amr_pick_example(ds)
    limit = len(ds) if not num_samples or num_samples <= 0 else min(num_samples, len(ds))
    for idx in range(limit):
        sample = ds[idx]
        prompt = _amr_build_prompt(sample, example_user, example_amr)
        sample_id = sample.get("id", f"idx-{idx}")
        yield PromptItem(sample_id=sample_id, prompt_text=prompt)


# =========================
# APE21 (Calc-ape210k)
# =========================

APE21_PROMPT_TEMPLATE = """You are a math assistant.
Follow the example: give the final equation in the form x=... inside <final>...</final>.

Example:
Problem: {example_problem}
Answer: <final>{example_equation}</final>

Now solve the new problem.
Problem:
{problem}

Equation inside <final>...</final>:
"""


def _ape21_pick_example(ds) -> Tuple[str, str]:
    if len(ds) == 0:
        return ("", "")
    sample = ds[0]
    example_problem = sample.get("question") or sample.get("question_chinese") or ""
    example_equation = str(sample.get("equation", "") or sample.get("result", ""))
    return example_problem, example_equation


def _ape21_build_prompt(problem: str, example_problem: str, example_equation: str) -> str:
    return APE21_PROMPT_TEMPLATE.format(
        example_problem=example_problem.strip(),
        example_equation=example_equation.strip(),
        problem=problem.strip(),
    )


def load_ape21_prompts(
    data_file: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    **_: Any,
) -> Iterable[PromptItem]:
    dataset = "ape21"
    data_file = _resolve_data_file(dataset, data_file)
    split = _resolve_split(dataset, split)
    ds = load_dataset("parquet", data_files={split: data_file})[split]
    example_problem, example_equation = _ape21_pick_example(ds)
    limit = len(ds) if not num_samples or num_samples <= 0 else min(num_samples, len(ds))
    for idx in range(limit):
        sample = ds[idx]
        problem = sample.get("question") or sample.get("question_chinese") or ""
        sample_id = sample.get("id", f"idx-{idx}")
        prompt = _ape21_build_prompt(problem, example_problem, example_equation)
        yield PromptItem(sample_id=sample_id, prompt_text=prompt)


# =========================
# Big-Math
# =========================

BIGMATH_PROMPT_TEMPLATE = """You are a math expert.
Follow the example: first give a brief reasoning, then put the final answer inside <final>...</final> (no extra text after </final>).

Example:
Problem: {example_problem}
Reasoning: The quadratic inequality is solved by finding its roots and checking the interval; applying the condition narrows a to the required range.
Answer: <final>{example_answer}</final>

Now solve the new problem.
Problem:
{problem}

Reasoning and Answer:
"""


def _bigmath_pick_example(ds) -> Tuple[str, str]:
    if len(ds) == 0:
        return ("", "")
    ex = ds[0]
    return ex.get("problem", ""), str(ex.get("answer", ""))


def _bigmath_build_prompt(problem: str, example_problem: str, example_answer: str) -> str:
    return BIGMATH_PROMPT_TEMPLATE.format(
        example_problem=example_problem.strip(),
        example_answer=example_answer.strip(),
        problem=problem.strip(),
    )


def load_bigmath_prompts(
    data_file: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    **_: Any,
) -> Iterable[PromptItem]:
    dataset = "bigmath"
    data_file = _resolve_data_file(dataset, data_file)
    split = _resolve_split(dataset, split)
    ds = load_dataset("parquet", data_files={split: data_file})[split]
    example_problem, example_answer = _bigmath_pick_example(ds)
    limit = len(ds) if not num_samples or num_samples <= 0 else min(num_samples, len(ds))
    for idx in range(limit):
        sample = ds[idx]
        problem = sample.get("problem", "")
        sample_id = sample.get("__index_level_0__", f"idx-{idx}")
        prompt = _bigmath_build_prompt(problem, example_problem, example_answer)
        yield PromptItem(sample_id=sample_id, prompt_text=prompt)


# =========================
# Cheques Sample (VLM)
# =========================

CHEQUE_PROMPT_TEMPLATE = """You are a cheque OCR assistant. Given a cheque image, extract fields into JSON with this schema:

{{gt_parse": {{cheque_details": [{{"amt_in_words": string}}, {{"amt_in_figures": string}}, {{"payee_name": string}}, {{"bank_name": string}}, {{"cheque_date": string}}]}}}}

Only output the JSON WITHOUT any line breaks. Follow the value formatting in the example (dates, numbers, names) exactly. For amt_in_words, keep the OCR-like abbreviated spellings as seen in the image (common patterns: For->Four, Eigt->Eight, Thre->Three, On->One, Foty->Forty, Seenty->Seventy, Sixt->Sixty, Eihty->Eighty, Eihteen->Eighteen, Twlve->Twelve, Fiteen->Fifteen, Forteen->Fourteen, Nineten->Nineteen, Sxten->Sixteen); do not auto-correct.
{example_block}"""


def _cheque_build_prompt_text(example_json: str) -> str:
    example_block = ""
    if example_json:
        example_block = (
            "\n\nExample output (match this style):\n"
            f"{example_json}\n"
            "JSON for the new cheque image:"
        )
    return CHEQUE_PROMPT_TEMPLATE.format(example_block=example_block)


def load_chequessample_prompts(
    data_file: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    **_: Any,
) -> Iterable[PromptItem]:
    dataset = "chequessample"
    data_file = _resolve_data_file(dataset, data_file)
    split = _resolve_split(dataset, split)
    ds = load_dataset("parquet", data_files={split: data_file})[split]
    example_json = ds[0].get("ground_truth", "") if len(ds) > 0 else ""
    prompt_text = _cheque_build_prompt_text(example_json)
    limit = len(ds) if not num_samples or num_samples <= 0 else min(num_samples, len(ds))
    for idx in range(limit):
        sample = ds[idx]
        image = _load_image_from_bytes(sample["image"])
        sample_id = sample.get("id", f"idx-{idx}")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        yield PromptItem(
            sample_id=sample_id,
            prompt_text=prompt_text,
            vision_inputs=image,
            messages=messages,
        )


# =========================
# Hermes Function Calling
# =========================

HERMES_PROMPT_TEMPLATE = """You are a function-calling assistant. Follow the example and reply with JSON that matches the schema in the system prompt.


Example:
{example_system}

User: {example_user}
Assistant:
{example_answer}

Now answer the new request. DO NOT output any line breaks.
{system_msg}

User: {user_msg}

Assistant:(without line breaks)"""


def _hermes_extract_fields(sample: Mapping[str, Any]):
    conversations = sample.get("conversations", [])
    system_msg = conversations[0]["value"] if len(conversations) > 0 else ""
    user_msg = conversations[1]["value"] if len(conversations) > 1 else ""
    assistant_msg = conversations[2]["value"] if len(conversations) > 2 else None
    return system_msg.strip(), user_msg.strip(), assistant_msg


def _hermes_pick_example(ds) -> Tuple[str, str, str]:
    if len(ds) == 0:
        return ("", "", "")
    system_msg, user_msg, assistant_msg = _hermes_extract_fields(ds[0])
    return system_msg, user_msg, assistant_msg or ""


def _hermes_build_prompt(sample: Mapping[str, Any], example_system: str, example_user: str, example_answer: str) -> str:
    system_msg, user_msg, _ = _hermes_extract_fields(sample)
    return HERMES_PROMPT_TEMPLATE.format(
        example_system=example_system,
        example_user=example_user,
        example_answer=example_answer,
        system_msg=system_msg,
        user_msg=user_msg,
    )


def load_hermes_prompts(
    data_file: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    dataset_config: Optional[str] = None,
    **_: Any,
) -> Iterable[PromptItem]:
    dataset = "hermes"
    data_file = _resolve_data_file(dataset, data_file, dataset_config=dataset_config, split=split)
    split = _resolve_split(dataset, split)
    ds = load_dataset("json", data_files={split: data_file})[split]
    example_system, example_user, example_answer = _hermes_pick_example(ds)
    limit = len(ds) if not num_samples or num_samples <= 0 else min(num_samples, len(ds))
    for idx in range(limit):
        sample = ds[idx]
        prompt = _hermes_build_prompt(sample, example_system, example_user, example_answer)
        sample_id = sample.get("id", f"idx-{idx}")
        yield PromptItem(sample_id=sample_id, prompt_text=prompt)


# =========================
# HumanEval
# =========================

HUMANEVAL_PROMPT_TEMPLATE = """You are a Python coding assistant.
Complete the following function according to the docstring. Output only the Python code.
You MUST include the full function signature exactly as given (the line starting with "def ..."), and then implement the body.
If the task contains import lines above the function, keep them.

Example:
Task:
{example_task}

Python code:
```python
{example_code}
```

Now solve the new task.

Task:
{task}

Python code:
"""


def _humaneval_pick_example(ds) -> Tuple[str, str]:
    if len(ds) == 0:
        return ("", "")
    s = ds[0]
    return s.get("prompt", ""), s.get("canonical_solution", "")


def _humaneval_build_prompt(task: str, example_task: str, example_code: str) -> str:
    return HUMANEVAL_PROMPT_TEMPLATE.format(
        example_task=example_task.strip(),
        example_code=example_code.strip(),
        task=task.strip(),
    )


def load_humaneval_prompts(
    data_file: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    **_: Any,
) -> Iterable[PromptItem]:
    dataset = "humaneval"
    data_file = _resolve_data_file(dataset, data_file)
    split = _resolve_split(dataset, split)
    ds = load_dataset("parquet", data_files={split: data_file})[split]
    example_task, example_code = _humaneval_pick_example(ds)
    limit = len(ds) if not num_samples or num_samples <= 0 else min(num_samples, len(ds))
    for idx in range(limit):
        sample = ds[idx]
        task = sample["prompt"]
        sample_id = sample.get("task_id", f"idx-{idx}")
        prompt = _humaneval_build_prompt(task, example_task, example_code)
        yield PromptItem(sample_id=sample_id, prompt_text=prompt)


# =========================
# JSONSchemaBench
# =========================

JSSCHEMA_PROMPT_TEMPLATE = """You are given a JSON Schema that defines the structure of a JSON object.
Follow the example and produce a single valid JSON object that conforms to the schema.

Example Schema:
{example_schema}
Example Output:
{example_output}

Now generate for the new schema.

JSON Schema:
{schema_text}

Output JSON:
"""


JSSCHEMA_EXAMPLE_SCHEMA = """{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "number"}
  },
  "required": ["name", "age"]
}"""

JSSCHEMA_EXAMPLE_OUTPUT = """{"name": "Alice", "age": 30}"""


def _jsschema_build_prompt(schema_text: str) -> str:
    return JSSCHEMA_PROMPT_TEMPLATE.format(
        example_schema=JSSCHEMA_EXAMPLE_SCHEMA,
        example_output=JSSCHEMA_EXAMPLE_OUTPUT,
        schema_text=schema_text.strip(),
    )


def load_jsschemabench_prompts(
    data_file: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    subset: Optional[str] = None,
    **_: Any,
) -> Iterable[PromptItem]:
    dataset = "jsschemabench"
    data_file = _resolve_data_file(dataset, data_file, subset=subset, split=split)
    split = _resolve_split(dataset, split)
    ds = load_dataset("parquet", data_files={split: data_file})[split]
    limit = len(ds) if not num_samples or num_samples <= 0 else min(num_samples, len(ds))
    for idx in range(limit):
        sample = ds[idx]
        schema_text = sample["json_schema"]
        sample_id = sample.get("unique_id", f"idx-{idx}")
        prompt = _jsschema_build_prompt(schema_text)
        yield PromptItem(sample_id=sample_id, prompt_text=prompt)


# =========================
# LaTeX OCR (VLM)
# =========================

LATEXOCR_BASE_PROMPT = (
    "You are a LaTeX OCR assistant. "
    "Given an image of a mathematical expression, transcribe it into LaTeX. "
    "Only output the LaTeX code WITHOUT any text."
)


def _latexocr_build_prompt_text(example_latex: str) -> str:
    if example_latex:
        return (
            f"{LATEXOCR_BASE_PROMPT}\n\nExample LaTeX: {example_latex}\n"
            "Now transcribe the new image WITHOUT any text:"
        )
    return LATEXOCR_BASE_PROMPT


def load_latexocr_prompts(
    data_file: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    **_: Any,
) -> Iterable[PromptItem]:
    dataset = "latexocr"
    data_file = _resolve_data_file(dataset, data_file)
    split = _resolve_split(dataset, split)
    ds = load_dataset("parquet", data_files={split: data_file})[split]
    example_latex = ds[0].get("text", "") if len(ds) > 0 else ""
    prompt_text = _latexocr_build_prompt_text(example_latex)
    limit = len(ds) if not num_samples or num_samples <= 0 else min(num_samples, len(ds))
    for idx in range(limit):
        sample = ds[idx]
        image = _load_image_from_bytes(sample["image"])
        sample_id = sample.get("id", f"idx-{idx}")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        yield PromptItem(
            sample_id=sample_id,
            prompt_text=prompt_text,
            vision_inputs=image,
            messages=messages,
        )


# =========================
# Planetarium
# =========================

PLANETARIUM_EXAMPLE_DESCRIPTION = (
    "A blocksworld with four blocks b1 b2 b3 b4. Initially the arm is empty, b3 is on b1, b4 is on b2, "
    "b1 and b2 are on the table, and b3 and b4 are clear. Goal: b3 on b2, b4 on b1, b1 and b2 on the table, "
    "b3 and b4 clear, and arm empty."
)

PLANETARIUM_EXAMPLE_PDDL = """(define (problem swap_to_swap_2_2)
    (:domain blocksworld)
    (:requirements :strips)
    (:objects b1 b2 b3 b4)
    (:init (arm-empty) (clear b3) (clear b4) (on b3 b1) (on b4 b2) (on-table b1) (on-table b2))
    (:goal (and (arm-empty) (on-table b2) (on-table b1) (on b3 b2) (clear b3) (on b4 b1) (clear b4)))
)"""

PLANETARIUM_PROMPT_TEMPLATE = """You are a planning assistant.
Write a complete PDDL problem definition for the description.

Output requirements:
- Output ONLY the PDDL problem; no extra text or code fences.
- Include the outer (define (problem ...)) wrapper.
- Include exactly one :domain, :requirements, :objects, :init, and :goal.
- Use concrete object names; do NOT use "..." placeholders.

Example:
Description: {example_nl}
PDDL:
{example_pddl}

Now generate PDDL for the new description.
Problem description:
{nl}

PDDL:
"""


def _planetarium_build_prompt(nl: str) -> str:
    return PLANETARIUM_PROMPT_TEMPLATE.format(
        example_nl=PLANETARIUM_EXAMPLE_DESCRIPTION.strip(),
        example_pddl=PLANETARIUM_EXAMPLE_PDDL.strip(),
        nl=nl.strip(),
    )


def load_planetarium_prompts(
    data_file: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    **_: Any,
) -> Iterable[PromptItem]:
    dataset = "planetarium"
    data_file = _resolve_data_file(dataset, data_file)
    split = _resolve_split(dataset, split)
    ds = load_dataset("parquet", data_files={split: data_file})[split]
    limit = len(ds) if not num_samples or num_samples <= 0 else min(num_samples, len(ds))
    for idx in range(limit):
        sample = ds[idx]
        nl = sample["natural_language"]
        sample_id = sample.get("id", f"idx-{idx}")
        prompt = _planetarium_build_prompt(nl)
        yield PromptItem(sample_id=sample_id, prompt_text=prompt)


# =========================
# SMCalFlow
# =========================

SMCALFLOW_PROMPT_TEMPLATE = """You are a semantic parsing assistant.
Follow the example and output ONLY the Lispress program.

Example:
User: {example_user}
Lispress: {example_lispress}

Now parse the new utterance.
User: {utterance}

Lispress:
"""


def _smcalflow_pick_example(ds) -> Tuple[str, str]:
    if len(ds) == 0:
        return ("", "")
    ex = ds[0]
    return ex.get("user_utterance", ""), ex.get("lispress", "")


def _smcalflow_build_prompt(utterance: str, example_user: str, example_lispress: str) -> str:
    return SMCALFLOW_PROMPT_TEMPLATE.format(
        example_user=example_user.strip(),
        example_lispress=(example_lispress or "").strip(),
        utterance=utterance.strip(),
    )


def load_smcalflow_prompts(
    data_file: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    **_: Any,
) -> Iterable[PromptItem]:
    dataset = "smcalflow"
    data_file = _resolve_data_file(dataset, data_file)
    split = _resolve_split(dataset, split)
    ds = load_dataset("parquet", data_files={split: data_file})[split]
    example_user, example_lispress = _smcalflow_pick_example(ds)
    limit = len(ds) if not num_samples or num_samples <= 0 else min(num_samples, len(ds))
    for idx in range(limit):
        sample = ds[idx]
        utterance = sample["user_utterance"]
        sample_id = sample.get("idx", f"idx-{idx}")
        prompt = _smcalflow_build_prompt(utterance, example_user, example_lispress)
        yield PromptItem(sample_id=sample_id, prompt_text=prompt)


# =========================
# FOLIO
# =========================

FOLIO_TASK_SPECIFICATION = (
    "Given a problem description and a question. The task is to parse the problem and the question into first-order "
    "logic formulas.\n"
    "The grammar of the first-order logic formula is defined as follows:\n"
    "1) logical conjunction of expr1 and expr2: expr1 {and} expr2\n"
    "2) logical disjunction of expr1 and expr2: expr1 {or} expr2\n"
    "3) logical exclusive disjunction of expr1 and expr2: expr1 {xor} expr2\n"
    "4) logical negation of expr1: {not}expr1\n"
    "5) expr1 implies expr2: expr1 {implies} expr2\n"
    "6) expr1 if and only if expr2: expr1 {iff} expr2\n"
    "7) logical universal quantification: {forall} x\n"
    "8) logical existential quantification: {exists} x. These are the ONLY operations in the grammar.\n"
    "Output format requirements:\n"
    "- Only output the logic program; do not include extra text or analysis.\n"
    "- Sections must be exactly: Predicates:, Premises:, Conclusion:, then '------'.\n"
    "- Each line in Premises and Conclusion MUST be: <FOL formula> ::: <English gloss>.\n"
    "- Do NOT write 'Premise:' or 'Conclusion:' inside the formula line.\n"
    "- The formula must use ONLY the grammar tokens above and parentheses; do not use words like "
    "'if', 'then', 'not', 'and', 'or', or symbols like ->, <->, =.\n"
    "- Terms are only variables or constants; do NOT nest predicates as arguments (e.g., IsStudent(Bonnie(x)) is invalid).\n"
    "- The Conclusion section must contain exactly one formula line.\n"
    "------"
)

FOLIO_EXAMPLE_QUESTION = (
    "Problem:\n"
    "All people who regularly drink coffee are dependent on caffeine. People either regularly drink coffee or joke "
    "about being addicted to caffeine. No one who jokes about being addicted to caffeine is unaware that caffeine is a "
    "drug. Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that "
    "caffeine is a drug. If Rina is not a person dependent on caffeine and a student, then Rina is either a person "
    "dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.\n"
    "Question:\n"
    "Based on the above information, is the following statement true, false, or uncertain? Rina is either a person who "
    "jokes about being addicted to caffeine or is unaware that caffeine is a drug.\n"
    "###"
)

FOLIO_EXAMPLE_RESPONSE = (
    "Predicates:\n"
    "Dependent(x) ::: x is a person dependent on caffeine.\n"
    "Drinks(x) ::: x regularly drinks coffee.\n"
    "Jokes(x) ::: x jokes about being addicted to caffeine.\n"
    "Unaware(x) ::: x is unaware that caffeine is a drug.\n"
    "Student(x) ::: x is a student.\n"
    "Premises:\n"
    "{forall} x (Drinks(x) {implies} Dependent(x)) ::: All people who regularly drink coffee are dependent on "
    "caffeine.\n"
    "{forall} x (Drinks(x) {xor} Jokes(x)) ::: People either regularly drink coffee or joke about being addicted to "
    "caffeine.\n"
    "{forall} x (Jokes(x) {implies} {not}Unaware(x)) ::: No one who jokes about being addicted to caffeine is unaware "
    "that caffeine is a drug. \n"
    "(Student(rina) {and} Unaware(rina)) {xor} {not}(Student(rina) {or} Unaware(rina)) ::: Rina is either a student and "
    "unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug.\n"
    "Conclusion:\n"
    "Jokes(rina) {xor} Unaware(rina) ::: Rina is either a person who jokes about being addicted to caffeine or is "
    "unaware that caffeine is a drug.\n"
    "------"
)


def _resolve_folio_data_file(data_file: Optional[str], split: str) -> str:
    if data_file:
        return data_file
    split_key = split.lower()
    if split_key in {"train", "training"}:
        return "datasets/FOLIO/folio_v2_train.jsonl"
    return "datasets/FOLIO/folio_v2_validation.jsonl"


def _build_folio_prompt(premises: str, conclusion: str) -> str:
    question = (
        "Based on the above information, is the following statement true, false, or uncertain? "
        f"{conclusion.strip()}"
    )
    problem_block = f"Problem:\n{premises.strip()}\nQuestion:\n{question}\n###"
    parts = [
        FOLIO_TASK_SPECIFICATION.strip(),
        "Answer the question exactly like the example below (Predicates/Premises/Conclusion, end with '------').",
        "Example:",
        FOLIO_EXAMPLE_QUESTION.strip(),
        FOLIO_EXAMPLE_RESPONSE.strip(),
        "Now solve the new problem.",
        problem_block,
    ]
    return "\n\n".join(parts)


def load_folio_prompts(
    data_file: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    **_: Any,
) -> Iterable[PromptItem]:
    dataset = "folio"
    split = _resolve_split(dataset, split)
    data_file = _resolve_folio_data_file(data_file, split)
    ds = load_dataset("json", data_files={split: data_file})[split]

    limit = len(ds) if not num_samples or num_samples <= 0 else min(num_samples, len(ds))
    for idx in range(limit):
        sample: Dict[str, Any] = ds[idx]
        sample_id = str(sample.get("example_id", f"idx-{idx}"))
        premises = sample.get("premises", "")
        conclusion = sample.get("conclusion", "")
        prompt_text = _build_folio_prompt(premises, conclusion)
        meta = {"label": sample.get("label")}
        yield PromptItem(sample_id=sample_id, prompt_text=prompt_text, messages=None, meta=meta)


# =========================
# SMILES-eval
# =========================

SMILESEVAL_PROMPT_TEMPLATE = """You are a chemistry expert. Convert the description into a valid canonical SMILES string. Follow these SMILES syntax rules and dataset style:

- Use standard ASCII SMILES; output a single SMILES with no spaces or extra text.
- Use aromatic lowercase atoms (e.g., c, n) for aromatic rings; use uppercase for aliphatic atoms.
- Use ring-closure digits (1-9) to close rings and parentheses for branches.
- Use bond symbols as needed: single (implicit), double '=', triple '#'.
- Use bracket atoms for charges, isotopes, explicit hydrogens, or uncommon valence (e.g., [N+], [O-], [13C], [2H], [nH]).
- Encode stereochemistry with '@'/'@@' for chiral centers and '/' and '\\\\' for E/Z double bonds.
- Use '.' to separate disconnected components (salts/ions).
- Do not include atom mapping (e.g., [C:1]) or wildcard atoms ('*').

You may include reasoning, but wrap the final SMILES with <final>...</final>.

Example:
Description: {example_input}
Reasoning: Analyze the description and identify the molecule.
SMILES: <final>{example_output}</final>

Now convert the new description.
Description: {description}

SMILES:
"""


def _smileseval_pick_example(ds) -> Tuple[str, str]:
    if len(ds) == 0:
        return ("", "")
    ex = ds[0]
    return ex.get("input", ""), ex.get("output", "")


def _smileseval_build_prompt(description: str, example_input: str, example_output: str) -> str:
    return SMILESEVAL_PROMPT_TEMPLATE.format(
        example_input=example_input.strip(),
        example_output=str(example_output).strip(),
        description=description.strip(),
    )


def load_smileseval_prompts(
    data_file: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    **_: Any,
) -> Iterable[PromptItem]:
    dataset = "smileseval"
    data_file = _resolve_data_file(dataset, data_file)
    split = _resolve_split(dataset, split)
    ds = load_dataset("parquet", data_files={split: data_file})[split]
    example_input, example_output = _smileseval_pick_example(ds)
    limit = len(ds) if not num_samples or num_samples <= 0 else min(num_samples, len(ds))
    for idx in range(limit):
        sample = ds[idx]
        description = sample["input"]
        sample_id = sample.get("instance_id", f"idx-{idx}")
        prompt = _smileseval_build_prompt(description, example_input, example_output)
        yield PromptItem(sample_id=sample_id, prompt_text=prompt)


# =========================
# Text2SQL
# =========================

TEXT2SQL_PROMPT_TEMPLATE = """You are a SQL generation assistant.
Given the database context and the user request, write a valid SQL query.
Use only the provided tables/columns.

Rules:
- Output a single SQL statement only, wrapped in <final>...</final>.
- Do not include any explanation or markdown/code fences; only place the SQL inside the <final> block.
- Start the first line of the SQL with a SQL keyword (SELECT/WITH/INSERT/UPDATE/DELETE).

Example:
Database context:
{example_sql_context}

User request:
{example_sql_prompt}

SQL:
<final>
{example_sql}
</final>

Now solve the new case. Output only the SQL inside <final>...</final>.

Database context:
{sql_context}

User request:
{sql_prompt}

SQL:
"""


def _text2sql_pick_example(ds) -> Tuple[str, str, str]:
    if len(ds) == 0:
        return ("", "", "")
    s = ds[0]
    return s.get("sql_context") or "", s.get("sql_prompt") or "", s.get("sql") or ""


def _text2sql_build_prompt(
    sql_context: str,
    sql_prompt: str,
    example_sql_context: str,
    example_sql_prompt: str,
    example_sql: str,
) -> str:
    return TEXT2SQL_PROMPT_TEMPLATE.format(
        sql_context=(sql_context or "").strip(),
        sql_prompt=sql_prompt.strip(),
        example_sql_context=(example_sql_context or "").strip(),
        example_sql_prompt=example_sql_prompt.strip(),
        example_sql=example_sql.strip(),
    )


def load_text2sql_prompts(
    data_file: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    **_: Any,
) -> Iterable[PromptItem]:
    dataset = "text2sql"
    data_file = _resolve_data_file(dataset, data_file)
    split = _resolve_split(dataset, split)
    ds = load_dataset("parquet", data_files={split: data_file})[split]
    example_sql_context, example_sql_prompt, example_sql = _text2sql_pick_example(ds)
    limit = len(ds) if not num_samples or num_samples <= 0 else min(num_samples, len(ds))
    for idx in range(limit):
        sample = ds[idx]
        sql_prompt = sample["sql_prompt"]
        sql_context = sample.get("sql_context") or ""
        sample_id = sample.get("id", f"idx-{idx}")
        prompt = _text2sql_build_prompt(
            sql_context,
            sql_prompt,
            example_sql_context,
            example_sql_prompt,
            example_sql,
        )
        yield PromptItem(sample_id=sample_id, prompt_text=prompt)


# =========================
# Generic fallback
# =========================


def load_prompts(
    dataset: str,
    data_file: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    *,
    data_format: Optional[str] = None,
    id_field: Optional[str] = None,
    prompt_template: Optional[str] = None,
    text_field: Optional[str] = None,
    meta_fields: Optional[Sequence[str]] = None,
    **_: Any,
) -> Iterable[PromptItem]:
    """
    Generic prompt loader for unknown datasets.
    """
    dataset_key = _normalize_dataset_name(dataset)
    spec = DATASET_SPECS.get(dataset_key, DatasetSpec(name=dataset_key))
    data_file = _resolve_data_file(dataset_key, data_file)
    split = _resolve_split(dataset_key, split)
    data_format = data_format or spec.data_format or _infer_data_format(data_file)
    id_field = id_field or spec.id_field
    prompt_template = prompt_template or spec.prompt_template

    ds_dict = load_dataset(data_format, data_files={split: data_file})
    if split not in ds_dict:
        raise ValueError(f"split {split!r} not found in data file; available: {list(ds_dict.keys())}")
    ds = ds_dict[split]

    limit = len(ds) if not num_samples or num_samples <= 0 else min(num_samples, len(ds))
    for idx in range(limit):
        sample = ds[idx]
        sample_id = _resolve_sample_id(sample, idx, id_field)
        prompt_text = _build_prompt_text(
            dataset=dataset_key,
            sample_id=sample_id,
            sample=sample,
            prompt_template=prompt_template,
            text_field=text_field,
        )
        meta = _build_meta(sample, meta_fields)
        yield PromptItem(
            sample_id=sample_id,
            prompt_text=prompt_text,
            vision_inputs=None,
            messages=None,
            meta=meta,
        )


PROMPT_LOADERS = {
    "amr": load_amr_prompts,
    "bfcl": load_bfcl_prompts,
    "ape21": load_ape21_prompts,
    "bigmath": load_bigmath_prompts,
    "chequessample": load_chequessample_prompts,
    "hermes": load_hermes_prompts,
    "humaneval": load_humaneval_prompts,
    "jsschemabench": load_jsschemabench_prompts,
    "latexocr": load_latexocr_prompts,
    "planetarium": load_planetarium_prompts,
    "folio": load_folio_prompts,
    "smcalflow": load_smcalflow_prompts,
    "smileseval": load_smileseval_prompts,
    "text2sql": load_text2sql_prompts,
}
