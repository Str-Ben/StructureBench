#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generic text-only generation entrypoint.
- Uses data.prompts to prepare per-dataset prompts.
- Runs a shared transformers generation loop.
- Writes sample_id-aligned JSONL for downstream evaluation.
"""

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from common.cli import add_common_gen_args
from common.cg_utils import load_template, generate_with_generator
from common.constrained_generation import ConstrainedGenerator
from common.text_model_utils import (
    add_common_model_args,
    load_text_model_and_tokenizer,
)
from data import prompts as prompt_lib


def _json_default(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _parse_prompt_kwargs(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - CLI validation
        raise argparse.ArgumentTypeError(f"Invalid JSON for --prompt_kwargs: {exc}") from exc


def _get_field(item: Any, key: str, default=None):
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _resolve_prompt_loader(dataset: str):
    """Locate a loader from prompts.py; supports either get_prompt_loader or load_prompts."""
    if hasattr(prompt_lib, "get_prompt_loader"):
        loader = prompt_lib.get_prompt_loader(dataset)
        if loader is None:
            raise KeyError(f"dataset {dataset!r} not registered in prompts.py")
        return loader
    if hasattr(prompt_lib, "load_prompts"):
        return lambda **kwargs: prompt_lib.load_prompts(dataset=dataset, **kwargs)
    raise RuntimeError("prompts.py must expose load_prompts(dataset=...) or get_prompt_loader(name).")


def _invoke_prompt_loader(
    loader,
    dataset: str,
    data_file: Optional[str],
    split: str,
    num_samples: int,
    prompt_kwargs: Dict[str, Any],
) -> Iterable[Any]:
    """Call loader with only the kwargs it accepts to keep compatibility across datasets."""
    sig = inspect.signature(loader)
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    candidate_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "dataset_name": dataset,
        "data_file": data_file,
        "split": split,
        "num_samples": num_samples,
        **prompt_kwargs,
    }
    call_kwargs: Dict[str, Any] = {}
    for key, value in candidate_kwargs.items():
        if key in sig.parameters or accepts_var_kw:
            call_kwargs[key] = value
    return loader(**call_kwargs)


def _ensure_prompt_text(item: Any, idx: int) -> str:
    prompt_text = _get_field(item, "prompt_text", None)
    if prompt_text is None:
        prompt_text = _get_field(item, "prompt", None)
    if prompt_text is None:
        raise ValueError(f"Prompt missing for sample {idx}; expected 'prompt_text' or 'prompt'.")
    return str(prompt_text)

def _get_model_basename(model_name_or_path: str) -> str:
    if not model_name_or_path:
        return ""
    return os.path.basename(model_name_or_path.rstrip("/\\"))


def _should_append_no_think(model_name_or_path: str, think_mode: str) -> bool:
    if think_mode != "off":
        return False
    model_basename = _get_model_basename(model_name_or_path)
    return model_basename in {"Qwen3-8B", "Qwen3-0.6B"}


def _append_no_think(prompt_text: str) -> str:
    if prompt_text is None:
        return "/no_think"
    stripped = prompt_text.rstrip()
    if stripped.endswith("/no_think"):
        return prompt_text
    if not stripped:
        return "/no_think"
    return f"{stripped}\n/no_think"


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


def _should_apply_global_limit(dataset: str) -> bool:
    return (dataset or "").strip().lower() != "bfcl"


def run_generation(args):
    prompt_kwargs = _parse_prompt_kwargs(args.prompt_kwargs)
    loader = _resolve_prompt_loader(args.dataset)
    prompt_iter = _invoke_prompt_loader(
        loader=loader,
        dataset=args.dataset,
        data_file=args.data_file,
        split=args.split,
        num_samples=args.num_samples,
        prompt_kwargs=prompt_kwargs,
    )

    model, tokenizer, _config = load_text_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        trust_remote_code=not args.no_trust_remote_code,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    generator = ConstrainedGenerator(
        mode=args.mode,
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=not args.no_trust_remote_code,
        model_name_or_path=args.model_name_or_path,
    )
    constraint_content, constraint_type = load_template(args.constraint_template)

    output_file = args.output_file or (Path("result") / f"{args.dataset}_pred.jsonl")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append_output else "w"

    apply_global_limit = _should_apply_global_limit(args.dataset)
    with output_path.open(mode, encoding="utf-8") as fout:
        for idx, item in enumerate(prompt_iter):
            if apply_global_limit and args.num_samples and idx >= args.num_samples:
                break
            sample_id = _get_field(item, "sample_id", f"idx-{idx}")
            prompt_text = _ensure_prompt_text(item, idx)
            if _should_append_no_think(args.model_name_or_path, args.think):
                prompt_text = _append_no_think(prompt_text)


            raw_output = generate_with_generator(
                generator,
                prompt=prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                frequency_penalty=args.frequency_penalty,
                constraint=constraint_content,
                constraint_type=constraint_type,
            )

            output_status = "ok"
            if args.think == "on" or "<think>" in str(raw_output):
                cleaned_output, output_status = _strip_think_blocks(raw_output)
                raw_output = cleaned_output

            record: Dict[str, Any] = {
                "dataset": args.dataset,
                "sample_id": sample_id,
                "raw_output": raw_output if output_status == "ok" else "",
                "model_name_or_path": args.model_name_or_path,
                "output_status": output_status,
                "gen_params": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "frequency_penalty": args.frequency_penalty,
                    "num_samples": args.num_samples,
                    "think": args.think,
                },
            }
            meta = _get_field(item, "meta", None)
            if meta:
                record["meta"] = meta
            if args.store_prompt:
                record["prompt"] = prompt_text

            fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

            if args.verbose:
                preview = prompt_text if len(prompt_text) <= 200 else f"{prompt_text[:200]} ..."
                print(f"[Sample {idx}] id={sample_id}")
                print(f"Prompt: {preview}")
                if output_status == "truncated_think":
                    print("Output: <truncated_think>\n")
                else:
                    print(f"Output: {raw_output.strip()}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic text-only generation runner using data.prompts.",
    )
    parser = add_common_model_args(parser)
    parser = add_common_gen_args(parser)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name registered in prompts.py.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Optional dataset file path; forwarded to prompts loader.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split forwarded to prompts loader.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to process (also forwarded to prompts loader).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to JSONL output file (sample_id + raw_output); defaults to result/<dataset>_pred.jsonl.",
    )
    parser.add_argument(
        "--prompt_kwargs",
        type=str,
        default=None,
        help="JSON string of extra kwargs for prompts loader (defaults to loader's full-scope behavior).",
    )
    parser.add_argument(
        "--think",
        type=str,
        default="off",
        choices=["off", "on"],
        help="Qwen3 think-mode handling: off=append /no_think for Qwen3-8B/0.6B; on=strip <think> blocks.",
    )
    parser.add_argument(
        "--append_output",
        action="store_true",
        help="Append to output_file instead of overwriting.",
    )
    parser.add_argument(
        "--store_prompt",
        action="store_true",
        help="Store prompt text in the output JSONL for debugging (off by default).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print prompt/output previews during generation (on by default).",
    )
    parser.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Disable verbose prompt/output previews.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="prompt",
        choices=["prompt", "xgrammar", "guidance", "outlines", "llama_cpp"],
        help="generation mode: prompt(only prompt)、xgrammar、guidance、outlines、llama_cpp。",
    )
    parser.add_argument(
        "--constraint_template",
        type=str,
        default=None,
        help="constraint template path: .json(JSON Schema)/.cfg(CFG)/.ebnf/.regex etc.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_generation(args)


if __name__ == "__main__":
    main()
