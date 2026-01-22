#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generic vision-language generation entrypoint.
- Uses data.prompts to build chat-style messages (with optional images).
- Runs shared VLM helpers and writes JSONL outputs for evaluation.
"""

import argparse
import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from common.cli import add_common_gen_args
from common.text_model_utils import add_common_model_args
from common.vl_model_utils import (
    build_vl_inputs,
    generate_minicpm_chat,
    generate_vl,
    load_vl_model_and_processor,
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


def _merge_messages(item: Any) -> List[Dict[str, Any]]:
    """Return messages; if missing, build from prompt_text + vision_inputs."""
    messages = _get_field(item, "messages", None)
    vision_inputs = _get_field(item, "vision_inputs", None)
    prompt_text = _get_field(item, "prompt_text", None)
    if prompt_text is None:
        prompt_text = _get_field(item, "prompt", None)

    if messages is None:
        content = []
        if vision_inputs is not None:
            content.append({"type": "image", "image": vision_inputs})
        if prompt_text is not None:
            content.append({"type": "text", "text": prompt_text})
        if not content:
            raise ValueError("messages or prompt_text/vision_inputs must be provided by prompts loader.")
        messages = [{"role": "user", "content": content}]

    has_image = any(
        isinstance(msg, Mapping)
        and any(isinstance(part, Mapping) and part.get("type") == "image" for part in msg.get("content", []))
        for msg in (messages if isinstance(messages, list) else [])
    )
    if not has_image and vision_inputs is not None:
        cloned = list(messages)
        if not cloned:
            cloned.append({"role": "user", "content": []})
        cloned[0] = dict(cloned[0])
        cloned[0]["content"] = list(cloned[0].get("content", []))
        cloned[0]["content"].insert(0, {"type": "image", "image": vision_inputs})
        messages = cloned
    return messages


def _extract_image(messages: List[Dict[str, Any]], fallback=None):
    for msg in messages:
        for part in msg.get("content", []):
            if isinstance(part, Mapping) and part.get("type") == "image":
                return part.get("image")
    return fallback


def _extract_text(messages: List[Dict[str, Any]], fallback: Optional[str] = None) -> str:
    text_parts: List[str] = []
    for msg in messages:
        for part in msg.get("content", []):
            if isinstance(part, Mapping) and part.get("type") == "text":
                piece = part.get("text")
                if piece is not None:
                    text_parts.append(str(piece))
    if text_parts:
        return "\n\n".join(text_parts)
    return fallback or ""


def _is_minicpm(model) -> bool:
    return "minicpm" in model.__class__.__name__.lower()


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

    model, processor = load_vl_model_and_processor(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=not args.no_trust_remote_code,
    )
    use_minicpm_chat = args.use_minicpm_chat or (args.auto_minicpm_chat and _is_minicpm(model))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append_output else "w"

    with output_path.open(mode, encoding="utf-8") as fout:
        for idx, item in enumerate(prompt_iter):
            if args.num_samples and idx >= args.num_samples:
                break
            sample_id = _get_field(item, "sample_id", f"idx-{idx}")
            vision_inputs = _get_field(item, "vision_inputs", None)
            prompt_text = _get_field(item, "prompt_text", None) or _get_field(item, "prompt", None)

            messages = _merge_messages(item)

            if use_minicpm_chat:
                image_input = _extract_image(messages, fallback=vision_inputs)
                text_prompt = _extract_text(messages, fallback=prompt_text)
                if image_input is None:
                    raise ValueError(f"MiniCPM chat requires an image for sample {sample_id}.")
                raw_output = generate_minicpm_chat(
                    model,
                    processor,
                    image=image_input,
                    text_prompt=text_prompt or "",
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    frequency_penalty=args.frequency_penalty,
                )
            else:
                vl_inputs = build_vl_inputs(messages, processor, model)
                raw_output = generate_vl(
                    model,
                    processor,
                    vl_inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    frequency_penalty=args.frequency_penalty,
                )

            record: Dict[str, Any] = {
                "dataset": args.dataset,
                "sample_id": sample_id,
                "raw_output": raw_output,
                "model_name_or_path": args.model_name_or_path,
                "gen_params": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "frequency_penalty": args.frequency_penalty,
                    "num_samples": args.num_samples,
                    "think": None,
                },
            }
            meta = _get_field(item, "meta", None)
            if meta:
                record["meta"] = meta
            if args.store_prompt:
                record["prompt"] = prompt_text
                record["messages"] = messages

            fout.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

            if args.verbose:
                preview = prompt_text if (prompt_text and len(prompt_text) <= 200) else (f"{prompt_text[:200]} ..." if prompt_text else "<messages only>")
                print(f"[Sample {idx}] id={sample_id}")
                print(f"Prompt: {preview}")
                print(f"Output: {raw_output.strip()}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic VLM generation runner using data.prompts.",
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
        required=True,
        help="Path to JSONL output file (sample_id + raw_output).",
    )
    parser.add_argument(
        "--prompt_kwargs",
        type=str,
        default=None,
        help="JSON string of extra kwargs for prompts loader.",
    )
    parser.add_argument(
        "--append_output",
        action="store_true",
        help="Append to output_file instead of overwriting.",
    )
    parser.add_argument(
        "--store_prompt",
        action="store_true",
        help="Store prompt text/messages in the output JSONL for debugging.",
    )
    parser.add_argument(
        "--use_minicpm_chat",
        action="store_true",
        help="Force MiniCPM chat API even if auto detection is disabled.",
    )
    parser.add_argument(
        "--auto_minicpm_chat",
        action="store_true",
        default=True,
        help="Auto-switch to MiniCPM chat when the model class name contains 'MiniCPM'.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print prompt/output previews during generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_generation(args)


if __name__ == "__main__":
    main()
