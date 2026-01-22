#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Trace generation for Planetarium with xgrammar constraints.'''
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.cg_utils import load_template
from common.text_model_utils import add_common_model_args, load_text_model_and_tokenizer
from data import prompts as prompt_lib
from trace_generator.trace_utils import (
    build_xgrammar_processor,
    compute_masked_step_stats,
    ensure_prompt_text,
    get_field,
    run_trace_for_prompt,
)

DATASET_NAME = "planetarium"
DEFAULT_CONSTRAINT_TEMPLATE = "constraint_templates/pddl/pddl.ebnf"
DEFAULT_TRACE_FILE = "trace_generator/logs/planetarium_trace.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-token trace logs for Planetarium with xgrammar.",
    )
    parser = add_common_model_args(parser)
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Optional dataset file path override.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split override (defaults to dataset spec).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to trace (0 or negative means all).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to decode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Recorded temperature value (greedy decoding is always used).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Recorded top_p value (unused for greedy decoding).",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty applied before grammar masking.",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.2,
        help="Frequency penalty applied before grammar masking.",
    )
    parser.add_argument(
        "--constraint_template",
        type=str,
        default=DEFAULT_CONSTRAINT_TEMPLATE,
        help="Constraint template path (.ebnf or .json) for xgrammar.",
    )
    parser.add_argument(
        "--trace_file",
        type=str,
        default=DEFAULT_TRACE_FILE,
        help="Output JSONL path for trace logs.",
    )
    parser.add_argument(
        "--repeat_window",
        type=int,
        default=20,
        help="Window size for repeat_in_window checks.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    trust_remote_code = not args.no_trust_remote_code
    loader = prompt_lib.get_prompt_loader(DATASET_NAME)
    prompt_iter = loader(
        data_file=args.data_file,
        split=args.split,
        num_samples=args.num_samples,
    )

    model, tokenizer, config = load_text_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        trust_remote_code=trust_remote_code,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    constraint_content, constraint_type = load_template(args.constraint_template)
    if constraint_content is None or constraint_type is None:
        raise ValueError("constraint_template must resolve to a valid template.")
    grammar_processor = build_xgrammar_processor(
        tokenizer,
        config,
        constraint_content,
        constraint_type,
    )

    output_path = Path(args.trace_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, item in enumerate(prompt_iter):
            if args.num_samples and args.num_samples > 0 and idx >= args.num_samples:
                break
            sample_id = get_field(item, "sample_id", f"idx-{idx}")
            prompt_raw = ensure_prompt_text(item, idx)

            prompt_text, output_text, prompt_truncated, steps, error = run_trace_for_prompt(
                model=model,
                tokenizer=tokenizer,
                config=config,
                prompt=prompt_raw,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                frequency_penalty=args.frequency_penalty,
                grammar_processor=grammar_processor,
                repeat_window=args.repeat_window,
            )
            masked_step_count, total_steps, masked_steps_ratio = compute_masked_step_stats(
                steps
            )

            record = {
                "dataset": DATASET_NAME,
                "sample_id": sample_id,
                "output_text": output_text,
                "steps": steps,
                "masked_step_count": masked_step_count,
                "total_steps": total_steps,
                "masked_steps_ratio": masked_steps_ratio,
            }
            if error:
                record["error"] = error
            if prompt_truncated:
                record["prompt_truncated"] = True
            record["prompt"] = prompt_text
            record["repeat_window"] = args.repeat_window
            record["max_new_tokens"] = args.max_new_tokens
            record["temperature"] = args.temperature
            record["top_p"] = args.top_p
            record["repetition_penalty"] = args.repetition_penalty
            record["frequency_penalty"] = args.frequency_penalty
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
