#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLI helpers for generation scripts.
All dataset scripts should compose these to keep flags consistent.
"""

from argparse import ArgumentParser

from common.text_model_utils import add_common_model_args


def add_common_gen_args(parser: ArgumentParser) -> ArgumentParser:
    """Attach generation-related arguments shared by text-only tasks."""
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="每条样本最多生成多少个新 token。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="采样温度；表示贪心，不做随机采样。",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.1,
        help="nucleus sampling 的 top_p 截断阈值。",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="重复惩罚（>1 加强原文不重复）。",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.2,
        help="频率惩罚（>0 减少重复词的再次出现）。",
    )
    return parser


def build_common_parser(description: str) -> ArgumentParser:
    """
    Convenience builder that adds model args + generation args.
    Dataset scripts can still append their own data-related flags.
    """
    parser = ArgumentParser(description=description)
    parser = add_common_model_args(parser)
    parser = add_common_gen_args(parser)
    return parser
