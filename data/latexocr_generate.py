#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
latexocr_generate.py

在 LaTeX_OCR 数据集上调用 Qwen3-VL 等多模态模型进行生成：
- 从本地 parquet 数据加载图像 + LaTeX 文字
- 为每条样本构造包含图像的 prompt，请模型输出对应 LaTeX 代码
- 不做正确性验证，直接打印模型生成

用法示例（在 datacode 目录下执行）：

python -m data.latexocr_generate \
  --model_name_or_path ../Models/Qwen3-VL-2B-Instruct \
  --data_file ../datasets/LaTeX_OCR/human_handwrite/train-00000-of-00001.parquet \
  --split train \
  --num_samples 200 \
  --max_new_tokens 128 \
  --device cuda \
  --temperature 0
"""

import argparse
import io
import textwrap
from typing import Dict, Any

from datasets import load_dataset
from PIL import Image
from transformers import AutoTokenizer
from common.cli import add_common_gen_args, add_common_model_args
from common.vl_model_utils import (
    build_vl_inputs,
    load_vl_model_and_processor,
)
from common.vl_constrained_genenration import VLConstrainedGenerator
from common.cg_utils import load_template


# ============== Prompt 模板 ==============

BASE_PROMPT = (
    "You are a LaTeX OCR assistant. "
    "Given an image of a mathematical expression, transcribe it into LaTeX. "
    "Only output the LaTeX code WITHOUT any text."
)


def build_prompt_text(example_latex: str) -> str:
    """构造带有示例的纯文本提示。"""
    if example_latex:
        return (
            f"{BASE_PROMPT}\n\nExample LaTeX: {example_latex}\n"
            "Now transcribe the new image WITHOUT any text:"
        )
    return BASE_PROMPT


def load_image_from_bytes(image_field: Dict[str, Any]) -> Image.Image:
    """将 parquet 里的图片字段转为 PIL Image；兼容已解码的 PIL 对象。"""
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    img_bytes = image_field["bytes"]
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _normalize_dollar_block(text: Any) -> str:
    """提取 $$ ... $$ 内的内容并去除所有空白符，用于后处理比较。"""
    s = str(text).strip()
    start = s.find("$$")
    end = s.rfind("$$")
    if start != -1 and end != -1 and end > start + 2:
        s = s[start + 2 : end]
    return "".join(s.split())


# ============== 主流程 ==============

def run_generation(
    model_name_or_path: str,
    data_file: str,
    split: str,
    num_samples: int,
    max_new_tokens: int,
    device: str = "cuda",
    temperature: float = 0.0,
    top_p: float = 1.0,
    trust_remote_code: bool = True,
    torch_dtype: str = "auto",
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    mode: str = "prompt",
    constraint_template: str | None = None,
):
    # 1. 加载数据集（本地 parquet）
    ds_dict = load_dataset(
        "parquet",
        data_files={split: data_file},
    )
    assert split in ds_dict, f"split {split!r} not found in data file; available: {list(ds_dict.keys())}"
    ds = ds_dict[split]

    print(f"Loaded dataset file: {data_file} [{split}]")
    print(f"num_rows = {len(ds)}")

    # 2. 加载模型 & processor
    model, processor = load_vl_model_and_processor(
        model_name_or_path=model_name_or_path,
        device=device,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )

    # 2.1 创建 VL 约束生成器（可选）
    # outlines/llguidance 需要 fast tokenizer，因此这里显式加载 fast 版本并传下去
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    generator = VLConstrainedGenerator(
        mode=mode,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        trust_remote_code=trust_remote_code,
        model_name_or_path=model_name_or_path,
    )
    constraint_content, constraint_type = load_template(constraint_template)

    # 3. 遍历前 num_samples 个样本
    if num_samples == -1:
        n = len(ds)
    else:
        n = min(num_samples, len(ds))
    print(f"\n=== Start generation (first {n} samples) ===\n")

    example_latex = ds[0].get("text", "") if len(ds) > 0 else ""
    prompt_text = build_prompt_text(example_latex)
    correct = 0

    for idx in range(n):
        sample = ds[idx]
        image = load_image_from_bytes(sample["image"])
        gold_latex = sample.get("text")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Print short preview (without showing the image)
        print("#" * 80)
        print(f"[Sample {idx}]")
        print("- Prompt preview (truncated) -")
        print(textwrap.shorten(prompt_text, width=200, placeholder=" ..."))
        print("-" * 80)

        # 4. Build inputs and generate
        inputs = build_vl_inputs(messages, processor, model)
        

        # For outlines/guidance, we need to pass the prompt_text explicitly
        if mode in ("outlines", "guidance"):
            inputs["prompt_text"] = prompt_text

        if mode == "outlines":
            outlines_prompt = None

            # MiniCPM-V: tokenizer/processor chat_template 通常无法处理 content(list[image,text])，这里手动抽取文本。
            cls_name = processor.__class__.__name__.lower()
            if "minicpm" in cls_name:
                text_parts = []
                for msg in messages:
                    for item in msg.get("content", []):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                text = "\n\n".join([t for t in text_parts if t is not None])
                if "<image" not in text:
                    text = "<image>./</image>\n" + text
                outlines_prompt = text
            else:
                # Prefer tokenizer chat template (some processors don't ship one)
                if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
                    # 注意：许多 chat_template 不支持多模态 content(list)。这里只用于支持纯文本模板。
                    outlines_prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                # Fallback to processor chat template if available
                elif hasattr(processor, "apply_chat_template") and getattr(processor, "chat_template", None):
                    outlines_prompt = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                # Final fallback: plain text prompt
                else:
                    outlines_prompt = prompt_text
        else:
            outlines_prompt = prompt_text

        output_text = generator.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            # Pass raw prompt+image for outlines multimodal adapter
            prompt_text=outlines_prompt,
            image=image,
            # Pass constraint info
            json_schema=constraint_content if constraint_type == "json" else None,
            cfg=constraint_content if constraint_type == "cfg" else None,
            ebnf=constraint_content if constraint_type == "ebnf" else None,
            regex=constraint_content if constraint_type == "regex" else None,
            guidance_template=constraint_content if constraint_type == "guidance" else None,
        )

        print("Model output:")
        print(output_text.strip())
        if gold_latex is not None:
            pred_norm = _normalize_dollar_block(output_text)
            gold_norm = _normalize_dollar_block(gold_latex)
            is_correct = pred_norm == gold_norm
            correct += int(is_correct)
            print(f"[gold latex] {gold_latex}")
            print(f"[pred norm] {pred_norm}")
            print(f"[gold norm] {gold_norm}")
            print(f"[match] {'correct' if is_correct else 'wrong'}")
        print("\n")

    if n > 0:
        acc = correct / n if n else 0.0
        print("=" * 80)
        print(f"Accuracy: {correct}/{n} = {acc:.2%}")
        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LaTeX OCR generation with a multimodal model.",
    )
    parser = add_common_model_args(parser)
    parser = add_common_gen_args(parser)
    parser.set_defaults(max_new_tokens=512)
    parser.add_argument(
        "--data_file",
        type=str,
        default="datasets/LaTeX_OCR/small/validation-00000-of-00001.parquet",
        help="LaTeX_OCR data file (parquet).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split; must match the key in data_files. Default: validation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to generate from the dataset.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="prompt",
        choices=["prompt", "xgrammar", "guidance", "outlines", "llama_cpp"],
        help="生成模式：prompt（仅 prompt）、xgrammar、guidance、outlines、llama_cpp。",
    )
    parser.add_argument(
        "--constraint_template",
        type=str,
        default=None,
        help=(
            "约束模板路径：.json(JSON Schema)/.cfg(CFG)/.ebnf/.regex 等。"
            " 对 guidance 模式，也支持 guidance:template_name。"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    trust_remote_code = not args.no_trust_remote_code

    run_generation(
        model_name_or_path=args.model_name_or_path,
        data_file=args.data_file,
        split=args.split,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        trust_remote_code=trust_remote_code,
        torch_dtype=args.torch_dtype,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        mode=args.mode,
        constraint_template=args.constraint_template,
    )


if __name__ == "__main__":
    main()
