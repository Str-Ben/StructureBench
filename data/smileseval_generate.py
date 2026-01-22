#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
smileseval_generate.py

在 SMILES-eval 数据集上调用任意 transformers 模型进行生成：
- 从本地 parquet 数据加载化合物描述与目标 SMILES
- 在提示词中加入一个示例
- 生成完整输出并与 gold 逐条比对，统计准确率

用法示例（在 datacode 目录下执行）：

python -m data.smileseval_generate \
  --model_name_or_path ../Models/openPangu-Embedded-1B \
  --data_file ../datasets/smiles-eval/data/test-00000-of-00001.parquet \
  --split test \
  --num_samples 20 \
  --max_new_tokens 512 \
  --device cuda \
  --temperature 0
"""

import argparse
import re
import textwrap

from datasets import load_dataset
import partialsmiles
from rdkit import Chem

from common.cli import add_common_gen_args, add_common_model_args
from common.text_model_utils import load_text_model_and_tokenizer as shared_load_text_model_and_tokenizer
from common.constrained_generation import ConstrainedGenerator
from common.cg_utils import load_template, generate_with_generator, append_jsonl_log


# ============== Prompt 模板 ==============

PROMPT_TEMPLATE = """You are a chemistry expert. Convert the description into a valid canonical SMILES string. Follow these SMILES syntax rules and dataset style:

- Use standard ASCII SMILES; output a single SMILES with no spaces or extra text.
- Use aromatic lowercase atoms (e.g., c, n) for aromatic rings; use uppercase for aliphatic atoms.
- Use ring-closure digits (1-9) to close rings and parentheses for branches.
- Use bond symbols as needed: single (implicit), double '=', triple '#'.
- Use bracket atoms for charges, isotopes, explicit hydrogens, or uncommon valence (e.g., [N+], [O-], [13C], [2H], [nH]).
- Encode stereochemistry with '@'/'@@' for chiral centers and '/' and '\\\\' for E/Z double bonds.
- Use '.' to separate disconnected components (salts/ions).
- Do not include atom mapping (e.g., [C:1]) or wildcard atoms ('*').


Example:
Description: {example_input}
Reasoning: Analyze the description and identify the molecule.
SMILES: {example_output}

Now convert the new description.
Description: {description}

SMILES:
"""

def smiles_valid(output: str) -> bool:
    """Check if a SMILES string is valid using RDKit parsing."""
    if not output:
        return False
    try:
        return Chem.MolFromSmiles(output) is not None
    except Exception:
        return False


def are_smiles_equivalent(smiles_a: str, smiles_b: str, isomeric: bool = True):
    """Check if two SMILES strings represent the same molecule."""
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return False, True
    canon_smiles_a = Chem.MolToSmiles(mol_a, isomericSmiles=isomeric)
    canon_smiles_b = Chem.MolToSmiles(mol_b, isomericSmiles=isomeric)
    return canon_smiles_a == canon_smiles_b, False


def pick_example(ds):
    """从数据集中挑一条示例（description + SMILES）。"""
    if len(ds) == 0:
        return ("", "")
    ex = ds[0]
    return ex.get("input", ""), ex.get("output", "")


def build_prompt(description: str, example_input: str, example_output: str) -> str:
    """构造包含示例的 prompt。"""
    return PROMPT_TEMPLATE.format(
        example_input=example_input.strip(),
        example_output=str(example_output).strip(),
        description=description.strip(),
    )


# ============== 主流程 ==============

def extract_final_smiles(output_text: str) -> str:
    """从模型输出中抽取 <final>...</final> 里的 SMILES，若缺失则用更严格的回退规则。"""
    if not output_text:
        return ""
    match = re.search(r"<final>(.+?)</final>", output_text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    for line in reversed(output_text.splitlines()):
        line = line.strip()
        if not line:
            continue
        tagged = re.search(r"(?i)smiles\\s*[:=]\\s*(.+)", line)
        if tagged:
            candidate = tagged.group(1).strip()
            if candidate:
                return candidate
        if " " not in line:
            return line
    return output_text.strip()


def canonicalize_smiles(smiles: str) -> str:
    """将 SMILES 转成 RDKit Canonical 形式；解析失败返回空字符串。"""
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


def run_generation(
    model_name_or_path: str,
    data_file: str,
    split: str,
    num_samples: int,
    max_new_tokens: int,
    device: str = "cuda",
    trust_remote_code: bool = True,
    temperature: float = 0.0,
    top_p: float = 1.0,
    torch_dtype: str = "auto",
    attn_implementation: str = None,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    mode: str = "prompt",
    constraint_template: str | None = None,
    log_file: str | None = None,
):
    # 1. 加载 SMILES-eval 数据集
    ds_dict = load_dataset(
        "parquet",
        data_files={split: data_file},
    )
    assert split in ds_dict, f"split {split!r} not found in data file; available: {list(ds_dict.keys())}"
    ds = ds_dict[split]

    print(f"Loaded dataset file: {data_file} [{split}]")
    print(f"num_rows = {len(ds)}")

    # 2. 加载模型 & tokenizer
    model, tokenizer, config = shared_load_text_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    is_encoder_decoder = getattr(config, "is_encoder_decoder", False)

    # 2.1 创建约束生成器（可选）
    generator = ConstrainedGenerator(
        mode=mode,
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=trust_remote_code,
        model_name_or_path=model_name_or_path,
    )
    constraint_content, constraint_type = load_template(constraint_template)

    # the space in outlines is too small
    if constraint_type == "cfg":
        constraint_content = r"[A-Za-z0-9@\+\-\[\]\(\)=#\\/\.:%]+"
        constraint_type = "regex"
    # 3. 遍历前 num_samples 个样本
    if num_samples == -1:
        n = len(ds)
    else:
        n = min(num_samples, len(ds))
    print(f"\n=== Start generation (first {n} samples) ===\n")

    example_input, example_output = pick_example(ds)
    correct = 0
    syntax_errors = 0
    semantic_errors = 0
    comparison_errors = 0

    for idx in range(n):
        sample = ds[idx]
        description = sample["input"]
        gold_smiles = sample.get("output")
        sample_id = sample.get("instance_id", f"idx-{idx}")

        prompt = build_prompt(description, example_input, example_output)

        sample_log = {
            "task": "smiles",
            "model_name_or_path": model_name_or_path,
            "data_file": data_file,
            "split": split,
            "mode": mode,
            "constraint_template": constraint_template,
            "sample_id": sample_id,
            "description": description,
            "prompt_preview": prompt[:300] + "..." if len(prompt) > 300 else prompt,
            "gold_smiles": gold_smiles,
        }

        print("#" * 80)
        print(f"[Sample {idx}] id = {sample_id}")
        print("- Prompt preview (truncated) -")
        print(textwrap.shorten(prompt, width=300, placeholder=" ..."))
        print("-" * 80)

        # 4. Generate（统一调用约束生成器）
        output_text = generate_with_generator(
            generator,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            constraint=constraint_content,
            constraint_type=constraint_type,
        )

        print("Model output:")
        print(output_text.strip())

        sample_log["raw_output"] = output_text

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
            pred_canonical = canonicalize_smiles(final_smiles)
            gold_canonical = canonicalize_smiles(gold_str)
            print(f"[gold SMILES] {gold_str}")
            print(f"[final SMILES] {final_smiles}")
            print(f"[syntax_ok] {syntax_ok}")
            print(f"[valid_ok] {valid_ok}")
            print(f"[passed_tests] {passed_tests}")
            if compiler_output:
                print("[details]")
                print(compiler_output)
            print(f"[pred canonical] {pred_canonical}")
            print(f"[gold canonical] {gold_canonical}")
            sample_log["final_smiles"] = final_smiles
            sample_log["gold_str"] = gold_str
            sample_log["syntax_ok"] = bool(syntax_ok)
            sample_log["valid_ok"] = bool(valid_ok)
            sample_log["passed_tests"] = bool(passed_tests)
            sample_log["pred_canonical"] = pred_canonical
            sample_log["gold_canonical"] = gold_canonical
            if compiler_output:
                sample_log["details"] = compiler_output

            print(f"[match] {'correct' if passed_tests else 'wrong'}")

        # 写入逐样本日志（jsonl）
        if log_file:
            append_jsonl_log(log_file, sample_log, verbose=False)

        print("\n")

    if n > 0:
        acc = correct / n if n else 0.0
        syntax_rate = syntax_errors / n if n else 0.0
        semantic_rate = semantic_errors / n if n else 0.0
        comparison_rate = comparison_errors / n if n else 0.0
        print("=" * 80)
        print(f"Accuracy: {correct}/{n} = {acc:.2%}")
        print(f"Syntax errors: {syntax_errors}/{n} = {syntax_rate:.2%}")
        print(f"Semantic errors: {semantic_errors}/{n} = {semantic_rate:.2%}")
        if comparison_errors:
            print(f"Comparison errors: {comparison_errors}/{n} = {comparison_rate:.2%}")

        if log_file:
            summary_log = {
                "task": "smiles_summary",
                "model_name_or_path": model_name_or_path,
                "data_file": data_file,
                "split": split,
                "mode": mode,
                "constraint_template": constraint_template,
                "num_samples": n,
                "correct": correct,
                "accuracy": f"{acc:.4f}",
                "syntax_errors": syntax_errors,
                "semantic_errors": semantic_errors,
                "comparison_errors": comparison_errors,
            }
            append_jsonl_log(log_file, summary_log, verbose=False)

        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run generation on SMILES-eval dataset with a transformers model.",
    )
    parser = add_common_model_args(parser)
    parser = add_common_gen_args(parser)
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
        help="约束模板路径：.json(JSON Schema)/.cfg(CFG)/.ebnf/.regex 等。",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="逐样本输出日志文件路径（jsonl）。不传则不写日志。",
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
        trust_remote_code=trust_remote_code,
        temperature=args.temperature,
        top_p=args.top_p,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        mode=args.mode,
        constraint_template=args.constraint_template,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
