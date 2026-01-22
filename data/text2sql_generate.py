#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
text2sql_generate.py

在 synthetic_text_to_sql 数据集上调用任意 transformers 模型进行生成：
- 从本地 parquet 数据加载 NL 问题 + 数据库上下文
- 为每条样本构造 prompt，要求输出 SQL 查询
- 对输出做轻量后处理，并进行规范化对比（可选使用 SQL 解析器）

用法示例（在 datacode 目录下执行）：

python -m data.text2sql_generate \
  --model_name_or_path ../Models/MiniCPM4-0.5B \
  --data_file ../datasets/synthetic_text_to_sql/synthetic_text_to_sql_test.snappy.parquet \
  --split test \
  --num_samples 10 \
  --max_new_tokens 128 \
  --device cuda \
  --temperature 0 \
  --sql_parser auto \
  --sql_dialect sqlite
"""

import argparse
import re
import textwrap

from common.constrained_generation import ConstrainedGenerator
from common.cg_utils import load_template, generate_with_generator, append_jsonl_log

from datasets import load_dataset

from common.cli import add_common_gen_args, add_common_model_args
from common.text_model_utils import (
    load_text_model_and_tokenizer,
)


# ============== Prompt 模板 ==============

PROMPT_TEMPLATE = """You are a SQL generation assistant.
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


def build_prompt(
    sql_context: str,
    sql_prompt: str,
    example_sql_context: str,
    example_sql_prompt: str,
    example_sql: str,
) -> str:
    """将 context 和 NL 问题拼成统一 prompt，并加入示例。"""
    return PROMPT_TEMPLATE.format(
        sql_context=(sql_context or "").strip(),
        sql_prompt=sql_prompt.strip(),
        example_sql_context=(example_sql_context or "").strip(),
        example_sql_prompt=example_sql_prompt.strip(),
        example_sql=example_sql.strip(),
    )


def pick_example(ds):
    """取一条示例问题和 SQL。"""
    if len(ds) == 0:
        return ("", "", "")
    s = ds[0]
    return s.get("sql_context") or "", s.get("sql_prompt") or "", s.get("sql") or ""


FINAL_TAG_TOKEN_RE = re.compile(r"</?final>", re.IGNORECASE)
SQL_CODE_FENCE_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
SQL_KEYWORD_RE = re.compile(
    r"\b(select|with|insert|update|delete|create|drop|alter)\b",
    re.IGNORECASE,
)
SQL_START_RE = re.compile(
    r"^\s*(select|with|insert|update|delete|create|drop|alter)\b",
    re.IGNORECASE,
)

try:
    import sqlglot
    from sqlglot import parse_one

    SQLGLOT_AVAILABLE = True
except Exception:
    SQLGLOT_AVAILABLE = False
    parse_one = None
    sqlglot = None


def resolve_sql_parser(sql_parser: str) -> str:
    if sql_parser == "auto":
        return "sqlglot" if SQLGLOT_AVAILABLE else "none"
    if sql_parser == "sqlglot" and not SQLGLOT_AVAILABLE:
        print("Warning: sqlglot not installed, fallback to string normalization.")
        return "none"
    return sql_parser


def extract_sql_from_output(text: str) -> str:
    """Extract SQL using the required <final> and SELECT fallback rules."""
    if text is None:
        return ""
    s = text.strip()
    if not s:
        return ""

    depth = 0
    start = None
    last_span = None
    for match in FINAL_TAG_TOKEN_RE.finditer(s):
        token = match.group(0).lower()
        if token == "<final>":
            if depth == 0:
                start = match.end()
            depth += 1
        elif token == "</final>":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                last_span = (start, match.start())
                start = None

    if last_span:
        return s[last_span[0] : last_span[1]].strip()

    last_fence = None
    for match in SQL_CODE_FENCE_RE.finditer(s):
        last_fence = match
    if last_fence:
        return last_fence.group(1).strip()

    last_keyword = None
    for match in SQL_KEYWORD_RE.finditer(s):
        last_keyword = match
    if last_keyword:
        return s[last_keyword.start() :].strip()

    return ""


def normalize_sql_basic(sql: str) -> str:
    """压缩空白并去掉末尾分号，便于对比。"""
    if sql is None:
        return ""
    cleaned = sql.strip().rstrip(";")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


def normalize_sql(sql: str, sql_parser: str = "auto", sql_dialect: str = None) -> str:
    """可选使用 SQL 解析器做规范化。"""
    cleaned = normalize_sql_basic(sql)
    if not cleaned:
        return ""

    if sql_parser == "auto":
        sql_parser = "sqlglot" if SQLGLOT_AVAILABLE else "none"
    if sql_parser == "sqlglot" and parse_one is None:
        sql_parser = "none"

    if sql_parser == "none":
        return cleaned

    if sql_parser == "sqlglot" and parse_one is not None:
        try:
            expr = parse_one(cleaned, read=sql_dialect) if sql_dialect else parse_one(cleaned)
            expr = expr.normalize()
            canonical = expr.sql(dialect=sql_dialect) if sql_dialect else expr.sql()
            return normalize_sql_basic(canonical)
        except Exception:
            return cleaned

    return cleaned


def answers_match(
    pred_text: str,
    gold_text: str,
    sql_parser: str = "auto",
    sql_dialect: str = None,
) -> bool:
    """比较生成 SQL 与标注 SQL。"""
    if gold_text is None:
        return False
    _, pred_norm, gold_norm = normalize_for_match(
        pred_text,
        gold_text,
        sql_parser=sql_parser,
        sql_dialect=sql_dialect,
    )
    return pred_norm == gold_norm


def normalize_for_match(
    pred_text: str,
    gold_text: str,
    sql_parser: str = "auto",
    sql_dialect: str = None,
):
    """提取并规范化 SQL，供评测输出展示。"""
    pred_sql = extract_sql_from_output(pred_text)
    if gold_text is None:
        return pred_sql, "", ""
    pred_norm = normalize_sql(pred_sql, sql_parser=sql_parser, sql_dialect=sql_dialect)
    gold_norm = normalize_sql(str(gold_text), sql_parser=sql_parser, sql_dialect=sql_dialect)
    return pred_sql, pred_norm, gold_norm


def check_sql_syntax(sql: str, sql_parser: str, sql_dialect: str = None):
    """返回 True/False 表示可解析与否；解析器不可用时返回 None。"""
    if not sql:
        return False
    if sql_parser == "sqlglot" and parse_one is not None:
        try:
            if sql_dialect:
                parse_one(sql, read=sql_dialect)
            else:
                parse_one(sql)
            return True
        except Exception:
            return False
    return SQL_START_RE.match(sql) is not None


# ============== 模型加载相关 ==============
# 已统一使用 common.text_model_utils.load_text_model_and_tokenizer
# ============== 主流程 ==============

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
    sql_parser: str = "auto",
    sql_dialect: str = None,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    mode: str = "prompt",
    constraint_template: str | None = None,
    log_file: str | None = None,
):
    # 1. 加载 synthetic_text_to_sql 数据集（本地 parquet）
    ds_dict = load_dataset(
        "parquet",
        data_files={split: data_file},
    )
    assert split in ds_dict, f"split {split!r} not found in data file; available: {list(ds_dict.keys())}"
    ds = ds_dict[split]

    print(f"Loaded dataset file: {data_file} [{split}]")
    print(f"num_rows = {len(ds)}")

    # 2. 加载模型 & tokenizer
    model, tokenizer, config = load_text_model_and_tokenizer(
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

    # 2.2 加载约束模板（可选）
    constraint_content, constraint_type = load_template(constraint_template)

    # 3. 遍历前 num_samples 个样本
    if num_samples == -1:
        n = len(ds)
    else:
        n = min(num_samples, len(ds))
    resolved_parser = resolve_sql_parser(sql_parser)
    print(f"\n=== Start generation (first {n} samples) ===\n")
    print(f"SQL normalization parser: {resolved_parser}")
    syntax_check_mode = "sqlglot" if resolved_parser == "sqlglot" else "heuristic"
    print(f"SQL syntax check: {syntax_check_mode}")

    example_sql_context, example_sql_prompt, example_sql = pick_example(ds)
    correct = 0
    total_with_gold = 0
    syntax_errors = 0
    semantic_errors = 0

    for idx in range(n):
        sample = ds[idx]
        sql_prompt = sample["sql_prompt"]
        sql_context = sample.get("sql_context") or ""
        gold_sql = sample.get("sql")
        sample_id = sample.get("id", f"idx-{idx}")

        prompt = build_prompt(
            sql_context,
            sql_prompt,
            example_sql_context,
            example_sql_prompt,
            example_sql,
        )

        sample_log = {
            "task": "text2sql",
            "model_name_or_path": model_name_or_path,
            "data_file": data_file,
            "split": split,
            "mode": mode,
            "constraint_template": constraint_template,
            "sql_parser": resolved_parser,
            "sql_dialect": sql_dialect,
            "sample_id": sample_id,
            "sql_prompt": sql_prompt,
            "sql_context": sql_context,
            "gold_sql": gold_sql,
            "prompt_preview": prompt[:300] + "..." if len(prompt) > 300 else prompt,
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

        pred_sql, pred_norm, gold_norm = normalize_for_match(
            output_text,
            gold_sql,
            sql_parser=resolved_parser,
            sql_dialect=sql_dialect,
        )

        print("Model output (raw):")
        print(output_text.strip())
        print("Model output (postprocessed):")
        print(pred_sql if pred_sql else "<empty>")
        syntax_ok = check_sql_syntax(pred_sql, resolved_parser, sql_dialect=sql_dialect)
        if not syntax_ok:
            syntax_errors += 1
            print("[syntax] error")
        else:
            print("[syntax] ok")
        is_correct = False
        if gold_sql is not None:
            total_with_gold += 1
            is_correct = pred_norm == gold_norm
            if not syntax_ok:
                is_correct = False
            if syntax_ok and is_correct:
                correct += 1
            elif syntax_ok:
                semantic_errors += 1
            print(f"[normalized pred] {pred_norm}")
            print(f"[normalized gold] {gold_norm}")
            print(f"[match] {'correct' if is_correct else 'wrong'}")
        if not syntax_ok:
            result_label = "syntax error"
        elif is_correct:
            result_label = "correct"
        else:
            result_label = "semantic error"
        print(f"[result] {result_label}")

        # 写入逐样本日志（jsonl）
        if log_file:
            sample_log["raw_output"] = output_text
            sample_log["pred_sql"] = pred_sql
            sample_log["pred_norm"] = pred_norm
            sample_log["gold_norm"] = gold_norm
            sample_log["syntax_ok"] = bool(syntax_ok)
            if gold_sql is not None:
                sample_log["is_correct"] = bool(is_correct)
                sample_log["result_label"] = result_label
            append_jsonl_log(log_file, sample_log, verbose=False)

        print("\n")

    if total_with_gold:
        acc = correct / total_with_gold
        print("=" * 80)
        print(f"Accuracy: {correct}/{total_with_gold} = {acc:.2%}")
        print(f"Syntax errors: {syntax_errors}")
        print(f"Semantic errors (syntax ok but not equivalent): {semantic_errors}")

        if log_file:
            summary_log = {
                "task": "text2sql_summary",
                "model_name_or_path": model_name_or_path,
                "data_file": data_file,
                "split": split,
                "mode": mode,
                "constraint_template": constraint_template,
                "sql_parser": resolved_parser,
                "sql_dialect": sql_dialect,
                "num_samples": n,
                "total_with_gold": total_with_gold,
                "correct": correct,
                "accuracy": f"{acc:.4f}",
                "syntax_errors": syntax_errors,
                "semantic_errors": semantic_errors,
            }
            append_jsonl_log(log_file, summary_log, verbose=False)

        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SQL generation on synthetic_text_to_sql with a transformers model.",
    )
    parser = add_common_model_args(parser)
    parser = add_common_gen_args(parser)
    parser.add_argument(
        "--data_file",
        type=str,
        default="datasets/synthetic_text_to_sql/synthetic_text_to_sql_test.snappy.parquet",
        help="synthetic_text_to_sql data file (parquet).",
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
        "--sql_parser",
        type=str,
        default="auto",
        choices=["auto", "none", "sqlglot"],
        help="SQL normalization: auto=prefer sqlglot if available; none=string normalization only.",
    )
    parser.add_argument(
        "--sql_dialect",
        type=str,
        default=None,
        help="Optional sqlglot dialect, e.g., 'sqlite'/'postgres'.",
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
        sql_parser=args.sql_parser,
        sql_dialect=args.sql_dialect,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        mode=args.mode,
        constraint_template=args.constraint_template,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
