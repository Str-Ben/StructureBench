#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
humaneval_generate.py

在 openai_humaneval 数据集上调用任意 transformers 模型进行代码生成：
- 从本地 parquet 文件加载 HumanEval 题目
- 为每条样本构造 prompt，要求补全 Python 函数

用法示例（在 datacode 目录下执行）：

python -m data.humaneval_generate \
  --model_name_or_path ../Models/MiniCPM4-0.5B \
  --data_file ../datasets/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet \
  --split test \
  --num_samples 160 \
  --max_new_tokens 512 \
  --device cuda \
  --temperature 0 | tee pl_cpm05b.txt
"""

import argparse
import re
import signal
import textwrap
import traceback

from datasets import load_dataset

from common.cli import add_common_gen_args, add_common_model_args
from common.text_model_utils import load_text_model_and_tokenizer
from common.constrained_generation import ConstrainedGenerator
from common.cg_utils import load_template, generate_with_generator, append_jsonl_log


# ============== Prompt 模板 ==============

PROMPT_TEMPLATE = """You are a Python coding assistant.
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


def build_prompt(task: str, example_task: str, example_code: str) -> str:
    """把 HumanEval 的 prompt 嵌入统一提示，并加入示例。"""
    return PROMPT_TEMPLATE.format(
        example_task=example_task.strip(),
        example_code=example_code.strip(),
        task=task.strip(),
    )


def pick_example(ds):
    """取一条示例 HumanEval 题目和参考解。"""
    if len(ds) == 0:
        return ("", "")
    s = ds[0]
    return s.get("prompt", ""), s.get("canonical_solution", "")


def cleanup_generated_code(code: str) -> str:
    """简单清洗模型输出，去掉 Markdown 代码块等包装。"""
    if code is None:
        return ""
    text = str(code)
    stripped = text.lstrip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.rstrip()


def strip_think_content(text: str) -> str:
    """移除模型可能输出的 <think>...</think> 推理内容。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def extract_first_code_block(text: str) -> str:
    """从任意文本中提取第一个 ``` ``` 代码块内容。"""
    if not text or "```" not in text:
        return ""
    # 优先用正则提取匹配到的首个完整代码块
    match = re.search(r"```(?:\s*[\w.-]+)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    # 若缺少闭合 ```，退化为取第一个 ``` 之后的所有内容
    after = text.split("```", 1)[1]
    block_lines = after.splitlines()
    if block_lines:
        first_line = block_lines[0].strip()
        looks_like_lang = (
            len(block_lines) > 1
            and first_line
            and len(first_line.split()) == 1
            and not first_line.startswith(("def ", "class ", "from ", "import "))
        )
        if looks_like_lang:
            block_lines = block_lines[1:]
    return "\n".join(block_lines).strip()


def postprocess_generated_code(raw_output: str) -> str:
    """
    若输出本身就是代码，直接返回；若带前缀文本，则尝试提取 Markdown 代码块。
    """
    if raw_output is None:
        return ""
    stripped = strip_think_content(str(raw_output)).strip()
    if "```" not in stripped:
        return stripped
    if stripped.startswith("```"):
        return cleanup_generated_code(stripped)
    block = extract_first_code_block(stripped)
    return block or cleanup_generated_code(stripped)


def build_executable_script(task_prompt: str, generated_code: str, test_code: str, entry_point: str) -> str:
    """拼接 Prompt + 生成代码 + 测试用例 + 入口检查，得到可执行脚本。"""
    entry_literal = repr(entry_point)
    driver = textwrap.dedent(
        f"""
        if 'check' not in globals():
            raise RuntimeError('check function not defined')
        _entry_point = {entry_literal}
        if _entry_point not in globals():
            raise NameError(f"entry point {{_entry_point!r}} not found in generated code")
        check(globals()[_entry_point])
        """
    ).strip()
    parts = [
        str(task_prompt or "").rstrip(),
        cleanup_generated_code(generated_code),
        str(test_code or "").rstrip(),
        driver,
    ]
    return "\n\n".join(part for part in parts if part)


def exec_with_timeout(script: str, timeout_sec: float):
    """执行脚本并带超时控制，返回 (状态字符串, 错误信息)。"""
    timeout = timeout_sec if timeout_sec and timeout_sec > 0 else 1.0

    def _handler(signum, frame):
        raise TimeoutError(f"Execution exceeded {timeout} seconds")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    exec_globals = {"__builtins__": __builtins__, "__name__": "__main__"}
    try:
        exec(script, exec_globals, exec_globals)
        return "passed", "passed"
    except TimeoutError as exc:
        return "runtime_error", str(exc)
    except (SyntaxError, IndentationError):
        return "compile_error", traceback.format_exc()
    except Exception:
        return "runtime_error", traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


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
    eval_timeout: float = 5.0,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    mode: str = "prompt",
    constraint_template: str | None = None,
    log_file: str | None = None,
):
    # 1. 加载数据集
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
    constraint_content, constraint_type = load_template(constraint_template)

    # 3. 遍历前 num_samples 个样本
    n = min(num_samples, len(ds))
    print(f"\n=== Start generation (first {n} samples) ===\n")

    example_task, example_code = pick_example(ds)
    correct = 0
    compile_errors = 0
    runtime_errors = 0
    evaluated = 0

    for idx in range(n):
        sample = ds[idx]
        task = sample["prompt"]
        test_code = sample.get("test")
        entry_point = sample.get("entry_point")
        sample_id = sample.get("task_id", f"idx-{idx}")

        prompt = build_prompt(task, example_task, example_code)

        sample_log = {
            "task": "humaneval",
            "model_name_or_path": model_name_or_path,
            "data_file": data_file,
            "split": split,
            "mode": mode,
            "constraint_template": constraint_template,
            "sample_id": sample_id,
            "entry_point": entry_point,
            "eval_timeout": eval_timeout,
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

        print("Model output:")
        cleaned_output = postprocess_generated_code(output_text)
        print(cleaned_output)

        sample_log["raw_output"] = output_text
        sample_log["post_processed_output"] = cleaned_output

        if test_code and entry_point:
            evaluated += 1
            script = build_executable_script(task, cleaned_output, test_code, entry_point)
            status, err_msg = exec_with_timeout(script, eval_timeout)
            sample_log["evaluated"] = True
            sample_log["test_status"] = status
            sample_log["test_error"] = err_msg if status != "passed" else None

            if status == "passed":
                correct += 1
            elif status == "compile_error":
                compile_errors += 1
            else:
                runtime_errors += 1
            print(f"[test result] {status}")
            if status != "passed":
                print(err_msg)
        else:
            sample_log["evaluated"] = False
            sample_log["test_status"] = "skipped"
            sample_log["test_error"] = "missing test/entry_point"
            print("[test result] missing test/entry_point; skip dynamic validation.")

        if log_file:
            append_jsonl_log(log_file, sample_log, verbose=False)

        print("\n")

    if evaluated:
        acc = correct / evaluated
        print("=" * 80)
        print(f"AC: {correct}")
        print(f"CE: {compile_errors}")
        print(f"WA: {runtime_errors}")
        print(f"Accuracy: {correct}/{evaluated} = {acc:.2%} (by executing tests)")

        if log_file:
            summary_log = {
                "task": "humaneval_summary",
                "model_name_or_path": model_name_or_path,
                "data_file": data_file,
                "split": split,
                "mode": mode,
                "constraint_template": constraint_template,
                "num_samples": n,
                "evaluated": evaluated,
                "correct": correct,
                "accuracy": f"{acc:.4f}",
                "compile_errors": compile_errors,
                "runtime_errors": runtime_errors,
                "eval_timeout": eval_timeout,
            }
            append_jsonl_log(log_file, summary_log, verbose=False)

        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run code generation on openai_humaneval with a transformers model.",
    )
    parser = add_common_model_args(parser)
    parser = add_common_gen_args(parser)
    parser.add_argument(
        "--data_file",
        type=str,
        default="datasets/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet",
        help="HumanEval data file (parquet).",
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
        default=3,
        help="Number of samples to generate from the dataset.",
    )
    parser.add_argument(
        "--eval_timeout",
        type=float,
        default=5.0,
        help="Per-sample test execution timeout (seconds) to prevent infinite loops.",
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
        eval_timeout=args.eval_timeout,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        mode=args.mode,
        constraint_template=args.constraint_template,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
