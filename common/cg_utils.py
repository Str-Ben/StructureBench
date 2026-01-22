from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch

from .constrained_generation import ConstrainedGenerator
from .constrained_generation import GenerationMode


def load_template(constraint_template: Optional[str]) -> Tuple[Optional[Any], Optional[str]]:
    """加载约束模板。

    - 如果是文件路径，则根据扩展名加载内容。
    - 如果以 'guidance:' 开头，则将其解析为 guidance 模板名。
    """
    if not constraint_template:
        return None, None

    # 检查是否为 guidance 模板名
    if constraint_template.startswith("guidance:"):
        template_name = constraint_template.split(":", 1)[1]
        print(f"Using guidance template name: {template_name}")
        return template_name, "guidance"

    # 否则，按文件路径处理
    if os.path.exists(constraint_template):
        print(f"Loading constraint template from file: {constraint_template}")
        _, ext = os.path.splitext(constraint_template)
        ext = ext.lower()

        if ext == ".json":
            with open(constraint_template, "r", encoding="utf-8") as f:
                return json.load(f), "json"
        elif ext == ".ebnf":
            with open(constraint_template, "r", encoding="utf-8") as f:
                return f.read(), "ebnf"
        elif ext == ".cfg":
            with open(constraint_template, "r", encoding="utf-8") as f:
                return f.read(), "cfg"
        elif ext == ".regex":
            with open(constraint_template, "r", encoding="utf-8") as f:
                return f.read(), "regex"
    
    print(f"Warning: constraint_template path not found or format not recognized: '{constraint_template}'")
    return None, None


def _normalize_model_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """尽量把不同 processor 产物规范化为可用于 **inputs 传入 generate 的 dict。

    参考 common/vl_model_utils.py 的处理方式，但这里不主动 move 到 device/dtype，
    让下游 generator/model 自行处理（避免和 device_map/accelerate 冲突）。
    """
    if inputs is None:
        return {}

    # 有些 processor 返回 BatchFeature，携带 .data
    if hasattr(inputs, "data"):
        inputs = inputs.data  # type: ignore[assignment]

    if not isinstance(inputs, dict):
        raise TypeError(f"inputs must be a dict-like object, got: {type(inputs)}")

    normalized: Dict[str, Any] = {}
    for k, v in inputs.items():
        # Collapse nested single-item lists
        while isinstance(v, list) and len(v) == 1:
            v = v[0]

        # If still a list of tensors, stack them
        if isinstance(v, list) and v and all(hasattr(x, "shape") for x in v):
            try:
                v = torch.stack(v)  # type: ignore[arg-type]
            except Exception:
                # stack 失败就原样保留
                pass

        normalized[k] = v

    return normalized


def generate_with_generator(
    generator: ConstrainedGenerator,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    prompt: Optional[str] = None,
    inputs: Optional[Dict[str, Any]] = None,
    constraint: Optional[Any] = None,
    constraint_type: Optional[str] = None,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
) -> str:
    """统一的生成入口：支持纯文本 prompt 或 VLM inputs（二选一）。

    - 文本：传 prompt=...
    - 视觉语言：传 inputs={input_ids, pixel_values, ...}

    注意：这里仅做“接口层”支持，是否能真正约束生成取决于 ConstrainedGenerator
    对 inputs 形式的支持情况。
    """
    try:
        if (prompt is None) == (inputs is None):
            raise ValueError("generate_with_generator expects exactly one of `prompt` or `inputs`.")

        kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "frequency_penalty": frequency_penalty,
        }

        if prompt is not None:
            kwargs["prompt"] = prompt
        else:
            kwargs.update(_normalize_model_inputs(inputs or {}))

        if constraint is not None and constraint_type is not None:
            if constraint_type == "json":
                kwargs["json_schema"] = constraint
            elif constraint_type == "ebnf":
                kwargs["ebnf"] = constraint
            elif constraint_type == "cfg":
                kwargs["cfg"] = constraint
            elif constraint_type == "regex":
                kwargs["regex"] = constraint
            elif constraint_type == "guidance":
                kwargs["guidance_template"] = constraint
            else:
                # 默认或未知的约束类型
                kwargs["json_schema"] = constraint

        if generator.mode == GenerationMode.GUIDANCE:
            kwargs["python_cfg"] = True

        return generator.generate(**kwargs)
    except Exception as e:
        print(f"生成失败: {e}")
        import traceback

        traceback.print_exc()
        return ""


def append_jsonl_log(log_file: Optional[str], log_entry: dict, verbose: bool = True) -> bool:
    if not log_file:
        return False

    try:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        if verbose:
            print(f"结果已记录到: {log_file}")
        return True
    except Exception as e:
        print(f"写入日志文件失败: {e}")
        return False
