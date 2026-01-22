import json
import re
from typing import Any, Dict, List, Optional

_TOOLCALL_RE = re.compile(r"<toolcall>(.*?)</toolcall>", re.IGNORECASE | re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _strip_toolcall(text: str) -> str:
    match = _TOOLCALL_RE.search(text)
    if match:
        return match.group(1).strip()
    return text


def _strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.replace("```", "")


def _find_matching_bracket(text: str, start: int) -> Optional[int]:
    stack = []
    in_single = False
    in_double = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_single:
            if ch == "'":
                in_single = False
            continue
        if in_double:
            if ch == '"':
                in_double = False
            continue
        if ch == "'":
            in_single = True
            continue
        if ch == '"':
            in_double = True
            continue
        if ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if not stack:
                return None
            expected = "}" if stack[-1] == "{" else "]"
            if ch != expected:
                return None
            stack.pop()
            if not stack:
                return idx
    return None


def _extract_json_segment(text: str) -> str:
    for idx, ch in enumerate(text):
        if ch not in "[{":
            continue
        end = _find_matching_bracket(text, idx)
        if end is not None:
            return text[idx : end + 1]
    return text


def _load_json(text: str) -> Any:
    cleaned = _strip_code_fences(_strip_toolcall(text.strip()))
    if not cleaned:
        raise ValueError("Empty JSON input.")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        segment = _extract_json_segment(cleaned)
        if segment and segment != cleaned:
            try:
                return json.loads(segment)
            except json.JSONDecodeError as inner_exc:
                raise ValueError(f"Invalid JSON: {inner_exc}") from inner_exc
        raise ValueError(f"Invalid JSON: {exc}") from exc


def _normalize_call(entry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if "function" in entry:
        function_name = entry.get("function")
        if not function_name:
            raise ValueError("Missing function name in JSON output.")
        parameters = entry.get("parameters", {})
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise ValueError("JSON parameters must be an object.")
        return {str(function_name): parameters}

    if len(entry) == 1:
        function_name, parameters = next(iter(entry.items()))
        if not isinstance(parameters, dict):
            raise ValueError("JSON parameters must be an object.")
        return {str(function_name): parameters}

    raise ValueError("Invalid JSON function call object.")


def parse_json_function_call(source_code: str) -> List[Dict[str, Dict[str, Any]]]:
    payload = _load_json(source_code)
    if isinstance(payload, list):
        return [_normalize_call(entry) for entry in payload]
    if isinstance(payload, dict):
        return [_normalize_call(payload)]
    raise ValueError("JSON output must be an object or list of objects.")
