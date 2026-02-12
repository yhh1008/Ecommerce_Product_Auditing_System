from __future__ import annotations

import json
import re
from typing import Any

from ecommerce_audit.schema import AuditResult


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", flags=re.DOTALL)


def extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = JSON_BLOCK_PATTERN.search(text)
    if not match:
        raise ValueError("未找到 JSON 对象")
    return match.group(0)


def try_fix_json_once(raw_text: str) -> str:
    candidate = extract_json_object(raw_text)
    candidate = candidate.replace("\n", " ").replace("\t", " ")
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    json.loads(candidate)
    return candidate


def parse_or_fix_audit_result(raw_text: str) -> AuditResult:
    try:
        return AuditResult.model_validate_json(raw_text)
    except Exception:
        fixed = try_fix_json_once(raw_text)
        return AuditResult.model_validate_json(fixed)


def dump_result(result: AuditResult) -> dict[str, Any]:
    return result.model_dump()
