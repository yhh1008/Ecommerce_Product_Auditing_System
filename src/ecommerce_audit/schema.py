from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Category = Literal["apparel", "shoes", "bags", "beauty", "home"]
AttributeValue = str | bool | int | float


class AuditResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category
    attributes: dict[str, AttributeValue]
    violation: bool
    reason: str
    violation_types: list[str] = Field(default_factory=list)

    @field_validator("reason")
    @classmethod
    def reason_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reason 不能为空")
        return v


def parse_audit_result(raw: str | dict[str, Any]) -> AuditResult:
    if isinstance(raw, dict):
        return AuditResult.model_validate(raw)
    return AuditResult.model_validate_json(raw)


def is_valid_audit_result(raw: str | dict[str, Any]) -> bool:
    try:
        parse_audit_result(raw)
        return True
    except (ValidationError, json.JSONDecodeError, TypeError, ValueError):
        return False


def to_json(result: AuditResult) -> str:
    return result.model_dump_json(ensure_ascii=False)
