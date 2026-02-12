from ecommerce_audit.schema import AuditResult, is_valid_audit_result


def test_schema_valid():
    raw = '{"category":"shoes","attributes":{"color":"black"},"violation":false,"reason":"图文一致","violation_types":[]}'
    assert is_valid_audit_result(raw)
    parsed = AuditResult.model_validate_json(raw)
    assert parsed.category == "shoes"


def test_schema_invalid_extra_key():
    raw = '{"category":"shoes","attributes":{},"violation":false,"reason":"ok","violation_types":[],"extra":1}'
    assert not is_valid_audit_result(raw)
