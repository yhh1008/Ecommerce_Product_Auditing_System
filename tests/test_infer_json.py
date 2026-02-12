from ecommerce_audit.postprocess import parse_or_fix_audit_result


def test_parse_or_fix():
    raw = '结果如下：{"category":"shoes","attributes":{"color":"black"},"violation":false,"reason":"图文一致","violation_types":[],}'
    parsed = parse_or_fix_audit_result(raw)
    assert parsed.category == "shoes"
