from __future__ import annotations

from ecommerce_audit.schema import is_valid_audit_result, parse_audit_result


def format_reward(response_text: str) -> float:
    return 1.0 if is_valid_audit_result(response_text) else -1.0


def attribute_consistency_reward(response_text: str, reference_text: str) -> float:
    if not is_valid_audit_result(response_text) or not is_valid_audit_result(reference_text):
        return -1.0
    pred = parse_audit_result(response_text)
    ref = parse_audit_result(reference_text)
    keys = set(pred.attributes.keys()) | set(ref.attributes.keys())
    if not keys:
        return 0.0
    match = sum(1 for k in keys if pred.attributes.get(k) == ref.attributes.get(k))
    return (2.0 * match / len(keys)) - 1.0


def compliance_reward(response_text: str, reference_text: str) -> float:
    if not is_valid_audit_result(response_text) or not is_valid_audit_result(reference_text):
        return -1.0
    pred = parse_audit_result(response_text)
    ref = parse_audit_result(reference_text)
    reward = 0.0
    reward += 0.5 if pred.violation == ref.violation else -0.5
    reward += 0.5 if set(pred.violation_types) == set(ref.violation_types) else -0.5
    return reward


def length_penalty(response_text: str, max_chars: int = 800) -> float:
    return max(0.0, (len(response_text) - max_chars) / max_chars)


def total_reward(
    response_text: str,
    reference_text: str,
    rm_score: float,
    kl_divergence: float,
    beta: float = 0.02,
) -> float:
    fmt = format_reward(response_text)
    attr = attribute_consistency_reward(response_text, reference_text)
    comp = compliance_reward(response_text, reference_text)
    lpen = length_penalty(response_text)
    format_penalty = 0.0 if fmt > 0 else 1.0
    return rm_score + 0.4 * attr + 0.4 * comp - beta * kl_divergence - lpen - format_penalty
