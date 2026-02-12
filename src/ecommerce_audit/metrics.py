from __future__ import annotations

from collections import Counter
from typing import Any

from sklearn.metrics import accuracy_score, f1_score

from ecommerce_audit.schema import AuditResult, is_valid_audit_result, parse_audit_result


def json_success_rate(pred_texts: list[str]) -> float:
    if not pred_texts:
        return 0.0
    ok = sum(1 for t in pred_texts if is_valid_audit_result(t))
    return ok / len(pred_texts)


def violation_accuracy(preds: list[AuditResult], gts: list[AuditResult]) -> float:
    y_pred = [p.violation for p in preds]
    y_true = [g.violation for g in gts]
    return float(accuracy_score(y_true, y_pred))


def category_accuracy(preds: list[AuditResult], gts: list[AuditResult]) -> float:
    y_pred = [p.category for p in preds]
    y_true = [g.category for g in gts]
    return float(accuracy_score(y_true, y_pred))


def attribute_micro_f1(preds: list[AuditResult], gts: list[AuditResult]) -> float:
    y_true: list[int] = []
    y_pred: list[int] = []
    for pred, gt in zip(preds, gts, strict=True):
        keys = set(pred.attributes.keys()) | set(gt.attributes.keys())
        for k in keys:
            y_true.append(1)
            y_pred.append(int(pred.attributes.get(k) == gt.attributes.get(k)))
    if not y_true:
        return 0.0
    return float(f1_score(y_true, y_pred, average="micro"))


def hallucination_rate(preds: list[AuditResult], supports: list[dict[str, Any]]) -> float:
    """Support format: {'allowed_attributes': {'color': ['black', ...], ...}}."""
    if not preds:
        return 0.0
    hallucinated = 0
    for pred, support in zip(preds, supports, strict=True):
        allowed = support.get("allowed_attributes", {})
        sample_hallucinated = False
        for k, v in pred.attributes.items():
            if k not in allowed:
                sample_hallucinated = True
                break
            allowed_values = allowed[k]
            if allowed_values and v not in allowed_values:
                sample_hallucinated = True
                break
        hallucinated += int(sample_hallucinated)
    return hallucinated / len(preds)


def evaluate_from_json_text(pred_texts: list[str], gt_texts: list[str]) -> dict[str, float]:
    parsed_preds = [parse_audit_result(t) for t in pred_texts if is_valid_audit_result(t)]
    parsed_gts = [parse_audit_result(t) for t in gt_texts[: len(parsed_preds)]]
    return {
        "json_success_rate": json_success_rate(pred_texts),
        "category_accuracy": category_accuracy(parsed_preds, parsed_gts) if parsed_preds else 0.0,
        "violation_accuracy": violation_accuracy(parsed_preds, parsed_gts) if parsed_preds else 0.0,
        "attribute_micro_f1": attribute_micro_f1(parsed_preds, parsed_gts) if parsed_preds else 0.0,
    }


def violation_type_distribution(results: list[AuditResult]) -> dict[str, int]:
    cnt = Counter()
    for item in results:
        for v in item.violation_types:
            cnt[v] += 1
    return dict(cnt)
