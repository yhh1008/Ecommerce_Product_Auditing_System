#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import pandas as pd

from ecommerce_audit.metrics import (
    attribute_micro_f1,
    category_accuracy,
    hallucination_rate,
    json_success_rate,
    violation_accuracy,
)
from ecommerce_audit.schema import is_valid_audit_result, parse_audit_result


def load_json_col(df: pd.DataFrame, col: str) -> list[str]:
    return df[col].astype(str).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="parquet with columns: prediction, allowed_attributes(optional)")
    parser.add_argument("--gold", required=True, help="parquet with columns: gold_json")
    args = parser.parse_args()

    pred_df = pd.read_parquet(args.pred)
    gold_df = pd.read_parquet(args.gold)

    pred_texts = load_json_col(pred_df, "prediction")
    gold_texts = load_json_col(gold_df, "gold_json")

    valid_pairs = [
        (p, g)
        for p, g in zip(pred_texts, gold_texts, strict=False)
        if is_valid_audit_result(p) and is_valid_audit_result(g)
    ]
    preds = [parse_audit_result(p) for p, _ in valid_pairs]
    gts = [parse_audit_result(g) for _, g in valid_pairs]

    supports = []
    if "allowed_attributes" in pred_df.columns:
        for x in pred_df["allowed_attributes"].tolist()[: len(preds)]:
            if isinstance(x, str):
                supports.append({"allowed_attributes": json.loads(x)})
            elif isinstance(x, dict):
                supports.append({"allowed_attributes": x})
            else:
                supports.append({"allowed_attributes": {}})
    else:
        supports = [{"allowed_attributes": {}} for _ in preds]

    out = {
        "json_success_rate": json_success_rate(pred_texts),
        "category_accuracy": category_accuracy(preds, gts) if preds else 0.0,
        "violation_accuracy": violation_accuracy(preds, gts) if preds else 0.0,
        "attribute_micro_f1": attribute_micro_f1(preds, gts) if preds else 0.0,
        "hallucination_rate": hallucination_rate(preds, supports) if preds else 0.0,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
