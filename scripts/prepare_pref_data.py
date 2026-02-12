#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ecommerce_audit.schema import parse_audit_result


def corrupt_response(resp: dict) -> dict:
    out = json.loads(json.dumps(resp, ensure_ascii=False))
    if "attributes" in out and isinstance(out["attributes"], dict) and out["attributes"]:
        first_key = next(iter(out["attributes"]))
        out["attributes"][first_key] = "__wrong_value__"
    else:
        out.setdefault("attributes", {})["hallucinated_attribute"] = "fake"
    return out


def malformed_json(resp_str: str) -> str:
    return resp_str[:-1] if resp_str.endswith("}") else resp_str + "}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-train", default="data/processed/sft/train.parquet")
    parser.add_argument("--sft-val", default="data/processed/sft/val.parquet")
    parser.add_argument("--train-output", default="data/processed/pref/train.parquet")
    parser.add_argument("--val-output", default="data/processed/pref/val.parquet")
    args = parser.parse_args()

    def _convert(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in df.iterrows():
            chosen = r["response"]
            obj = parse_audit_result(chosen).model_dump()
            bad = corrupt_response(obj)
            rejected = json.dumps(bad, ensure_ascii=False)
            if len(rows) % 3 == 2:
                rejected = malformed_json(rejected)
            rows.append(
                {
                    "image": r["image"],
                    "prompt": r["prompt"],
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
        return pd.DataFrame(rows)

    train_df = pd.read_parquet(args.sft_train)
    val_df = pd.read_parquet(args.sft_val)

    pref_train = _convert(train_df)
    pref_val = _convert(val_df)

    Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_output).parent.mkdir(parents=True, exist_ok=True)
    pref_train.to_parquet(args.train_output, index=False)
    pref_val.to_parquet(args.val_output, index=False)

    print(f"pref data built: train={len(pref_train)} val={len(pref_val)}")


if __name__ == "__main__":
    main()
