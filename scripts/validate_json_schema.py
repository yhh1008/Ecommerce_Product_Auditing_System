#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ecommerce_audit.schema import is_valid_audit_result


def check_file(path: Path, columns: list[str]) -> dict[str, float]:
    df = pd.read_parquet(path)
    out = {}
    for col in columns:
        if col not in df.columns:
            out[col] = 0.0
            continue
        ok = sum(1 for x in df[col].astype(str).tolist() if is_valid_audit_result(x))
        out[col] = ok / max(len(df), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-train", default="data/processed/sft/train.parquet")
    parser.add_argument("--sft-val", default="data/processed/sft/val.parquet")
    parser.add_argument("--pref-train", default="data/processed/pref/train.parquet")
    parser.add_argument("--pref-val", default="data/processed/pref/val.parquet")
    args = parser.parse_args()

    checks = [
        (Path(args.sft_train), ["response"]),
        (Path(args.sft_val), ["response"]),
        (Path(args.pref_train), ["chosen", "rejected"]),
        (Path(args.pref_val), ["chosen", "rejected"]),
    ]

    for path, cols in checks:
        if not path.exists():
            print(f"[WARN] missing: {path}")
            continue
        result = check_file(path, cols)
        print(f"{path}: {result}")


if __name__ == "__main__":
    main()
