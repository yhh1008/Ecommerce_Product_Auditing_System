#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from ecommerce_audit.prompting import PromptBuilder
from ecommerce_audit.schema import AuditResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_annotations(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        raise ValueError("JSON root must be a list")
    raise ValueError(f"unsupported annotation file: {path}")


def build_sft_sample(ann: dict, prompt_builder: PromptBuilder) -> dict:
    result = AuditResult(
        category=ann["category"],
        attributes=ann["attributes"],
        violation=ann["violation"],
        reason=ann["reason"],
        violation_types=ann.get("violation_types", []),
    )
    prompt = prompt_builder.build_train_prompt(ann.get("text"))
    return {
        "image": ann["image"],
        "prompt": prompt,
        "response": result.model_dump_json(ensure_ascii=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/annotations.jsonl")
    parser.add_argument("--train-output", default="data/processed/sft/train.parquet")
    parser.add_argument("--val-output", default="data/processed/sft/val.parquet")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    anns = load_annotations(Path(args.input))
    pb = PromptBuilder()

    good = []
    bad_count = 0
    for ann in anns:
        try:
            good.append(build_sft_sample(ann, pb))
        except Exception:
            bad_count += 1

    df = pd.DataFrame(good)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_val = int(len(df) * args.val_ratio)
    val_df = df.iloc[:n_val]
    train_df = df.iloc[n_val:]

    Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_output).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(args.train_output, index=False)
    val_df.to_parquet(args.val_output, index=False)

    logger.info("built SFT data: train=%d val=%d dropped=%d", len(train_df), len(val_df), bad_count)


if __name__ == "__main__":
    main()
