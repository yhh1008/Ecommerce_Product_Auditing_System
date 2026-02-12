from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import optim
from transformers import AutoModelForCausalLM, AutoTokenizer

from ecommerce_audit.reward.reward_model import RewardModel, pairwise_loss


def _tokenize(tokenizer, prompts, responses, max_length):
    merged = [f"{p}\n{r}" for p, r in zip(prompts, responses, strict=False)]
    return tokenizer(
        merged,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def run(args: argparse.Namespace) -> None:
    train_df = pd.read_parquet(args.train_file)
    val_df = pd.read_parquet(args.val_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    backbone = AutoModelForCausalLM.from_pretrained(args.model_path)
    model = RewardModel(backbone).to(args.device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for start in range(0, len(train_df), args.batch_size):
            batch = train_df.iloc[start : start + args.batch_size]
            chosen_inputs = _tokenize(
                tokenizer,
                batch["prompt"].tolist(),
                batch["chosen"].tolist(),
                args.max_length,
            )
            rejected_inputs = _tokenize(
                tokenizer,
                batch["prompt"].tolist(),
                batch["rejected"].tolist(),
                args.max_length,
            )

            chosen_inputs = {k: v.to(args.device) for k, v in chosen_inputs.items()}
            rejected_inputs = {k: v.to(args.device) for k, v in rejected_inputs.items()}

            r_chosen = model(**chosen_inputs)
            r_rejected = model(**rejected_inputs)
            loss = pairwise_loss(r_chosen, r_rejected)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        print(f"epoch={epoch} train_loss={epoch_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(val_df), args.batch_size):
            batch = val_df.iloc[start : start + args.batch_size]
            chosen_inputs = _tokenize(
                tokenizer,
                batch["prompt"].tolist(),
                batch["chosen"].tolist(),
                args.max_length,
            )
            rejected_inputs = _tokenize(
                tokenizer,
                batch["prompt"].tolist(),
                batch["rejected"].tolist(),
                args.max_length,
            )
            chosen_inputs = {k: v.to(args.device) for k, v in chosen_inputs.items()}
            rejected_inputs = {k: v.to(args.device) for k, v in rejected_inputs.items()}

            r_chosen = model(**chosen_inputs)
            r_rejected = model(**rejected_inputs)
            correct += int((r_chosen > r_rejected).sum().item())
            total += len(batch)

    pairwise_acc = correct / max(total, 1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "reward_model.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps({"pairwise_accuracy": pairwise_acc}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"pairwise_accuracy={pairwise_acc:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
