#!/usr/bin/env bash
set -xeuo pipefail

MODEL_ID=${MODEL_ID:-$PWD/model/Qwen3-VL-2B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-$PWD/data/processed/pref/train.parquet}
VAL_FILE=${VAL_FILE:-$PWD/data/processed/pref/val.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/outputs/rm/qwen3_vl_rm}

python -m ecommerce_audit.reward.train_rm_stub \
  --model-path "$MODEL_ID" \
  --train-file "$TRAIN_FILE" \
  --val-file "$VAL_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 2 \
  --batch-size 16 \
  --lr 1e-5
