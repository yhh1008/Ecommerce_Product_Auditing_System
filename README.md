# Ecommerce Product Auditing System

A multimodal e-commerce product auditing pipeline on Qwen3-VL with VeRL.

## Features
- Structured JSON output for category, attributes, and compliance.
- Data pipeline for SFT and preference data.
- SFT / RM / GRPO training scripts for VeRL.
- Offline evaluation metrics and online FastAPI inference service.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data pipeline
```bash
python scripts/prepare_sft_data.py
python scripts/prepare_pref_data.py
python scripts/validate_json_schema.py
```

## Training
```bash
bash scripts/train_sft.sh
bash scripts/train_rm.sh
bash scripts/train_grpo.sh
```

## Evaluation and inference
```bash
python scripts/eval_offline.py --pred data/processed/eval_pred.parquet --gold data/processed/eval_gold.parquet
python scripts/run_infer_demo.py --image path/to/test.jpg --text "请审核该商品并输出JSON"
uvicorn ecommerce_audit.service:app --host 0.0.0.0 --port 8000
```
