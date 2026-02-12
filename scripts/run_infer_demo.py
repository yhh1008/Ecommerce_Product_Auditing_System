#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from ecommerce_audit.postprocess import parse_or_fix_audit_result
from ecommerce_audit.prompting import PromptBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", default="请审核该商品并输出 JSON")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForVision2Seq.from_pretrained(args.model).to(args.device)
    pb = PromptBuilder()

    image = Image.open(args.image).convert("RGB")
    prompt = pb.build_infer_prompt(args.text)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(args.device)
    output = model.generate(**inputs, max_new_tokens=256)
    raw = processor.decode(output[0], skip_special_tokens=True)

    print("RAW OUTPUT:")
    print(raw)

    parsed = parse_or_fix_audit_result(raw)
    print("\nPARSED JSON:")
    print(json.dumps(parsed.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
