from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from ecommerce_audit.postprocess import parse_or_fix_audit_result
from ecommerce_audit.prompting import PromptBuilder

MODEL_NAME = os.getenv("AUDIT_MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct")
DEVICE = os.getenv("AUDIT_DEVICE", "cuda")


class AuditRequest(BaseModel):
    image: str
    text: str | None = None
    image_is_url: bool = False
    return_raw_text: bool = False


class AuditResponse(BaseModel):
    result: dict[str, Any]
    raw_text: str | None = None


model = None
processor = None
prompt_builder = PromptBuilder()


async def _load_image(req: AuditRequest) -> Image.Image:
    if req.image_is_url:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(req.image)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
    try:
        data = base64.b64decode(req.image)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image payload: {exc}") from exc


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME).to(DEVICE)
    yield


app = FastAPI(title="Ecommerce Audit Service", lifespan=lifespan)


@app.post("/audit", response_model=AuditResponse)
async def audit(req: AuditRequest) -> AuditResponse:
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="model not ready")

    image = await _load_image(req)
    prompt = prompt_builder.build_infer_prompt(req.text)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(**inputs, max_new_tokens=256)
    raw_text = processor.decode(output[0], skip_special_tokens=True)

    try:
        parsed = parse_or_fix_audit_result(raw_text)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid model output: {exc}") from exc

    return AuditResponse(
        result=parsed.model_dump(),
        raw_text=raw_text if req.return_raw_text else None,
    )
