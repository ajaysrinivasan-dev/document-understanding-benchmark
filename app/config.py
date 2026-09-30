from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "document-understanding-benchmark")
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", "20971520"))
    vlm_model_name: str = os.getenv("VLM_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
    vlm_device: str = os.getenv("VLM_DEVICE", "auto")
    ocr_language: str = os.getenv("OCR_LANGUAGE", "eng")


settings = Settings()
