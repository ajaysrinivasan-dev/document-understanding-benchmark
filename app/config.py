from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "document-understanding-benchmark")
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", "20971520"))
    vlm_backend: str = os.getenv("VLM_BACKEND", "transformers")
    vlm_model_name: str = os.getenv("VLM_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
    vlm_device: str = os.getenv("VLM_DEVICE", "auto")
    vlm_base_url: str = os.getenv("VLM_BASE_URL", "http://127.0.0.1:11434")
    vlm_timeout_seconds: float = float(os.getenv("VLM_TIMEOUT_SECONDS", "300"))
    vlm_max_new_tokens: int = int(os.getenv("VLM_MAX_NEW_TOKENS", "2048"))
    vlm_context_size: int = int(os.getenv("VLM_CONTEXT_SIZE", "4096"))
    funsd_layout_checkpoint: str = os.getenv(
        "FUNSD_LAYOUT_CHECKPOINT", "nielsr/layoutlmv3-finetuned-funsd"
    )
    funsd_layout_processor_checkpoint: str = os.getenv(
        "FUNSD_LAYOUT_PROCESSOR_CHECKPOINT", "microsoft/layoutlmv3-base"
    )
    funsd_layout_device: str = os.getenv("FUNSD_LAYOUT_DEVICE", "auto")
    funsd_max_sequence_length: int = int(os.getenv("FUNSD_MAX_SEQUENCE_LENGTH", "512"))
    funsd_image_size: int = int(os.getenv("FUNSD_IMAGE_SIZE", "224"))
    funsd_confidence_threshold: float = float(os.getenv("FUNSD_CONFIDENCE_THRESHOLD", "0.0"))
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "tesseract")
    ocr_language: str = os.getenv("OCR_LANGUAGE", "eng")


settings = Settings()
