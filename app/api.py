from __future__ import annotations

import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from app.config import settings
from src.ocr.pipeline import OCRLayoutPipeline
from src.schemas import ExtractionResponse
from src.vlm.pipeline import QwenVLM

logger = logging.getLogger(__name__)
router = APIRouter()
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
ALLOWED_CONTENT_TYPES = {"application/pdf", "image/png", "image/jpeg", "image/tiff", "image/webp"}


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/info")
def info() -> dict[str, str | int]:
    return {
        "name": settings.app_name,
        "vlm_model": settings.vlm_model_name,
        "max_upload_bytes": settings.max_upload_bytes,
    }


@router.post("/extract", response_model=ExtractionResponse)
async def extract(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    pipeline: Annotated[str, Query(pattern="^(ocr|vlm|both)$")] = "ocr",
) -> ExtractionResponse:
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    suffix = Path(file.filename or "").suffix.casefold()
    if suffix not in ALLOWED_EXTENSIONS or (
        file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES
    ):
        raise HTTPException(status_code=415, detail="Unsupported document type")
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Uploaded document exceeds size limit")
    started = time.perf_counter()
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="du-", ignore_cleanup_errors=True) as directory:
            path = Path(directory) / f"document{suffix}"
            path.write_bytes(content)
            if pipeline in {"ocr", "both"}:
                results.append(
                    OCRLayoutPipeline(
                        settings.ocr_language,
                        settings.tesseract_cmd,
                    ).process(path, request_id)
                )
            if pipeline in {"vlm", "both"}:
                results.append(
                    QwenVLM(
                        settings.vlm_model_name,
                        settings.vlm_device,
                        max_new_tokens=settings.vlm_max_new_tokens,
                        backend=settings.vlm_backend,
                        base_url=settings.vlm_base_url,
                        timeout_seconds=settings.vlm_timeout_seconds,
                        context_size=settings.vlm_context_size,
                    ).process(path, request_id)
                )
    except (OSError, ValueError) as error:
        logger.exception("document processing failed request_id=%s", request_id)
        raise HTTPException(status_code=422, detail="Document could not be processed") from error
    duration = (time.perf_counter() - started) * 1000
    logger.info(
        "extraction request_id=%s pipeline=%s duration_ms=%.2f success=%s",
        request_id,
        pipeline,
        duration,
        not any(result.errors for result in results),
    )
    return ExtractionResponse(
        document_id=request_id,
        pipeline=pipeline,
        results=results,
        processing_time_ms=duration,
        warnings=[warning for result in results for warning in result.warnings],
        errors=[error for result in results for error in result.errors],
    )
