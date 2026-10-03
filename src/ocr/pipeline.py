from __future__ import annotations

import re
import time
from io import BytesIO
from math import isfinite
from pathlib import Path
from typing import Any

from src.extraction.tables import extract_tables
from src.ingestion.image_loader import load_document_pages
from src.layout.pipeline import group_words_into_lines
from src.schemas import BoundingBox, DocumentResult, OCRWord


class OCRLayoutPipeline:
    name = "ocr"

    def __init__(self, language: str = "eng", tesseract_cmd: str = "tesseract") -> None:
        self.language = language
        self.tesseract_cmd = tesseract_cmd

    def process(self, path: str | Path, document_id: str | None = None) -> DocumentResult:
        started = time.perf_counter()
        identifier = document_id or Path(path).stem
        result = DocumentResult(document_id=identifier, pipeline=self.name)
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            result.errors.append("OCR dependencies are not installed")
            result.processing_time_ms = (time.perf_counter() - started) * 1000
            return result
        configure_tesseract(pytesseract, self.tesseract_cmd)
        words: list[OCRWord] = []
        for page_number, content in enumerate(load_document_pages(path), start=1):
            with Image.open(BytesIO(content)) as image:
                data = pytesseract.image_to_data(
                    image, lang=self.language, output_type=pytesseract.Output.DICT
                )
            for index in range(len(data.get("text", []))):
                word = build_ocr_word(data, index, page_number)
                if word is not None:
                    words.append(word)
        layout_blocks = group_words_into_lines(words)
        result.fields = self._candidate_fields([word.text for word in words])
        result.tables = extract_tables(words, layout_blocks)
        result.warnings.append(
            "OCR fields and tables use deterministic heuristics and require "
            "annotation-based evaluation"
        )
        result.processing_time_ms = (time.perf_counter() - started) * 1000
        return result

    @staticmethod
    def _candidate_fields(words: list[str]) -> dict[str, str]:
        fields: dict[str, str] = {}
        text = " ".join(words)
        patterns = {
            "invoice_number": r"invoice\s*(?:no|number|#)?\s*[:#]?\s*([A-Za-z0-9-]+)",
            "total": r"total\s*[:$]?\s*([0-9,]+(?:\.\d{2})?)",
        }
        for name, pattern in patterns.items():
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                fields[name] = match.group(1)
        return fields


def configure_tesseract(pytesseract_module: Any, configured_value: str | None = None) -> str:
    """Set the pytesseract executable, using the platform PATH by default."""
    command = configured_value or "tesseract"
    pytesseract_module.pytesseract.tesseract_cmd = command
    return command


def build_ocr_word(data: dict[str, list[object]], index: int, page_number: int) -> OCRWord | None:
    """Convert one Tesseract image-to-data row, rejecting empty or invalid rows."""
    raw_text = data.get("text", [])[index] if index < len(data.get("text", [])) else ""
    text = str(raw_text).strip()
    if not text:
        return None

    confidence = _parse_confidence(data.get("conf", []), index)
    bbox = _parse_bbox(data, index)
    return OCRWord(text=text, confidence=confidence, page_number=page_number, bbox=bbox)


def _parse_confidence(values: list[object], index: int) -> float | None:
    if index >= len(values):
        return None
    try:
        value = float(str(values[index]))
    except (TypeError, ValueError):
        return None
    return value if isfinite(value) and 0 <= value <= 100 else None


def _parse_bbox(data: dict[str, list[object]], index: int) -> BoundingBox | None:
    keys = ("left", "top", "width", "height")
    if any(index >= len(data.get(key, [])) for key in keys):
        return None
    try:
        left, top, width, height = (float(str(data[key][index])) for key in keys)
    except (TypeError, ValueError):
        return None
    if not all(isfinite(value) for value in (left, top, width, height)):
        return None
    if left < 0 or top < 0 or width <= 0 or height <= 0:
        return None
    return BoundingBox(x0=left, y0=top, x1=left + width, y1=top + height)
