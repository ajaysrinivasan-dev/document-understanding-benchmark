from __future__ import annotations

import re
import time
from pathlib import Path

from src.ingestion.image_loader import load_document_pages
from src.schemas import DocumentResult, OCRWord


class OCRLayoutPipeline:
    name = "ocr"

    def __init__(self, language: str = "eng") -> None:
        self.language = language

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
        words: list[OCRWord] = []
        for page_number, content in enumerate(load_document_pages(path), start=1):
            image = Image.open(__import__("io").BytesIO(content))
            data = pytesseract.image_to_data(
                image, lang=self.language, output_type=pytesseract.Output.DICT
            )
            for index, text in enumerate(data["text"]):
                if not text.strip():
                    continue
                words.append(
                    OCRWord(
                        text=text, confidence=float(data["conf"][index]), page_number=page_number
                    )
                )
        result.fields = self._candidate_fields([word.text for word in words])
        result.warnings.append(
            "OCR/layout fields are heuristic candidates and require annotation-based evaluation"
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
