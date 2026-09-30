from __future__ import annotations

from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def load_document_pages(path: str | Path) -> list[bytes]:
    source = Path(path)
    if source.suffix.casefold() == ".pdf":
        from src.ingestion.pdf_loader import load_pdf_pages

        return load_pdf_pages(source)
    if source.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported document extension: {source.suffix}")
    return [source.read_bytes()]
