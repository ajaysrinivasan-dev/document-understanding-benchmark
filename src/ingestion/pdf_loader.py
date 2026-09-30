from __future__ import annotations

from pathlib import Path


def load_pdf_pages(path: str | Path) -> list[bytes]:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError("PDF support requires the optional 'ocr' dependencies") from error
    document = fitz.open(path)
    pages: list[bytes] = []
    for page in document:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pages.append(pixmap.tobytes("png"))
    return pages
