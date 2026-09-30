from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.schemas import DocumentResult


class DocumentProcessor(Protocol):
    name: str

    def process(self, path: str | Path, document_id: str | None = None) -> DocumentResult: ...
