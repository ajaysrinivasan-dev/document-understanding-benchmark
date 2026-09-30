from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.schemas import DocumentResult


class VLMClient(Protocol):
    def extract(self, path: str | Path, document_id: str) -> DocumentResult: ...
