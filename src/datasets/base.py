from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from src.schemas import Annotation


class DatasetAdapter(Protocol):
    """Interface implemented by local, reproducible annotation adapters."""

    name: str

    def load_annotations(self, split: str | None = None) -> list[Annotation]: ...

    def iter_documents(self, split: str | None = None) -> Iterable[tuple[Annotation, Path]]: ...

    def manifest(self, split: str | None = None) -> dict[str, object]: ...
