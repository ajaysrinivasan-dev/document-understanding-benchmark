from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from src.datasets.funsd_types import FUNSD_LABELS, FunsdEntity, FunsdLabel
from src.schemas import Annotation


class FunsdAdapter:
    """Convert a local FUNSD directory into canonical annotations.

    FUNSD entities are represented under ``fields`` by their original labels as
    lists of text values. Rich entity records, links, words, and boxes remain in
    ``metadata['entities']``. FUNSD has no table ground truth in its annotations,
    so this adapter always returns an empty ``tables`` list.
    """

    name = "funsd"
    source_url = "https://guillaumejaume.github.io/FUNSD/dataset.zip"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"FUNSD directory does not exist: {self.root}")

    def load_annotations(self, split: str | None = None) -> list[Annotation]:
        return [annotation for annotation, _ in self.iter_documents(split)]

    def iter_documents(self, split: str | None = None) -> list[tuple[Annotation, Path]]:
        documents: list[tuple[Annotation, Path]] = []
        for split_name, annotation_dir, image_dir in self._split_directories(split):
            for annotation_path in sorted(annotation_dir.glob("*.json")):
                annotation = self._load_annotation(annotation_path, split_name, image_dir)
                image_path = image_dir / f"{annotation.document_id}.png"
                if not image_path.is_file():
                    image_path = self._find_image(image_dir, annotation.document_id)
                documents.append((annotation, image_path))
        return documents

    def manifest(self, split: str | None = None) -> dict[str, object]:
        split_counts: dict[str, int] = {}
        for split_name, annotation_dir, _ in self._split_directories(split):
            split_counts[split_name] = len(list(annotation_dir.glob("*.json")))
        return {
            "dataset_name": "FUNSD",
            "dataset_version": "not specified by the source archive",
            "source": self.source_url,
            "document_count": sum(split_counts.values()),
            "splits": split_counts,
            "local_path": str(self.root),
            "annotation_format": "FUNSD JSON form entities converted to canonical Annotation",
            "license_reference": "Consult the official FUNSD source and accompanying release terms",
        }

    def _split_directories(self, split: str | None) -> list[tuple[str, Path, Path]]:
        candidates = {
            "train": "training_data",
            "training": "training_data",
            "test": "testing_data",
            "testing": "testing_data",
        }
        names = [split] if split else ["training", "testing"]
        directories: list[tuple[str, Path, Path]] = []
        for name in names:
            directory_name = candidates.get(name, name)
            split_root = self.root / directory_name
            annotation_dir = split_root / "annotations"
            image_dir = split_root / "images"
            if annotation_dir.is_dir() and image_dir.is_dir():
                directories.append((name, annotation_dir, image_dir))
            elif split is not None:
                raise FileNotFoundError(
                    f"FUNSD split '{split}' must contain {directory_name}/annotations and images"
                )
        if not directories:
            raise FileNotFoundError(
                f"No FUNSD splits found under {self.root}; expected "
                "training_data/ and/or testing_data/"
            )
        return directories

    @staticmethod
    def _load_annotation(annotation_path: Path, split: str, image_dir: Path) -> Annotation:
        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid FUNSD annotation: {annotation_path}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"FUNSD annotation must be an object: {annotation_path}")
        entities = payload.get("form", [])
        if not isinstance(entities, list):
            raise ValueError(f"FUNSD annotation 'form' must be a list: {annotation_path}")

        fields: dict[str, list[str]] = defaultdict(list)
        preserved_entities: list[dict[str, Any]] = []
        canonical_entities: list[dict[str, Any]] = []
        for entity in entities:
            if not isinstance(entity, dict):
                raise ValueError(f"FUNSD entity must be an object: {annotation_path}")
            label = entity.get("label")
            text = entity.get("text", "")
            preserved_entities.append(
                {
                    "id": entity.get("id"),
                    "label": label,
                    "text": text,
                    "box": entity.get("box"),
                    "linking": entity.get("linking", []),
                    "words": entity.get("words", []),
                }
            )
            if not isinstance(label, str) or not label.strip():
                continue
            if not isinstance(text, str):
                text = str(text)
            fields[label.strip().casefold()].append(text)
            if label.strip().casefold() in FUNSD_LABELS:
                box = entity.get("box")
                bbox: tuple[float, float, float, float] | None = None
                if isinstance(box, list) and len(box) == 4:
                    bbox = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                canonical_label = cast(FunsdLabel, label.strip().casefold())
                canonical_entities.append(
                    FunsdEntity(
                        entity_id=entity.get("id"),
                        label=canonical_label,
                        text=text,
                        bbox=bbox,
                        links=entity.get("linking", []),
                    ).model_dump()
                )
        return Annotation(
            document_id=annotation_path.stem,
            document_type="funsd_form",
            fields=dict(fields),
            tables=[],
            metadata={
                "dataset": "FUNSD",
                "split": split,
                "annotation_path": str(annotation_path),
                "image_path": str(image_dir / f"{annotation_path.stem}.png"),
                "entities": preserved_entities,
                "funsd_entities": canonical_entities,
                "source_payload_keys": sorted(payload.keys()),
            },
        )

    @staticmethod
    def _find_image(image_dir: Path, document_id: str) -> Path:
        matches = sorted(path for path in image_dir.glob(f"{document_id}.*") if path.is_file())
        if not matches:
            raise FileNotFoundError(f"No image found for FUNSD document: {document_id}")
        return matches[0]
