from __future__ import annotations

import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from src.datasets.funsd_types import FunsdEntity, FunsdLabel
from src.ingestion.image_loader import load_document_pages
from src.ocr.pipeline import build_ocr_word, configure_tesseract
from src.schemas import BoundingBox, DocumentResult, OCRWord


class LayoutLMUnavailableError(RuntimeError):
    pass


class LayoutLMv3FUNSDPipeline:
    """LayoutLMv3 token classification baseline for FUNSD entities."""

    name = "layoutlmv3"

    def __init__(
        self,
        checkpoint: str,
        device: str = "auto",
        max_sequence_length: int = 512,
        processor_checkpoint: str = "microsoft/layoutlmv3-base",
        image_size: int = 224,
        confidence_threshold: float = 0.0,
        language: str = "eng",
        tesseract_cmd: str = "tesseract",
    ) -> None:
        self.checkpoint = checkpoint
        self.processor_checkpoint = processor_checkpoint
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.language = language
        self.tesseract_cmd = tesseract_cmd
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForTokenClassification, LayoutLMv3Processor
        except ImportError as error:
            raise LayoutLMUnavailableError("LayoutLMv3 dependencies are not installed") from error
        self._torch = torch
        self._processor = LayoutLMv3Processor.from_pretrained(
            self.processor_checkpoint, apply_ocr=False
        )
        image_processor = getattr(self._processor, "image_processor", None)
        if image_processor is not None:
            image_processor.size = {"height": self.image_size, "width": self.image_size}
        self._model = AutoModelForTokenClassification.from_pretrained(self.checkpoint)
        id_to_label = self._model.config.id2label
        if not any(parse_bio_label(str(label))[1] is not None for label in id_to_label.values()):
            raise LayoutLMUnavailableError(
                "LayoutLMv3 checkpoint id2label does not expose FUNSD BIO labels"
            )
        if self.device != "auto":
            self._model.to(self.device)
        self._model.eval()

    def process(self, path: str | Path, document_id: str | None = None) -> DocumentResult:
        started = time.perf_counter()
        identifier = document_id or Path(path).stem
        result = DocumentResult(
            document_id=identifier,
            document_type="funsd_form",
            pipeline=self.name,
            metadata={
                "funsd_entities": [],
                "checkpoint": self.checkpoint,
                "processor_checkpoint": self.processor_checkpoint,
            },
        )
        try:
            self._load()
            entities: list[FunsdEntity] = []
            ocr_time_ms = 0.0
            inference_time_ms = 0.0
            for page_number, content in enumerate(load_document_pages(path), start=1):
                page_entities, page_ocr_ms, page_inference_ms = self._process_page(
                    content, page_number
                )
                entities.extend(page_entities)
                ocr_time_ms += page_ocr_ms
                inference_time_ms += page_inference_ms
            result.metadata["funsd_entities"] = [entity.model_dump() for entity in entities]
            result.metadata["ocr_time_ms"] = ocr_time_ms
            result.metadata["layout_model_inference_time_ms"] = inference_time_ms
            result.warnings.append(
                "FUNSD entities were reconstructed from Tesseract words and LayoutLMv3 BIO labels"
            )
        except (LayoutLMUnavailableError, OSError, RuntimeError, ValueError) as error:
            result.errors.append(str(error))
        result.processing_time_ms = (time.perf_counter() - started) * 1000
        return result

    def _process_page(
        self, content: bytes, page_number: int
    ) -> tuple[list[FunsdEntity], float, float]:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as error:
            raise LayoutLMUnavailableError("OCR dependencies are not installed") from error
        configure_tesseract(pytesseract, self.tesseract_cmd)
        ocr_started = time.perf_counter()
        with Image.open(BytesIO(content)) as image:
            image = image.convert("RGB")
            data = pytesseract.image_to_data(
                image, lang=self.language, output_type=pytesseract.Output.DICT
            )
            ocr_time_ms = (time.perf_counter() - ocr_started) * 1000
            words = [
                word
                for index in range(len(data.get("text", [])))
                if (word := build_ocr_word(data, index, page_number)) is not None
                and word.bbox is not None
            ]
            if not words:
                return [], ocr_time_ms, 0.0
            inference_started = time.perf_counter()
            labels, confidences = self._predict_words(image.copy(), words)
            inference_time_ms = (time.perf_counter() - inference_started) * 1000
            labels = [
                label if confidence >= self.confidence_threshold else "O"
                for label, confidence in zip(labels, confidences, strict=True)
            ]
        return (
            reconstruct_funsd_entities(words, labels, confidences),
            ocr_time_ms,
            inference_time_ms,
        )

    def _predict_words(self, image: Any, words: list[OCRWord]) -> tuple[list[str], list[float]]:
        boxes = [normalize_box(word.bbox, image.width, image.height) for word in words]
        encoding = self._processor(
            image,
            [[word.text for word in words]],
            boxes=[boxes],
            truncation=True,
            max_length=self.max_sequence_length,
            padding=True,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)
        model_device = next(self._model.parameters()).device
        model_inputs = {key: value.to(model_device) for key, value in encoding.items()}
        with self._torch.no_grad():
            logits = self._model(**model_inputs).logits[0]
        token_scores: dict[int, list[Any]] = defaultdict(list)
        for token_index, word_id in enumerate(word_ids):
            if word_id is not None and word_id < len(words):
                token_scores[word_id].append(logits[token_index])
        id_to_label = self._model.config.id2label
        labels: list[str] = []
        confidences: list[float] = []
        for word_index in range(len(words)):
            if word_index not in token_scores:
                labels.append("O")
                confidences.append(0.0)
                continue
            word_logits = self._torch.stack(token_scores[word_index]).mean(dim=0)
            probabilities = self._torch.softmax(word_logits, dim=-1)
            score, label_id = self._torch.max(probabilities, dim=-1)
            labels.append(str(id_to_label[int(label_id)]))
            confidences.append(float(score))
        return labels, confidences


def normalize_box(box: BoundingBox | None, width: int, height: int) -> list[int]:
    if box is None or width <= 0 or height <= 0:
        raise ValueError("a valid image-sized bounding box is required")
    return [
        max(0, min(1000, round(box.x0 / width * 1000))),
        max(0, min(1000, round(box.y0 / height * 1000))),
        max(0, min(1000, round(box.x1 / width * 1000))),
        max(0, min(1000, round(box.y1 / height * 1000))),
    ]


def reconstruct_funsd_entities(
    words: list[OCRWord], labels: list[str], confidences: list[float] | None = None
) -> list[FunsdEntity]:
    """Reconstruct contiguous FUNSD entities from word-level BIO predictions."""
    if len(words) != len(labels):
        raise ValueError("words and labels must have the same length")
    scores = confidences or [1.0] * len(words)
    if len(scores) != len(words):
        raise ValueError("words and confidences must have the same length")
    entities: list[FunsdEntity] = []
    current_words: list[OCRWord] = []
    current_label: FunsdLabel | None = None
    current_scores: list[float] = []

    def flush() -> None:
        nonlocal current_words, current_label, current_scores
        if current_label and current_words:
            entities.append(
                FunsdEntity(
                    entity_id=f"entity-{len(entities) + 1}",
                    label=current_label,
                    text=" ".join(word.text for word in current_words),
                    bbox=_union_box(current_words),
                    page_number=current_words[0].page_number,
                    confidence=sum(current_scores) / len(current_scores),
                )
            )
        current_words = []
        current_label = None
        current_scores = []

    for word, raw_label, confidence in zip(words, labels, scores, strict=True):
        prefix, label = parse_bio_label(raw_label)
        if label is None:
            flush()
        elif prefix == "B" or current_label != label:
            flush()
            current_label = label
            current_words = [word]
            current_scores = [confidence]
        else:
            current_words.append(word)
            current_scores.append(confidence)
    flush()
    return entities


def parse_bio_label(raw_label: str) -> tuple[str, FunsdLabel | None]:
    label = raw_label.upper().replace("_", "-")
    if label.startswith("LABEL-"):
        label = label.removeprefix("LABEL-")
    if label == "O":
        return "O", None
    if "-" in label:
        prefix, label = label.split("-", 1)
    else:
        prefix = "B"
    if label.casefold() not in {"question", "answer", "header", "other"}:
        return "O", None
    return prefix if prefix in {"B", "I"} else "B", cast(FunsdLabel, label.casefold())


def _union_box(words: list[OCRWord]) -> tuple[float, float, float, float]:
    boxes = [word.bbox for word in words if word.bbox is not None]
    if not boxes:
        raise ValueError("entity words must have bounding boxes")
    return (
        min(box.x0 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.x1 for box in boxes),
        max(box.y1 for box in boxes),
    )
