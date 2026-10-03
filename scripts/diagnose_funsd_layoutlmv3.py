from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TypeGuard

from PIL import Image

from app.config import settings
from src.datasets.funsd import FunsdAdapter
from src.layout.funsd_pipeline import (
    LayoutLMv3FUNSDPipeline,
    normalize_box,
    parse_bio_label,
)
from src.ocr.pipeline import build_ocr_word, configure_tesseract
from src.schemas import Annotation, BoundingBox, OCRWord


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic-only OCR/control-input LayoutLMv3 analysis for one FUNSD document"
    )
    parser.add_argument("--document", default="82092117")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/funsd"))
    parser.add_argument("--checkpoint", default=settings.funsd_layout_checkpoint)
    parser.add_argument("--device", default=settings.funsd_layout_device)
    args = parser.parse_args()

    adapter = FunsdAdapter(args.dataset_root)
    annotation, image_path = _find_document(adapter, args.document)
    image = Image.open(image_path).convert("RGB")
    print("DIAGNOSTIC ONLY: control-input results must never enter official benchmark reports")
    print(f"document_id: {args.document}")
    print(f"image: {image_path}")
    print(f"image width/height: {image.width}/{image.height}")
    print(f"TESSERACT_CMD: {settings.tesseract_cmd}")

    ocr_words = _run_tesseract(image, page_number=1)
    control_words = _control_words(annotation.metadata.get("entities", []))
    print(f"OCR word count: {len(ocr_words)}")
    print(f"FUNSD control word count: {len(control_words)}")
    _print_word_samples("OCR", ocr_words, image.width, image.height)
    _print_word_samples("FUNSD CONTROL (oracle-style)", control_words, image.width, image.height)

    pipeline = LayoutLMv3FUNSDPipeline(
        checkpoint=args.checkpoint,
        device=args.device,
        max_sequence_length=settings.funsd_max_sequence_length,
        image_size=settings.funsd_image_size,
        confidence_threshold=settings.funsd_confidence_threshold,
        language=settings.ocr_language,
        tesseract_cmd=settings.tesseract_cmd,
    )
    pipeline._load()
    print(f"checkpoint: {args.checkpoint}")
    print(f"model id2label: {pipeline._model.config.id2label}")
    _verify_expected_labels(pipeline._model.config.id2label)

    ocr_result = _run_inputs(pipeline, image, ocr_words, "OCR INPUT")
    control_result = _run_inputs(
        pipeline, image, control_words, "FUNSD CONTROL INPUT (oracle-style)"
    )
    print("Official benchmark reminder: only OCR-derived words/boxes are valid benchmark inputs.")
    print(f"OCR input errors: {ocr_result['errors']}")
    print(f"Control input errors: {control_result['errors']}")


def _find_document(adapter: FunsdAdapter, document_id: str) -> tuple[Annotation, Path]:
    for annotation, image_path in adapter.iter_documents("testing"):
        if annotation.document_id == document_id:
            return annotation, image_path
    raise FileNotFoundError(f"FUNSD testing document not found: {document_id}")


def _run_tesseract(image: Image.Image, page_number: int) -> list[OCRWord]:
    import pytesseract

    configure_tesseract(pytesseract, settings.tesseract_cmd)
    data = pytesseract.image_to_data(
        image, lang=settings.ocr_language, output_type=pytesseract.Output.DICT
    )
    words = [
        word
        for index in range(len(data.get("text", [])))
        if (word := build_ocr_word(data, index, page_number)) is not None and word.bbox is not None
    ]
    print(f"Tesseract executable configured: {pytesseract.pytesseract.tesseract_cmd}")
    return words


def _control_words(entities: Any) -> list[OCRWord]:
    words: list[OCRWord] = []
    if not isinstance(entities, list):
        return words
    for entity in entities:
        if not isinstance(entity, dict) or not isinstance(entity.get("words"), list):
            continue
        for source_word in entity["words"]:
            if not isinstance(source_word, dict):
                continue
            text = source_word.get("text")
            box = source_word.get("box")
            if not isinstance(text, str) or not text.strip() or not _valid_box(box):
                continue
            words.append(
                OCRWord(
                    text=text.strip(),
                    page_number=1,
                    bbox=BoundingBox(
                        x0=float(box[0]),
                        y0=float(box[1]),
                        x1=float(box[2]),
                        y1=float(box[3]),
                    ),
                )
            )
    return words


def _valid_box(box: Any) -> TypeGuard[list[int | float]]:
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, (int, float)) for value in box)
        and box[2] >= box[0]
        and box[3] >= box[1]
    )


def _print_word_samples(name: str, words: list[OCRWord], width: int, height: int) -> None:
    print(f"first 10 {name} words/boxes/normalized boxes:")
    for word in words[:10]:
        print(
            f"  {word.text!r} {word.bbox.model_dump() if word.bbox else None} "
            f"{normalize_box(word.bbox, width, height) if word.bbox else None}"
        )


def _verify_expected_labels(id_to_label: dict[Any, Any]) -> None:
    expected = {"O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"}
    normalized = {str(label).upper().replace("_", "-") for label in id_to_label.values()}
    print(f"expected FUNSD labels present: {sorted(expected & normalized)}")
    missing = sorted(expected - normalized)
    if missing:
        print(f"WARNING missing expected labels: {missing}")


def _run_inputs(
    pipeline: LayoutLMv3FUNSDPipeline,
    image: Image.Image,
    words: list[OCRWord],
    name: str,
) -> dict[str, Any]:
    import torch

    boxes = [normalize_box(word.bbox, image.width, image.height) for word in words]
    encoding = pipeline._processor(
        image,
        [[word.text for word in words]],
        boxes=[boxes],
        truncation=True,
        max_length=pipeline.max_sequence_length,
        padding=True,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)
    print(f"\n{name}")
    print(f"supplied words: {len(words)}")
    print(f"processor keys: {sorted(encoding.keys())}")
    for key in ("input_ids", "bbox", "attention_mask", "pixel_values"):
        value = encoding.get(key)
        print(f"{key}: shape={tuple(value.shape) if value is not None else None}")
    pixel_values = encoding.get("pixel_values")
    print(
        "pixel_values RGB-compatible: "
        f"{pixel_values is not None and pixel_values.ndim == 4 and pixel_values.shape[1] == 3}"
    )
    mapped_ids = sorted({word_id for word_id in word_ids if word_id is not None})
    print(f"sequence length: {encoding['input_ids'].shape[1]}")
    print(f"word_ids mapped: {len(mapped_ids)}; first mappings: {word_ids[:15]}")

    model_device = next(pipeline._model.parameters()).device
    model_inputs = {key: value.to(model_device) for key, value in encoding.items()}
    with torch.no_grad():
        logits = pipeline._model(**model_inputs).logits[0]
    id_to_label = pipeline._model.config.id2label
    word_logits: dict[int, list[Any]] = {}
    for token_index, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in word_logits:
            word_logits[word_id] = []
        if word_id is not None:
            word_logits[word_id].append(logits[token_index])
    predictions: list[dict[str, Any]] = []
    for word_index, word in enumerate(words):
        if word_index not in word_logits:
            continue
        probabilities = torch.softmax(torch.stack(word_logits[word_index]).mean(dim=0), dim=-1)
        top_values, top_ids = torch.topk(probabilities, k=min(3, probabilities.shape[-1]))
        top = [
            (str(id_to_label[int(label_id)]), float(value))
            for value, label_id in zip(top_values, top_ids, strict=True)
        ]
        predictions.append(
            {"word": word.text, "label": top[0][0], "probability": top[0][1], "top3": top}
        )
    labels = [prediction["label"] for prediction in predictions]
    non_o = [
        prediction
        for prediction in predictions
        if parse_bio_label(prediction["label"])[1] is not None
    ]
    print(f"unique predicted labels: {sorted(set(labels))}")
    print(f"non-O predictions: {len(non_o)}")
    print("first 10 word predictions:")
    for prediction in predictions[:10]:
        print(f"  {prediction}")
    return {"predictions": predictions, "errors": []}


if __name__ == "__main__":
    main()
