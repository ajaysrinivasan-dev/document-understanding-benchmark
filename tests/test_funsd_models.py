from pathlib import Path

import pytest

from src.datasets.funsd_types import FunsdEntity
from src.evaluation.funsd_metrics import entity_metrics, evaluate_funsd_dataset
from src.layout.funsd_pipeline import (
    LayoutLMv3FUNSDPipeline,
    normalize_box,
    reconstruct_funsd_entities,
)
from src.schemas import Annotation, BoundingBox, DocumentResult, OCRWord
from src.vlm.pipeline import QwenVLM


def ocr_word(text: str, x: float, y: float = 10) -> OCRWord:
    return OCRWord(
        text=text,
        page_number=1,
        bbox=BoundingBox(x0=x, y0=y, x1=x + 20, y1=y + 10),
    )


def test_bio_reconstruction_groups_words_and_preserves_geometry():
    entities = reconstruct_funsd_entities(
        [ocr_word("What", 10), ocr_word("is", 35), ocr_word("name", 60), ocr_word("Ada", 10, 40)],
        ["B-QUESTION", "I-QUESTION", "I-QUESTION", "B-ANSWER"],
        [0.9, 0.8, 0.7, 0.95],
    )
    assert [(entity.label, entity.text) for entity in entities] == [
        ("question", "What is name"),
        ("answer", "Ada"),
    ]
    assert entities[0].bbox == (10.0, 10.0, 80.0, 20.0)
    assert entities[0].confidence == pytest.approx(0.8)


def test_bio_subword_alignment_uses_word_ids_and_unknown_labels_are_outside():
    entities = reconstruct_funsd_entities(
        [ocr_word("Title", 1), ocr_word("Body", 1, 40)], ["LABEL_QUESTION", "O"]
    )
    assert len(entities) == 1
    assert entities[0].label == "question"


def test_box_normalization_uses_layoutlm_1000_coordinate_space():
    box = BoundingBox(x0=10, y0=20, x1=110, y1=220)
    assert normalize_box(box, 200, 400) == [50, 50, 550, 550]


def test_entity_metrics_cover_exact_text_and_empty_predictions():
    expected = [FunsdEntity(label="question", text="Name", bbox=(0, 0, 100, 40))]
    predicted = [FunsdEntity(label="question", text="Name", bbox=(0, 0, 100, 40))]
    metrics = entity_metrics(expected, predicted)
    assert metrics["entity_f1"] == 1.0
    assert metrics["exact_text_match"] == 1.0
    empty_metrics = entity_metrics(expected, [])
    assert empty_metrics["entity_recall"] == 0.0


def test_funsd_evaluator_excludes_other_from_benchmark_metrics():
    entity = FunsdEntity(label="question", text="Name", bbox=(0, 0, 100, 40))
    other = FunsdEntity(label="other", text="Noise", bbox=(100, 0, 160, 40))
    annotation = Annotation(
        document_id="doc-1",
        document_type="funsd_form",
        metadata={"funsd_entities": [entity.model_dump(), other.model_dump()]},
    )
    result = DocumentResult(
        document_id="doc-1",
        document_type="funsd_form",
        pipeline="layoutlmv3",
        metadata={"funsd_entities": [entity.model_dump()]},
    )
    report = evaluate_funsd_dataset([annotation], [result], "layoutlmv3")
    assert report["metrics"]["entity_f1"] == 1.0
    assert report["metrics"]["expected_entities"] == 1


def test_qwen_funsd_mode_validates_json_without_model_download(monkeypatch, tmp_path: Path):
    model = QwenVLM("not-loaded")
    monkeypatch.setattr(model, "_load", lambda: None)
    monkeypatch.setattr(
        model,
        "_infer",
        lambda path, prompt: (
            '{"entities": [{"label": "header", "text": "Title", '
            '"bbox": [0, 0, 10, 10], "links": []}]}'
        ),
    )
    result = model.process(tmp_path / "image.png", "doc-1", task_mode="funsd")
    assert result.errors == []
    assert result.metadata["funsd_entities"][0]["label"] == "header"


def test_qwen_funsd_malformed_output_fails_closed(monkeypatch, tmp_path: Path):
    model = QwenVLM("not-loaded")
    monkeypatch.setattr(model, "_load", lambda: None)
    monkeypatch.setattr(model, "_infer", lambda path, prompt: "[]")
    result = model.process(tmp_path / "image.png", "doc-1", task_mode="funsd")
    assert result.metadata["funsd_entities"] == []
    assert result.errors


def test_funsd_error_does_not_become_missed_entity_flood():
    entity = FunsdEntity(label="question", text="Name", bbox=(0, 0, 100, 40))
    annotation = Annotation(
        document_id="doc-1",
        document_type="funsd_form",
        metadata={"funsd_entities": [entity.model_dump()]},
    )
    error_result = DocumentResult(
        document_id="doc-1",
        document_type="funsd_form",
        pipeline="vlm",
        errors=["VLM returned malformed JSON"],
    )
    report = evaluate_funsd_dataset([annotation], [error_result], "vlm")
    failures = report["documents"][0]["failure_categories"]
    assert failures["VLM formatting error"] == 1
    assert failures["missed entity"] == 0
    assert report["error_document_count"] == 1


def test_mocked_layout_and_qwen_outputs_share_funsd_evaluator(monkeypatch, tmp_path: Path):
    entity = FunsdEntity(label="question", text="Name", bbox=(0, 0, 100, 40))
    annotation = Annotation(
        document_id="doc-1",
        document_type="funsd_form",
        metadata={"funsd_entities": [entity.model_dump()]},
    )
    layout = LayoutLMv3FUNSDPipeline("mock-checkpoint")
    (tmp_path / "image.png").write_bytes(b"mock image bytes")
    monkeypatch.setattr(layout, "_load", lambda: None)
    monkeypatch.setattr(
        layout,
        "_process_page",
        lambda content, page: ([entity], 1.0, 2.0),
    )
    layout_result = layout.process(tmp_path / "image.png", "doc-1")
    qwen_result = DocumentResult(
        document_id="doc-1",
        document_type="funsd_form",
        pipeline="vlm",
        metadata={"funsd_entities": [entity.model_dump()]},
    )

    layout_report = evaluate_funsd_dataset([annotation], [layout_result], "layoutlmv3")
    qwen_report = evaluate_funsd_dataset([annotation], [qwen_result], "vlm")
    assert layout_report["metrics"]["entity_f1"] == 1.0
    assert qwen_report["metrics"]["entity_f1"] == 1.0
