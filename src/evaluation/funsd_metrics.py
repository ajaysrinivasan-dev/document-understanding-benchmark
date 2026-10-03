from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.datasets.funsd_types import FunsdEntity
from src.extraction.normalization import normalize_text
from src.schemas import Annotation, DocumentResult

BENCHMARK_LABELS = ("question", "answer", "header")

FAILURE_CATEGORIES = (
    "OCR recognition error",
    "missed entity",
    "incorrect entity label",
    "incorrect entity boundary",
    "incorrect text",
    "question/answer confusion",
    "header confusion",
    "entity linking error",
    "VLM formatting error",
    "token/word alignment error",
)


def evaluate_funsd_document(annotation: Annotation, result: DocumentResult) -> dict[str, Any]:
    expected = _metadata_entities(annotation.metadata.get("funsd_entities", []))
    predicted = _metadata_entities(result.metadata.get("funsd_entities", []))
    expected = _benchmark_entities(expected)
    predicted = _benchmark_entities(predicted)
    metrics = entity_metrics(expected, predicted)
    failures = categorize_failures(expected, predicted, result)
    return {
        "document_id": annotation.document_id,
        "document_type": annotation.document_type,
        "status": "error" if result.errors else "ok",
        "pipeline": result.pipeline,
        "errors": result.errors,
        "warnings": result.warnings,
        "processing_time_ms": result.processing_time_ms,
        "timing": {
            key: value for key, value in result.metadata.items() if key.endswith("_time_ms")
        },
        "metrics": metrics,
        "failure_categories": failures["counts"],
        "failure_examples": failures["examples"],
    }


def evaluate_funsd_dataset(
    annotations: list[Annotation], results: list[DocumentResult], pipeline: str
) -> dict[str, Any]:
    by_id = {result.document_id: result for result in results}
    documents = []
    for annotation in annotations:
        result = by_id.get(
            annotation.document_id,
            DocumentResult(document_id=annotation.document_id, errors=["missing result"]),
        )
        documents.append(evaluate_funsd_document(annotation, result))
    valid_documents = [document for document in documents if document["status"] == "ok"]
    aggregate = _aggregate_documents(valid_documents)
    latency = _latency_summary(results)
    return {
        "dataset": "FUNSD",
        "pipeline": pipeline,
        "document_count": len(documents),
        "valid_document_count": len(valid_documents),
        "error_document_count": len(documents) - len(valid_documents),
        "metrics": aggregate,
        "latency": latency,
        "documents": documents,
        "linking_evaluated": False,
        "linking_note": (
            "Entity linking is preserved but not scored because neither baseline "
            "predicts links reliably."
        ),
    }


def entity_metrics(
    expected: list[FunsdEntity], predicted: list[FunsdEntity], iou_threshold: float = 0.5
) -> dict[str, Any]:
    localization_pairs = _greedy_pairs(
        expected, predicted, same_label=False, threshold=iou_threshold
    )
    entity_pairs = _greedy_pairs(expected, predicted, same_label=True, threshold=iou_threshold)
    exact_text = sum(
        normalize_text(expected[index].text) == normalize_text(predicted[predicted_index].text)
        for index, predicted_index in entity_pairs
    )
    per_label: dict[str, dict[str, float | int]] = {}
    for label in BENCHMARK_LABELS:
        label_expected = [entity for entity in expected if entity.label == label]
        label_predicted = [entity for entity in predicted if entity.label == label]
        label_pairs = _greedy_pairs(
            label_expected, label_predicted, same_label=True, threshold=iou_threshold
        )
        stats = _prf(len(label_pairs), len(label_predicted), len(label_expected))
        per_label[label] = {
            **stats,
            "matched": len(label_pairs),
            "expected": len(label_expected),
            "predicted": len(label_predicted),
        }
    stats = _prf(len(entity_pairs), len(predicted), len(expected))
    localization = _prf(len(localization_pairs), len(predicted), len(expected))
    return {
        "entity_precision": stats["precision"],
        "entity_recall": stats["recall"],
        "entity_f1": stats["f1"],
        "localization_precision": localization["precision"],
        "localization_recall": localization["recall"],
        "localization_f1": localization["f1"],
        "exact_text_match": exact_text / len(entity_pairs) if entity_pairs else 0.0,
        "per_label": per_label,
        "expected_entities": len(expected),
        "predicted_entities": len(predicted),
        "matched_entities": len(entity_pairs),
        "localized_entities": len(localization_pairs),
        "exact_text_matches": exact_text,
    }


def categorize_failures(
    expected: list[FunsdEntity], predicted: list[FunsdEntity], result: DocumentResult
) -> dict[str, Any]:
    counts = {category: 0 for category in FAILURE_CATEGORIES}
    examples: list[dict[str, Any]] = []
    if result.errors:
        if result.pipeline == "vlm":
            counts["VLM formatting error"] += 1
            examples.append(
                {"category": "VLM formatting error", "reason": "pipeline returned an error"}
            )
        return {"counts": counts, "examples": examples}
    matched = set(index for index, _ in _greedy_pairs(expected, predicted, same_label=True))
    for expected_index, predicted_index in _greedy_pairs(expected, predicted, same_label=True):
        if normalize_text(expected[expected_index].text) != normalize_text(
            predicted[predicted_index].text
        ):
            counts["incorrect text"] += 1
            if len(examples) < 20:
                examples.append(
                    {
                        "category": "incorrect text",
                        "expected_entity_id": expected[expected_index].entity_id,
                    }
                )
    for expected_index, target in enumerate(expected):
        if expected_index in matched:
            continue
        overlap = [candidate for candidate in predicted if _iou(target, candidate) > 0]
        category = "missed entity"
        if overlap:
            candidate = max(overlap, key=lambda item: _iou(target, item))
            if candidate.label != target.label:
                category = "incorrect entity label"
                if {candidate.label, target.label} == {"question", "answer"}:
                    category = "question/answer confusion"
                elif {candidate.label, target.label} == {"header", "other"}:
                    category = "header confusion"
            elif normalize_text(candidate.text) != normalize_text(target.text):
                category = "incorrect text"
            else:
                category = "incorrect entity boundary"
        counts[category] += 1
        if len(examples) < 20:
            examples.append({"category": category, "expected_entity_id": target.entity_id})
    return {"counts": counts, "examples": examples}


def _aggregate_documents(documents: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, float] = defaultdict(float)
    label_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"matched": 0, "expected": 0, "predicted": 0}
    )
    expected = predicted = 0
    failure_counts: dict[str, int] = defaultdict(int)
    for document in documents:
        metrics = document["metrics"]
        expected += metrics["expected_entities"]
        predicted += metrics["predicted_entities"]
        totals["matched_entities"] += metrics["matched_entities"]
        totals["localized_entities"] += metrics["localized_entities"]
        totals["exact_text_matches"] += metrics["exact_text_matches"]
        for label, label_metrics in metrics["per_label"].items():
            label_totals[label]["matched"] += int(label_metrics["matched"])
            label_totals[label]["expected"] += int(label_metrics["expected"])
            label_totals[label]["predicted"] += int(label_metrics["predicted"])
        for category, count in document["failure_categories"].items():
            failure_counts[category] += count
    entity_stats = _prf(int(totals["matched_entities"]), predicted, expected) if documents else {}
    localization_stats = (
        _prf(int(totals["localized_entities"]), predicted, expected) if documents else {}
    )
    per_label = {
        label: {
            **_prf(values["matched"], values["predicted"], values["expected"]),
            "expected": values["expected"],
            "predicted": values["predicted"],
        }
        for label, values in label_totals.items()
    }
    return {
        "entity_precision": entity_stats.get("precision"),
        "entity_recall": entity_stats.get("recall"),
        "entity_f1": entity_stats.get("f1"),
        "localization_f1": localization_stats.get("f1"),
        "exact_text_match": (
            totals["exact_text_matches"] / totals["matched_entities"]
            if totals["matched_entities"]
            else None
        ),
        "per_label": per_label,
        "expected_entities": expected,
        "predicted_entities": predicted,
        "failure_categories": dict(failure_counts),
    }


def _latency_summary(results: list[DocumentResult]) -> dict[str, Any]:
    total = sorted(
        result.processing_time_ms for result in results if result.processing_time_ms is not None
    )
    ocr = [result.metadata["ocr_time_ms"] for result in results if "ocr_time_ms" in result.metadata]
    layout = [
        result.metadata["layout_model_inference_time_ms"]
        for result in results
        if "layout_model_inference_time_ms" in result.metadata
    ]
    vlm = [
        result.metadata["vlm_inference_time_ms"]
        for result in results
        if "vlm_inference_time_ms" in result.metadata
    ]
    return {
        "document_count": len(total),
        "average_total_ms": sum(total) / len(total) if total else None,
        "p50_total_ms": _percentile(total, 0.50),
        "p95_total_ms": _percentile(total, 0.95),
        "average_ocr_ms": sum(ocr) / len(ocr) if ocr else None,
        "average_layout_model_inference_ms": sum(layout) / len(layout) if layout else None,
        "average_vlm_inference_ms": sum(vlm) / len(vlm) if vlm else None,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    index = min(len(values) - 1, max(0, round((len(values) - 1) * percentile)))
    return values[index]


def _benchmark_entities(entities: list[FunsdEntity]) -> list[FunsdEntity]:
    """Keep only the FUNSD labels used by the official token-classification task."""
    return [entity for entity in entities if entity.label in BENCHMARK_LABELS]


def _metadata_entities(value: Any) -> list[FunsdEntity]:
    if not isinstance(value, list):
        return []
    entities: list[FunsdEntity] = []
    for item in value:
        if isinstance(item, dict):
            entities.append(FunsdEntity.model_validate(item))
    return entities


def _greedy_pairs(
    expected: list[FunsdEntity],
    predicted: list[FunsdEntity],
    same_label: bool,
    threshold: float = 0.5,
) -> list[tuple[int, int]]:
    candidates = sorted(
        (
            _iou(target, candidate),
            expected_index,
            predicted_index,
        )
        for expected_index, target in enumerate(expected)
        for predicted_index, candidate in enumerate(predicted)
        if (not same_label or target.label == candidate.label)
        and _iou(target, candidate) >= threshold
    )
    pairs: list[tuple[int, int]] = []
    used_expected: set[int] = set()
    used_predicted: set[int] = set()
    for _, expected_index, predicted_index in reversed(candidates):
        if expected_index not in used_expected and predicted_index not in used_predicted:
            pairs.append((expected_index, predicted_index))
            used_expected.add(expected_index)
            used_predicted.add(predicted_index)
    return pairs


def _iou(left: FunsdEntity, right: FunsdEntity) -> float:
    if left.bbox is None or right.bbox is None:
        return 0.0
    x0 = max(left.bbox[0], right.bbox[0])
    y0 = max(left.bbox[1], right.bbox[1])
    x1 = min(left.bbox[2], right.bbox[2])
    y1 = min(left.bbox[3], right.bbox[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left.bbox[2] - left.bbox[0]) * max(0.0, left.bbox[3] - left.bbox[1])
    right_area = max(0.0, right.bbox[2] - right.bbox[0]) * max(0.0, right.bbox[3] - right.bbox[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _prf(true_positive: int, predicted: int, actual: int) -> dict[str, float]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
