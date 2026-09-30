from __future__ import annotations

from typing import Any

from src.extraction.normalization import normalize_fields, normalize_table


def _prf(true_positive: int, predicted: int, actual: int) -> dict[str, float]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def field_metrics(expected: dict[str, Any], predicted: dict[str, Any]) -> dict[str, Any]:
    expected_normalized = normalize_fields(expected)
    predicted_normalized = normalize_fields(predicted)
    keys = set(expected_normalized) | set(predicted_normalized)
    matches = sum(
        1
        for key in keys
        if key in expected_normalized
        and key in predicted_normalized
        and expected_normalized[key] == predicted_normalized[key]
    )
    stats = _prf(matches, len(predicted_normalized), len(expected_normalized))
    return {"exact_match": matches / len(keys) if keys else 1.0, **stats, "field_count": len(keys)}


def table_metrics(
    expected: list[dict[str, Any]], predicted: list[dict[str, Any]]
) -> dict[str, Any]:
    expected_tables = [normalize_table(table) for table in expected]
    predicted_tables = [normalize_table(table) for table in predicted]
    table_count = min(len(expected_tables), len(predicted_tables))
    columns = sum(
        expected_tables[index]["columns"] == predicted_tables[index]["columns"]
        for index in range(table_count)
    )
    expected_cells = [cell for table in expected_tables for row in table["rows"] for cell in row]
    predicted_cells = [cell for table in predicted_tables for row in table["rows"] for cell in row]
    cell_matches = sum(
        left == right for left, right in zip(expected_cells, predicted_cells, strict=False)
    )
    cell_stats = _prf(cell_matches, len(predicted_cells), len(expected_cells))
    return {
        "table_detection_accuracy": 1.0 if bool(expected_tables) == bool(predicted_tables) else 0.0,
        "column_match": columns / len(expected_tables)
        if expected_tables
        else (1.0 if not predicted_tables else 0.0),
        "row_match": sum(
            expected_tables[index]["rows"] == predicted_tables[index]["rows"]
            for index in range(table_count)
        )
        / len(expected_tables)
        if expected_tables
        else (1.0 if not predicted_tables else 0.0),
        "cell_accuracy": cell_matches / max(len(expected_cells), len(predicted_cells), 1),
        **{f"cell_{key}": value for key, value in cell_stats.items()},
    }
