from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def normalize_date(value: Any) -> str:
    text = normalize_text(value)
    for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            continue
    return text


def normalize_number(value: Any) -> str:
    text = normalize_text(value).replace(",", "")
    text = re.sub(r"[^0-9.\-()]", "", text)
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"
    try:
        formatted = format(Decimal(text), "f")
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted or "0"
    except (InvalidOperation, ValueError):
        return text


def normalize_value(field_name: str, value: Any) -> Any:
    if value is None:
        return None
    name = field_name.casefold()
    if any(token in name for token in ("date", "dob")):
        return normalize_date(value)
    if any(token in name for token in ("amount", "total", "subtotal", "tax", "price", "quantity")):
        return normalize_number(value)
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return [normalize_value(field_name, item) for item in value]
    return value


def normalize_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {key: normalize_value(key, value) for key, value in fields.items()}


def normalize_table(table: dict[str, Any]) -> dict[str, Any]:
    columns = [normalize_text(column) for column in table.get("columns", [])]
    rows = [[normalize_text(cell) for cell in row] for row in table.get("rows", [])]
    return {**table, "columns": columns, "rows": rows}
