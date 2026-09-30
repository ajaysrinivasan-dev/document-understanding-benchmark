from typing import Any


def extract_tables(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tables = payload.get("tables", [])
    return tables if isinstance(tables, list) else []
