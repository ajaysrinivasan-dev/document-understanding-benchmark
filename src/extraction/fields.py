from typing import Any


def extract_fields(payload: dict[str, Any]) -> dict[str, Any]:
    fields = payload.get("fields", {})
    return fields if isinstance(fields, dict) else {}
