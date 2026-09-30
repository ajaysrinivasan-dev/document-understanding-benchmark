from pathlib import Path


def validate_document_path(path: str | Path) -> Path:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return source
