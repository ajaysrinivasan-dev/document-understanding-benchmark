from __future__ import annotations

from src.evaluation.metrics import field_metrics, table_metrics
from src.schemas import Annotation, DocumentResult


def evaluate_document(annotation: Annotation, result: DocumentResult) -> dict:
    fields = field_metrics(annotation.fields, result.fields)
    tables = table_metrics(
        [table.model_dump() for table in annotation.tables],
        [table.model_dump() for table in result.tables],
    )
    return {
        "document_id": annotation.document_id,
        "document_type": annotation.document_type,
        "fields": fields,
        "tables": tables,
    }


def evaluate_dataset(annotations: list[Annotation], results: list[DocumentResult]) -> dict:
    by_id = {result.document_id: result for result in results}
    documents = [
        evaluate_document(
            annotation,
            by_id.get(
                annotation.document_id,
                DocumentResult(document_id=annotation.document_id, errors=["missing result"]),
            ),
        )
        for annotation in annotations
    ]
    return {"document_count": len(documents), "documents": documents}
