from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.evaluator import evaluate_dataset
from src.schemas import Annotation, DocumentResult


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate stored document results against annotations"
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/results.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("reports/results.md"))
    args = parser.parse_args()
    annotations = [
        Annotation.model_validate(item)
        for item in json.loads(args.annotations.read_text(encoding="utf-8"))
    ]
    results = [
        DocumentResult.model_validate(item)
        for item in json.loads(args.results.read_text(encoding="utf-8"))
    ]
    report = evaluate_dataset(annotations, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    failures = [
        document
        for document in report["documents"]
        if document["fields"]["f1"] < 1 or document["tables"]["cell_accuracy"] < 1
    ]
    markdown = [
        "# Evaluation Results",
        "",
        f"Documents evaluated: {report['document_count']}",
        "",
        f"Documents with field or table mismatches: {len(failures)}",
        "",
        "## Failure analysis",
        "",
    ]
    markdown.extend(
        f"- `{item['document_id']}` ({item['document_type']}): "
        f"field F1={item['fields']['f1']:.3f}, "
        f"table cell accuracy={item['tables']['cell_accuracy']:.3f}"
        for item in failures
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(markdown) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
