from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from src.benchmarking.resources import environment_info
from src.ocr.pipeline import OCRLayoutPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure pipeline latency on real documents")
    parser.add_argument(
        "--documents", type=Path, required=True, help="Directory containing documents"
    )
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    args = parser.parse_args()
    documents = [path for path in args.documents.iterdir() if path.is_file()]
    pipeline = OCRLayoutPipeline()
    measurements = []
    for document in documents:
        started = time.perf_counter()
        result = pipeline.process(document)
        measurements.append(
            {
                "document_id": result.document_id,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "errors": result.errors,
            }
        )
    payload = {
        "environment": {**environment_info(), "platform": platform.platform()},
        "pipeline": pipeline.name,
        "document_count": len(documents),
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
