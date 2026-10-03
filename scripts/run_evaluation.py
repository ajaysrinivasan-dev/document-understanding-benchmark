from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

from app.config import settings
from src.datasets.funsd import FunsdAdapter
from src.evaluation.evaluator import evaluate_dataset
from src.evaluation.funsd_metrics import evaluate_funsd_dataset
from src.layout.funsd_pipeline import LayoutLMv3FUNSDPipeline
from src.ocr.pipeline import OCRLayoutPipeline
from src.schemas import Annotation, DocumentResult
from src.vlm.pipeline import QwenVLM


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate stored document results against annotations"
    )
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--dataset", choices=["funsd"])
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split", help="FUNSD split, such as training or testing")
    parser.add_argument("--pipeline", choices=["ocr", "layoutlmv3", "vlm"], default="ocr")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = (
            Path("reports") / f"funsd_{args.pipeline}_results.json"
            if args.dataset == "funsd"
            else Path("reports/results.json")
        )
    if args.markdown_output is None:
        args.markdown_output = (
            Path("reports") / f"funsd_{args.pipeline}_results.md"
            if args.dataset == "funsd"
            else Path("reports/results.md")
        )
    dataset_manifest: dict[str, object] | None = None
    if args.dataset == "funsd":
        if args.dataset_root is None:
            parser.error("--dataset-root is required with --dataset funsd")
        if args.pipeline == "ocr":
            parser.error("FUNSD evaluation requires --pipeline layoutlmv3 or --pipeline vlm")
        adapter = FunsdAdapter(args.dataset_root)
        annotations = adapter.load_annotations(args.split)
        dataset_manifest = adapter.manifest(args.split)
        results = (
            _load_results(args.results)
            if args.results is not None
            else _run_pipeline(adapter, args.split, args.pipeline)
        )
    else:
        if args.annotations is None or args.results is None:
            parser.error("--annotations and --results are required without --dataset")
        annotations = _load_annotations(args.annotations)
        results = _load_results(args.results)
    if dataset_manifest is not None:
        report = evaluate_funsd_dataset(annotations, results, args.pipeline)
        report["dataset_manifest"] = dataset_manifest
        report["runtime"] = _runtime_metadata(args.pipeline)
    else:
        report = evaluate_dataset(annotations, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = _funsd_markdown(report) if dataset_manifest else _generic_markdown(report)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(markdown) + "\n", encoding="utf-8")


def _load_annotations(path: Path) -> list[Annotation]:
    return [
        Annotation.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))
    ]


def _load_results(path: Path | None) -> list[DocumentResult]:
    if path is None:
        raise ValueError("results path is required")
    return [
        DocumentResult.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))
    ]


def _run_pipeline(adapter: FunsdAdapter, split: str | None, pipeline: str) -> list[DocumentResult]:
    processor: Any
    if pipeline == "layoutlmv3":
        processor = LayoutLMv3FUNSDPipeline(
            checkpoint=settings.funsd_layout_checkpoint,
            processor_checkpoint=settings.funsd_layout_processor_checkpoint,
            device=settings.funsd_layout_device,
            max_sequence_length=settings.funsd_max_sequence_length,
            image_size=settings.funsd_image_size,
            confidence_threshold=settings.funsd_confidence_threshold,
            language=settings.ocr_language,
            tesseract_cmd=settings.tesseract_cmd,
        )
    elif pipeline == "vlm":
        processor = QwenVLM(
            settings.vlm_model_name,
            settings.vlm_device,
            max_new_tokens=settings.vlm_max_new_tokens,
            backend=settings.vlm_backend,
            base_url=settings.vlm_base_url,
            timeout_seconds=settings.vlm_timeout_seconds,
            context_size=settings.vlm_context_size,
        )
    else:
        processor = OCRLayoutPipeline(settings.ocr_language, settings.tesseract_cmd)
    documents = adapter.iter_documents(split)
    results: list[DocumentResult] = []
    for index, (annotation, image_path) in enumerate(documents, start=1):
        print(
            f"[{pipeline}] processing {index}/{len(documents)}: {annotation.document_id}",
            flush=True,
        )
        results.append(_process_document(processor, image_path, annotation.document_id, pipeline))
    return results


def _process_document(
    processor: Any, image_path: Path, document_id: str, pipeline: str
) -> DocumentResult:
    if pipeline == "vlm":
        return processor.process(image_path, document_id, task_mode="funsd")
    return processor.process(image_path, document_id)


def _generic_markdown(report: dict) -> list[str]:
    failures = [
        document
        for document in report["documents"]
        if document["fields"]["f1"] < 1 or document["tables"]["cell_accuracy"] < 1
    ]
    return [
        "# Evaluation Results",
        "",
        f"Documents evaluated: {report['document_count']}",
        "",
        f"Documents with field or table mismatches: {len(failures)}",
        "",
        "## Failure analysis",
        "",
        *(
            f"- `{item['document_id']}` ({item['document_type']}): "
            f"field F1={item['fields']['f1']:.3f}, "
            f"table cell accuracy={item['tables']['cell_accuracy']:.3f}"
            for item in failures
        ),
    ]


def _funsd_markdown(report: dict) -> list[str]:
    metrics = report["metrics"]
    formatting_errors = sum(
        1
        for document in report["documents"]
        if "VLM formatting error" in document["failure_categories"]
        and document["failure_categories"]["VLM formatting error"] > 0
    )
    formatting_error_rate = (
        formatting_errors / report["document_count"] if report["document_count"] else 0.0
    )
    runtime = report.get("runtime", {})
    runtime_lines = [
        f"- Python: {runtime['python_version']}",
        f"- OS: {runtime['os']}",
        f"- Machine: {runtime['machine']}",
    ]
    for key in (
        "checkpoint",
        "device",
        "image_size",
        "max_sequence_length",
        "tesseract_cmd",
        "backend",
        "model",
        "max_new_tokens",
    ):
        if key in runtime:
            runtime_lines.append(f"- {key}: {runtime[key]}")
    return [
        "# FUNSD Evaluation Results",
        "",
        f"Documents evaluated: {report['document_count']}",
        f"Valid documents: {report['valid_document_count']}",
        f"Documents with runtime errors: {report['error_document_count']}",
        f"Pipeline: {report['pipeline']}",
        f"VLM formatting error rate: {formatting_error_rate:.3f}",
        "",
        f"Entity precision: {_format_metric(metrics['entity_precision'])}",
        f"Entity recall: {_format_metric(metrics['entity_recall'])}",
        f"Entity F1: {_format_metric(metrics['entity_f1'])}",
        f"Localization F1: {_format_metric(metrics['localization_f1'])}",
        f"Exact text match: {_format_metric(metrics['exact_text_match'])}",
        "",
        "## Runtime",
        "",
        *runtime_lines,
        "",
        "## Failure categories",
        "",
        *(f"- {category}: {count}" for category, count in metrics["failure_categories"].items()),
        "",
        "## Runtime errors",
        "",
        *(
            f"- `{document['document_id']}`: {'; '.join(document['errors'])}"
            for document in report["documents"]
            if document["errors"]
        ),
        "",
        "Entity linking is preserved but not scored.",
    ]


def _format_metric(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "not available"


def _runtime_metadata(pipeline: str) -> dict[str, object]:
    metadata: dict[str, object] = {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "machine": platform.machine(),
    }
    if pipeline == "layoutlmv3":
        metadata["checkpoint"] = settings.funsd_layout_checkpoint
        metadata["processor_checkpoint"] = settings.funsd_layout_processor_checkpoint
        metadata["device"] = settings.funsd_layout_device
        metadata["image_size"] = settings.funsd_image_size
        metadata["max_sequence_length"] = settings.funsd_max_sequence_length
        metadata["tesseract_cmd"] = settings.tesseract_cmd
    elif pipeline == "vlm":
        metadata["backend"] = settings.vlm_backend
        metadata["model"] = settings.vlm_model_name
        metadata["device"] = settings.vlm_device
        metadata["max_new_tokens"] = settings.vlm_max_new_tokens
        metadata["context_size"] = settings.vlm_context_size
    return metadata


if __name__ == "__main__":
    main()
