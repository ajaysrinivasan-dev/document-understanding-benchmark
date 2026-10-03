# Evaluation

## Dataset

No real annotated dataset is included in this repository. Development fixtures, if added, are labeled under `data/samples/` and are not benchmark evidence. Put a versioned annotation manifest under `data/annotations/` and keep private source documents outside Git.

## FUNSD adapter

FUNSD is used here for form understanding and semantic entity extraction. It is not an invoice dataset. The explicit setup command is:

```powershell
py scripts/setup_funsd.py --output data/funsd
```

The adapter expects the extracted public archive layout `training_data/annotations`, `training_data/images`, and optionally the corresponding `testing_data` directories. It does not download during evaluation. Use `py scripts/run_evaluation.py --dataset funsd --dataset-root data/funsd --split testing --pipeline layoutlmv3` or `--pipeline vlm` to load local annotations, run the selected existing pipeline, convert predictions to `DocumentResult`, and write JSON/Markdown reports. Supplying `--results` instead uses stored predictions and does not run inference.

Each FUNSD entity is mapped to a lower-case label key in `Annotation.fields`, with values kept as ordered lists of text, including `question`, `answer`, `header`, and source-level `other`. Original labels, IDs, text, boxes, links, words, split, source annotation path, and image path are retained in `Annotation.metadata`. Documents use `document_type="funsd_form"`. FUNSD does not provide direct table ground truth in this adapter, so `Annotation.tables` is always empty and no table metric is claimed for FUNSD. For the official LayoutLMv3-style token task, source-level `other` is represented as the `O` class and is therefore excluded from entity metrics; only question, answer, and header are benchmark entity labels.

## FUNSD task and fair comparison

The FUNSD experiment is entity extraction over the official train/test split. Invoice-specific regexes such as `invoice_number` and `total` are not used. The OCR+layout baseline is LayoutLMv3 token classification: Tesseract words and boxes are passed to the configured checkpoint, subword logits are aggregated back to OCR-word IDs, and BIO labels are reconstructed into entities. The default checkpoint is `nielsr/layoutlmv3-finetuned-funsd`; its model-card provenance should be recorded with each experiment, and it can be replaced with `FUNSD_LAYOUT_CHECKPOINT`. The processor is independently configured with `FUNSD_LAYOUT_PROCESSOR_CHECKPOINT` and defaults to `microsoft/layoutlmv3-base` to keep the image/tokenizer preprocessing contract explicit.

The Qwen pipeline has an explicit FUNSD mode that requests only `question`, `answer`, `header`, and `other` entities with original-image pixel boxes and optional links. Its JSON is validated with Pydantic and malformed output fails closed. Qwen can run through the Transformers backend or an optional local Ollama backend; the Ollama path sends RGB page images as base64 to `/api/chat` and requests schema-constrained JSON. Both paths return `DocumentResult` with `metadata["funsd_entities"]`, process the same images, and never read ground-truth annotations during inference.

Both FUNSD paths explicitly convert page images to RGB before model preprocessing. Tesseract is configured through `TESSERACT_CMD`, defaulting to `tesseract`; Windows deployments can provide an absolute executable path when subprocess PATH inheritance is unreliable, and Linux/macOS deployments can use `tesseract` or another configured absolute path.

## FUNSD metrics

Entity matching is one-to-one and greedy by descending box IoU. A prediction matches an expected entity for entity precision/recall/F1 only when the label agrees and IoU is at least 0.5. Localization metrics use the same IoU threshold without requiring the label to agree. Exact text match is computed among label-and-localization matches after whitespace/case normalization. Aggregate scores are micro-averaged over valid documents, with per-label precision/recall/F1 and document-level metrics included. Documents that fail before inference are reported separately and excluded from aggregate scores; metrics are `null` when no valid documents exist. Failure categories use safe document IDs and entity IDs only.

Entity linking is preserved from source annotations and accepted from Qwen output, but linking is not scored because the LayoutLMv3 baseline does not predict links reliably. FUNSD has no table ground truth in this adapter, so no table benchmark is reported. Full JSON/Markdown reports are generated locally and ignored by Git; the concise recorded comparison is maintained in `docs/FUNSD_RESULTS.md`.

FUNSD reports also record total latency, p50/p95 total latency, average Tesseract time, average LayoutLMv3 inference time, and average Qwen inference time when those components run successfully. Hardware, checkpoint, and runtime availability must be recorded with any published experiment.

## Recorded FUNSD comparison

The locally recorded LayoutLMv3 and Qwen/Ollama runs are summarized in [docs/FUNSD_RESULTS.md](FUNSD_RESULTS.md). The summary records the exact test population, validity counts, metrics, latency, runtime configuration, and Qwen formatting-failure rate without exposing document contents.

## LayoutLMv3 diagnostic

Use `py scripts/diagnose_funsd_layoutlmv3.py --document 82092117` for a one-document diagnostic when OCR-derived predictions are unexpectedly all `O`. It prints the configured checkpoint `id2label`, OCR words/boxes, FUNSD annotation words/boxes, normalized boxes, processor tensors, word-ID mappings, and top-three token probabilities for OCR-input and control-input runs. The control run uses FUNSD annotation word/text/box information as an oracle-style compatibility diagnostic only; it must never be used by the official evaluator or benchmark results. Official benchmarking continues to use only Tesseract-derived words and boxes and does not use ground-truth inputs.

## Annotation format

Each JSON item has `document_id`, `document_type`, a `fields` object, and a `tables` array. Tables contain `columns`, `rows`, and optional `page_number`. Extra provenance belongs in `metadata`.

## Metrics

Fields report exact key/value agreement, normalized agreement, precision, recall, F1, and per-document/per-field breakdowns. Normalization collapses surrounding/repeated whitespace and case for text, parses common date formats to ISO dates, and removes currency/grouping punctuation from numeric values without changing significant digits.

Tables report detection agreement, ordered column and row agreement, and cell-level precision/recall/F1 and accuracy. Row and cell matching is order-sensitive by design; a future alignment experiment may add an explicitly documented alternative.

## Failure categories

The reporting layer should preserve missing fields, hallucinated fields, incorrect values, OCR errors, reading-order problems, table structure/cell errors, multi-page issues, low resolution, unusual layouts, handwriting, and malformed model formatting. The system must not hide these behind an aggregate score.

## OCR baseline behavior

The baseline uses Tesseract OCR and preserves valid `left`, `top`, `width`, and `height` data as `OCRWord.bbox` values with page numbers. A deterministic geometric layout step groups words into approximate lines by vertical proximity and orders each line left to right. Candidate fields are extracted with small regular-expression heuristics. Tables use explicitly named **heuristic table candidate extraction**: repeated aligned rows and column starts are required, otherwise no table is returned. This is not a learned document-layout or general table-understanding model, and candidates require annotation-based evaluation.

## Runtime measurements

`python scripts/run_benchmark.py --documents <directory>` records per-document latency and environment metadata. CPU, RAM, GPU, model load time, and throughput should be added by the deployment environment when available; this repository never invents those measurements. Record Python, OS, hardware, model name/configuration, dataset version, and pipeline with each experiment.
