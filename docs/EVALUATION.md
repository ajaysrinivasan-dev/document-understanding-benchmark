# Evaluation

## Dataset

No real annotated dataset is included in this repository. Development fixtures, if added, are labeled under `data/samples/` and are not benchmark evidence. Put a versioned annotation manifest under `data/annotations/` and keep private source documents outside Git.

## Annotation format

Each JSON item has `document_id`, `document_type`, a `fields` object, and a `tables` array. Tables contain `columns`, `rows`, and optional `page_number`. Extra provenance belongs in `metadata`.

## Metrics

Fields report exact key/value agreement, normalized agreement, precision, recall, F1, and per-document/per-field breakdowns. Normalization collapses surrounding/repeated whitespace and case for text, parses common date formats to ISO dates, and removes currency/grouping punctuation from numeric values without changing significant digits.

Tables report detection agreement, ordered column and row agreement, and cell-level precision/recall/F1 and accuracy. Row and cell matching is order-sensitive by design; a future alignment experiment may add an explicitly documented alternative.

## Failure categories

The reporting layer should preserve missing fields, hallucinated fields, incorrect values, OCR errors, reading-order problems, table structure/cell errors, multi-page issues, low resolution, unusual layouts, handwriting, and malformed model formatting. The system must not hide these behind an aggregate score.

## Runtime measurements

`python scripts/run_benchmark.py --documents <directory>` records per-document latency and environment metadata. CPU, RAM, GPU, model load time, and throughput should be added by the deployment environment when available; this repository never invents those measurements. Record Python, OS, hardware, model name/configuration, dataset version, and pipeline with each experiment.
