# Document Understanding Benchmark

A reproducible framework for comparing an OCR/layout baseline with a configurable vision-language document extraction pipeline. It supports invoice/form fields and tables through a shared validated schema, deterministic normalization, independent evaluation, latency measurement, and a FastAPI service.

## Status and scope

No raw dataset files are committed. A local FUNSD copy may be configured under the ignored `data/funsd/` directory; generated full reports remain ignored, while the concise recorded benchmark summary is tracked in `docs/FUNSD_RESULTS.md`. OCR, PDF, UI, LayoutLMv3, and VLM integrations are optional dependencies. The default tests use no GPU, model download, or private document. Qwen supports a configurable `transformers` backend and an optional local Ollama backend through `VLM_BACKEND`; `VLM_CONTEXT_SIZE` defaults to 4096 for reproducible local runs and can be increased when the runtime has enough context budget. Model weights/runtime provisioning remain deployment concerns.

## Install

```powershell
py -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Install optional capabilities with `py -m pip install -e ".[ocr,ui]"`, `.[layout]`, or `.[vlm]` as appropriate. Tesseract itself must also be installed for OCR. Configure its executable with `TESSERACT_CMD`; the default is `tesseract`, which uses the process PATH. Windows may require an absolute path such as `C:\Program Files\Tesseract-OCR\tesseract.exe`, while Linux/macOS can use `tesseract` or another configured absolute path.

## Run API and UI

```powershell
uvicorn app.main:app --reload
streamlit run ui/streamlit_app.py
```

`GET /health` is model-independent. Upload a document with `POST /extract?pipeline=ocr`, `vlm`, or `both`; the response contains structured results, warnings, errors, and measured processing time.

## Evaluation and benchmark

```powershell
py scripts/run_evaluation.py --annotations data/annotations/annotations.json --results reports/predictions.json
py scripts/run_benchmark.py --documents data/raw --output reports/benchmark.json
```

The commands require real input files. They do not create fake scores. See [docs/EVALUATION.md](docs/EVALUATION.md) for annotation and metric rules. The recorded FUNSD comparison is summarized in [docs/FUNSD_RESULTS.md](docs/FUNSD_RESULTS.md).

## FUNSD dataset

FUNSD is a form-understanding dataset, not an invoice dataset. Its question, answer, header, and source-level other entities are preserved in the canonical `Annotation` representation, while original entity records, links, boxes, words, split, and source paths are preserved in metadata. For the benchmark, `other` follows the standard FUNSD token-classification convention and is treated as the `O`/non-entity class, so entity metrics cover question, answer, and header only. FUNSD annotations do not provide table ground truth, so the adapter returns `tables=[]`.

Dataset setup is explicit and never runs during evaluation:

```powershell
py scripts/setup_funsd.py --output data/funsd
py scripts/run_evaluation.py --dataset funsd --dataset-root data/funsd --split testing --pipeline layoutlmv3
py scripts/run_evaluation.py --dataset funsd --dataset-root data/funsd --split testing --pipeline vlm
```

FUNSD compares two implementations of the same entity task: the LayoutLMv3 token-classification baseline (`FUNSD_LAYOUT_CHECKPOINT`, default `nielsr/layoutlmv3-finetuned-funsd`) and Qwen's explicit FUNSD entity JSON mode. For local inference, set `VLM_BACKEND=ollama`, `VLM_MODEL_NAME=qwen2.5vl:3b`, and `VLM_BASE_URL=http://127.0.0.1:11434` after installing the model in Ollama; `VLM_CONTEXT_SIZE` controls the Ollama context budget for long structured outputs. The existing Transformers backend remains available for Hugging Face runtimes. The setup command downloads the public archive, checks ZIP integrity, optionally verifies `--sha256`, extracts the documented `training_data/` and `testing_data/` structure, and fails if the layout is not present. Raw FUNSD files and generated reports are excluded from Git.

## Quality checks

```powershell
pytest -q
ruff check .
ruff format --check .
```

## Docker

```powershell
docker compose up --build
```

The image runs the API as a non-root user and excludes datasets, reports, secrets, and model caches. VLM deployment needs a separately provisioned compatible runtime/GPU and model access.

## OCR baseline and limitations

The OCR baseline uses Tesseract OCR, preserves valid word bounding boxes and page numbers, groups words into deterministic geometric lines, applies heuristic field extraction, and performs heuristic table candidate extraction from repeated row and column alignment. It is an explainable baseline, not a learned document-layout or general table-understanding model. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/EVALUATION.md](docs/EVALUATION.md), and [docs/SECURITY.md](docs/SECURITY.md). Multi-page rendering, table reconstruction, model-specific Qwen image message formatting, and resource counters should be validated against the chosen production runtime and real dataset before deployment claims.

For FUNSD, invoice regexes are not used. LayoutLMv3 consumes Tesseract words and normalized 0-1000 boxes, aggregates subword logits back to OCR words, and reconstructs `B-/I-` entities for `question`, `answer`, `header`, and `other`. The model checkpoint and processor checkpoint are independently configurable through `FUNSD_LAYOUT_CHECKPOINT` and `FUNSD_LAYOUT_PROCESSOR_CHECKPOINT`; the default processor is `microsoft/layoutlmv3-base`. Qwen receives a separate FUNSD prompt and returns validated entities in original image-pixel coordinates. Neither path receives ground-truth annotations during inference.
