# Document Understanding Benchmark

A reproducible framework for comparing an OCR/layout baseline with a configurable vision-language document extraction pipeline. It supports invoice/form fields and tables through a shared validated schema, deterministic normalization, independent evaluation, latency measurement, and a FastAPI service.

## Status and scope

The repository contains no real dataset and no generated accuracy claims. OCR, PDF, UI, and VLM integrations are optional dependencies. The default tests use no GPU, model download, or private document. The Qwen adapter is configured through `VLM_MODEL_NAME`; model weights and runtime provisioning remain deployment concerns.

## Install

```powershell
py -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Install optional capabilities with `py -m pip install -e ".[ocr,ui]"` or `.[vlm]` as appropriate. Tesseract itself must also be installed for OCR.

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

The commands require real input files. They do not create fake scores. See [docs/EVALUATION.md](docs/EVALUATION.md) for annotation and metric rules.

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

## Architecture and limitations

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/EVALUATION.md](docs/EVALUATION.md), and [docs/SECURITY.md](docs/SECURITY.md). The OCR field/table extraction is intentionally a baseline heuristic, not a claim of general document intelligence. Multi-page rendering, table reconstruction, model-specific Qwen image message formatting, and resource counters should be validated against the chosen production runtime and real dataset before deployment claims.
