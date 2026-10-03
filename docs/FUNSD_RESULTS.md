# FUNSD Model Comparison

Benchmark: FUNSD testing split, 50 documents. Both pipelines use the same evaluator and entity-matching threshold (IoU >= 0.5). FUNSD entity metrics score question, answer, and header; source-level `other` is the non-entity/O class. No ground-truth annotations are passed to either pipeline during inference.

| Metric | LayoutLMv3 + Tesseract | Qwen2.5-VL 3B via Ollama |
|---|---:|---:|
| Documents evaluated | 50 | 50 |
| Valid documents | 50 | 31 |
| Runtime-error documents | 0 | 19 |
| Entity precision | 0.191 | 0.024 |
| Entity recall | 0.232 | 0.019 |
| Entity F1 | 0.210 | 0.021 |
| Localization F1 | 0.229 | 0.036 |
| Exact text match | 0.440 | 0.438 |
| p50 total latency (ms) | 1018.6 | 16465.8 |
| p95 total latency (ms) | 1913.0 | 24998.2 |
| Average total latency (ms) | 1197.4 | 17850.3 |
| VLM formatting-error rate | 0.000 | 0.380 |

## Per-label F1

| Label | LayoutLMv3 | Qwen2.5-VL 3B |
|---|---:|---:|
| question | 0.215 | 0.006 |
| answer | 0.198 | 0.026 |
| header | 0.248 | 0.051 |

## Interpretation

The LayoutLMv3 run completed all 50 documents without runtime errors. The Qwen2.5-VL run completed 31 documents successfully and 19 documents failed closed because the model response reached an invalid/incomplete JSON state. Qwen inference was therefore evaluated only on its 31 valid outputs for aggregate entity metrics, while the 19 formatting failures are reported separately rather than treated as entity misses.

The current experiment is a baseline comparison, not a model-quality claim. The Qwen path is constrained by the local Ollama runtime and the selected 3B checkpoint. The recorded Qwen run used a 4096-token Ollama context and 2048-token generation budget; 19 documents reached malformed/incomplete JSON under that configuration. The LayoutLMv3 path is constrained by Tesseract OCR quality and the model input pipeline; the recorded run uses the FUNSD-finetuned model checkpoint with the independent `microsoft/layoutlmv3-base` processor. The next improvement should target one identified failure mode at a time and be re-evaluated on the same held-out test split.

## Reproduction

LayoutLMv3:
```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
py scripts/run_evaluation.py --dataset funsd --dataset-root data/funsd --split testing --pipeline layoutlmv3
```

Qwen via Ollama:
```powershell
$env:VLM_BACKEND="ollama"
$env:VLM_MODEL_NAME="qwen2.5vl:3b"
$env:VLM_BASE_URL="http://127.0.0.1:11434"
$env:VLM_MAX_NEW_TOKENS="2048"
py scripts/run_evaluation.py --dataset funsd --dataset-root data/funsd --split testing --pipeline vlm
```

Generated local result files are kept under `reports/` and ignored by Git. This comparison is the concise, human-reviewable summary of the recorded local runs.
