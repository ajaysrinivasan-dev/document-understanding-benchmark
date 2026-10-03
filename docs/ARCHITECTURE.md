# Architecture

```mermaid
flowchart TD
  D[Document] --> P[Preprocessing]
  P --> O[OCR + Layout Pipeline]
  P --> V[VLM Pipeline]
  O --> S[Common Extraction Schema]
  V --> S
  S --> N[Deterministic Normalization]
  N --> E[Evaluation]
  E --> R[Reports]
  U[Streamlit UI] --> A[FastAPI]
  A --> O
  A --> V
```

## FUNSD experiment

```mermaid
flowchart LR
  I[FUNSD test image] --> T[Tesseract OCR]
  T --> W[Words + pixel boxes]
  W --> L[LayoutLMv3 token classification]
  L --> B[BIO/subword aggregation]
  B --> E1[FUNSD entities]
  I --> Q[Qwen FUNSD prompt]
  Q --> J[Strict JSON validation]
  J --> E2[FUNSD entities]
  E1 --> M[Shared entity evaluator]
  E2 --> M
  G[FUNSD annotations] --> M
```

Processors are isolated behind a common `DocumentResult` contract. FUNSD entities are carried in `DocumentResult.metadata["funsd_entities"]`; generic fields and tables remain separate. Both pipelines use the official test split and are evaluated by the same entity matcher. VLM weights and LayoutLMv3 checkpoints are optional and external to the default container image; CPU-only tests use fakes and mocked outputs.
