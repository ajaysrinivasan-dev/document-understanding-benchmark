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

Processors are isolated behind a common `DocumentResult` contract. API and UI code do not own extraction logic. VLM weights are optional and external to the default container image; CPU-only tests use fakes and malformed-output fixtures.
