# FUNSD Failure Analysis

These observations are derived from the recorded local FUNSD testing-split runs. The report uses document IDs and aggregate counts only; document contents are intentionally omitted.

## LayoutLMv3 + Tesseract baseline

50/50 documents completed without runtime errors.

The largest recorded failure categories were:
- incorrect text: 1069
- missed entity: 445
- question/answer confusion: 178
- incorrect entity boundary: 67
- incorrect entity label: 55

A diagnostic run on document 82092117 found 199 Tesseract words versus 223 source FUNSD words. The LayoutLMv3 model produced non-O predictions from the OCR input, so the earlier all-O preprocessing problem is fixed. The word-count mismatch is evidence of OCR/tokenization divergence and is a plausible contributor to the baseline errors, but it is not by itself a proven causal diagnosis.

## Qwen2.5-VL 3B via Ollama

The recorded run completed 31/50 documents successfully. 19/50 documents (38%) failed closed with malformed or incomplete JSON.

Among the 31 valid outputs, the largest entity-level failure categories were:
- missed entity: 400
- incorrect text: 141
- question/answer confusion: 124
- incorrect entity label: 103
- incorrect entity boundary: 68

The formatting failures are reported separately and are not converted into synthetic entity misses.

Direct testing showed that some long FUNSD pages caused Qwen2.5-VL 3B to consume the full generation budget and repeat content instead of closing the JSON object. Increasing the local context budget from 4096 to 8192 did not provide a reliable fix on a difficult page when the generation budget was still exhausted. The implementation therefore keeps strict validation/fail-closed behavior rather than accepting partial or truncated JSON.

## Next experiment

The next model change should be isolated and measured on the same held-out FUNSD testing split. Do not tune thresholds against the same reported test results. Preserve the current LayoutLMv3 result as the baseline checkpoint and treat Qwen formatting reliability as a separate runtime-quality metric.
