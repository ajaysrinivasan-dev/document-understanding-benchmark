from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.schemas import DocumentResult

EXTRACTION_PROMPT = (
    "Extract the document into JSON only. Use this shape: "
    '{"document_type": string|null, "fields": object, '
    '"tables": [{"columns": [string], "rows": [[string]], '
    '"page_number": integer|null}]}. Do not add prose. '
    "Preserve unknown values as null or omit them; never guess."
)


class VLMUnavailableError(RuntimeError):
    pass


class QwenVLM:
    name = "vlm"

    def __init__(self, model_name: str, device: str = "auto", max_new_tokens: int = 2048) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as error:
            raise VLMUnavailableError("VLM dependencies are not installed") from error
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, device_map=self.device
        )

    def process(self, path: str | Path, document_id: str | None = None) -> DocumentResult:
        started = time.perf_counter()
        identifier = document_id or Path(path).stem
        result = DocumentResult(document_id=identifier, pipeline=self.name)
        try:
            self._load()
            raw = self._infer(path)
            payload = parse_json_object(raw)
            validated = DocumentResult.model_validate(
                {"document_id": identifier, "pipeline": self.name, **payload}
            )
            result = validated
        except (VLMUnavailableError, ValidationError, ValueError, RuntimeError) as error:
            result.errors.append(str(error))
        result.processing_time_ms = (time.perf_counter() - started) * 1000
        return result

    def _infer(self, path: str | Path) -> str:
        try:
            from PIL import Image

            from src.ingestion.image_loader import load_document_pages
        except ImportError as error:
            raise VLMUnavailableError("VLM image dependencies are not installed") from error
        import io

        images = [
            Image.open(io.BytesIO(content)).convert("RGB") for content in load_document_pages(path)
        ]
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": EXTRACTION_PROMPT}],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=images, padding=True, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated = generated[:, inputs.input_ids.shape[1] :]
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0]


def parse_json_object(raw: str) -> dict[str, Any]:
    candidate = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError("VLM returned malformed JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("VLM JSON result must be an object")
    return payload
