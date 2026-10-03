from __future__ import annotations

import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.datasets.funsd_types import FunsdPrediction
from src.schemas import DocumentResult

EXTRACTION_PROMPT = (
    "Extract the document into JSON only. Use this shape: "
    '{"document_type": string|null, "fields": object, '
    '"tables": [{"columns": [string], "rows": [[string]], '
    '"page_number": integer|null}]}. Do not add prose. '
    "Preserve unknown values as null or omit them; never guess."
)
FUNSD_EXTRACTION_PROMPT = (
    "Extract FUNSD form entities as JSON only. Return exactly an object with an "
    '"entities" array. Each entity must have label set to one of question, answer, '
    "header, other; text; bbox as [x0,y0,x1,y1] in original image pixels; and links "
    "as an array of integer ID pairs when visible, otherwise []. Do not return invoice "
    "fields, tables, or prose."
)


class VLMUnavailableError(RuntimeError):
    pass


class QwenVLM:
    name = "vlm"

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 2048,
        backend: str = "transformers",
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 300.0,
        context_size: int = 4096,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.backend = backend
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.context_size = context_size
        self._model: Any = None
        self._processor: Any = None
        self._task_mode = "generic"

    def _load(self) -> None:
        if self._model is not None:
            return
        if self.backend == "ollama":
            self._load_ollama()
            self._model = True
            return
        if self.backend != "transformers":
            raise VLMUnavailableError(f"Unsupported VLM backend: {self.backend}")
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as error:
            raise VLMUnavailableError("VLM dependencies are not installed") from error
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, device_map=self.device
        )

    def _load_ollama(self) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/api/tags",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise VLMUnavailableError(f"Ollama server is unavailable at {self.base_url}") from error
        if not isinstance(payload, dict):
            raise VLMUnavailableError("Ollama returned an invalid model list")
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise VLMUnavailableError("Ollama returned an invalid model list")
        names = {str(model.get("name")) for model in models if isinstance(model, dict)}
        if self.model_name not in names:
            raise VLMUnavailableError(f"Ollama model '{self.model_name}' is not installed")

    def process(
        self,
        path: str | Path,
        document_id: str | None = None,
        task_mode: str = "generic",
    ) -> DocumentResult:
        started = time.perf_counter()
        identifier = document_id or Path(path).stem
        result = DocumentResult(
            document_id=identifier,
            document_type="funsd_form" if task_mode == "funsd" else None,
            pipeline=self.name,
            metadata={"funsd_entities": []} if task_mode == "funsd" else {},
        )
        self._task_mode = task_mode
        try:
            self._load()
            inference_started = time.perf_counter()
            raw = self._infer(
                path, FUNSD_EXTRACTION_PROMPT if task_mode == "funsd" else EXTRACTION_PROMPT
            )
            inference_time_ms = (time.perf_counter() - inference_started) * 1000
            if task_mode == "funsd":
                result.metadata["vlm_inference_time_ms"] = inference_time_ms
            payload = parse_json_object(raw)
            if task_mode == "funsd":
                validated = FunsdPrediction.model_validate(payload)
                result = DocumentResult(
                    document_id=identifier,
                    document_type="funsd_form",
                    pipeline=self.name,
                    metadata={
                        "funsd_entities": [entity.model_dump() for entity in validated.entities],
                        "coordinate_space": "original_image_pixels",
                        "vlm_inference_time_ms": inference_time_ms,
                    },
                )
            else:
                result = DocumentResult.model_validate(
                    {"document_id": identifier, "pipeline": self.name, **payload}
                )
        except (OSError, VLMUnavailableError, ValidationError, ValueError, RuntimeError) as error:
            result.errors.append(str(error))
        result.processing_time_ms = (time.perf_counter() - started) * 1000
        return result

    def _infer(self, path: str | Path, prompt: str = EXTRACTION_PROMPT) -> str:
        try:
            from PIL import Image

            from src.ingestion.image_loader import load_document_pages
        except ImportError as error:
            raise VLMUnavailableError("VLM image dependencies are not installed") from error
        import io

        images = [
            Image.open(io.BytesIO(content)).convert("RGB") for content in load_document_pages(path)
        ]
        if self.backend == "ollama":
            return self._infer_ollama(images, prompt)
        if self.backend != "transformers":
            raise VLMUnavailableError(f"Unsupported VLM backend: {self.backend}")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": prompt}],
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

    def _infer_ollama(self, images: list[Any], prompt: str) -> str:
        encoded_images = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            encoded_images.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt, "images": encoded_images}],
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": self.max_new_tokens,
                "num_ctx": self.context_size,
            },
        }
        if self._task_mode == "funsd":
            body["format"] = FunsdPrediction.model_json_schema()
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (
            OSError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as error:
            raise VLMUnavailableError("Ollama inference request failed") from error
        message = payload.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Ollama returned an invalid chat response")
        return message["content"]


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
