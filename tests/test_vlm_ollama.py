from __future__ import annotations

import json

from PIL import Image

from src.datasets.funsd_types import FunsdPrediction
from src.vlm.pipeline import QwenVLM, VLMUnavailableError


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_ollama_backend_validates_installed_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeResponse({"models": [{"name": "qwen2.5vl:3b"}]}),
    )
    model = QwenVLM("qwen2.5vl:3b", backend="ollama")
    model._load()
    assert model._model is True


def test_ollama_backend_rejects_missing_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeResponse({"models": [{"name": "other"}]}),
    )
    model = QwenVLM("qwen2.5vl:3b", backend="ollama")
    try:
        model._load()
    except VLMUnavailableError as error:
        assert "not installed" in str(error)
    else:
        raise AssertionError("missing Ollama model was not rejected")


def test_ollama_funsd_request_uses_schema_and_image(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "message": {
                    "content": (
                        '{"entities": [{"label": "header", "text": "Title", '
                        '"bbox": [0, 0, 10, 10], "links": []}]}'
                    )
                }
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    image = Image.new("RGB", (8, 8))
    model = QwenVLM("qwen2.5vl:3b", backend="ollama")
    model._task_mode = "funsd"
    raw = model._infer_ollama([image], "prompt")

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "qwen2.5vl:3b"
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0
    assert body["format"] == FunsdPrediction.model_json_schema()
    assert isinstance(body["messages"][0]["images"][0], str)
    assert len(body["messages"][0]["images"][0]) > 10
    assert json.loads(raw)["entities"][0]["label"] == "header"
