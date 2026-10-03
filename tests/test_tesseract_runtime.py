from __future__ import annotations

import sys
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from src.layout.funsd_pipeline import LayoutLMv3FUNSDPipeline
from src.ocr.pipeline import configure_tesseract


def test_tesseract_command_uses_configured_value():
    fake = SimpleNamespace(pytesseract=SimpleNamespace(tesseract_cmd=None))
    assert configure_tesseract(fake, r"C:\custom\tesseract.exe") == r"C:\custom\tesseract.exe"
    assert fake.pytesseract.tesseract_cmd == r"C:\custom\tesseract.exe"


def test_tesseract_command_falls_back_to_path_command():
    fake = SimpleNamespace(pytesseract=SimpleNamespace(tesseract_cmd=None))
    assert configure_tesseract(fake) == "tesseract"
    assert fake.pytesseract.tesseract_cmd == "tesseract"


def test_layoutlm_page_conversion_changes_grayscale_to_rgb(monkeypatch):
    image = Image.new("L", (20, 20), color=255)
    content = BytesIO()
    image.save(content, format="PNG")
    seen_modes: list[str] = []

    fake_pytesseract = SimpleNamespace(
        pytesseract=SimpleNamespace(tesseract_cmd=None),
        Output=SimpleNamespace(DICT="dict"),
    )
    fake_pytesseract.image_to_data = lambda image, lang, output_type: {
        "text": ["Name"],
        "conf": ["95"],
        "left": [1],
        "top": [1],
        "width": [10],
        "height": [10],
    }
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)
    pipeline = LayoutLMv3FUNSDPipeline("mock-checkpoint")

    def fake_predict(processed_image, words):
        seen_modes.append(processed_image.mode)
        return ["O"], [1.0]

    monkeypatch.setattr(pipeline, "_predict_words", fake_predict)
    pipeline._process_page(content.getvalue(), 1)

    assert seen_modes == ["RGB"]
    assert fake_pytesseract.pytesseract.tesseract_cmd == "tesseract"
