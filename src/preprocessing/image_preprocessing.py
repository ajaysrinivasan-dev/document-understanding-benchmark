from __future__ import annotations

from io import BytesIO


def validate_image_bytes(content: bytes) -> None:
    try:
        from PIL import Image

        with Image.open(BytesIO(content)) as image:
            image.verify()
    except ImportError:
        return
    except Exception as error:
        raise ValueError("Invalid image content") from error
