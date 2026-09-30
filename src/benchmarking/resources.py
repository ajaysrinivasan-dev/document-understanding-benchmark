from __future__ import annotations

import os
import platform
import sys


def environment_info() -> dict[str, str | None]:
    return {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "cpu": platform.processor() or None,
        "gpu": os.getenv("CUDA_VISIBLE_DEVICES"),
    }
