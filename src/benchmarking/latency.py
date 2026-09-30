from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def measure_call(function: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return value, (time.perf_counter() - started) * 1000
