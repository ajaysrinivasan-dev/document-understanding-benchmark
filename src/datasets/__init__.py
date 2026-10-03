"""Dataset adapters for reproducible evaluation."""

from src.datasets.base import DatasetAdapter
from src.datasets.funsd import FunsdAdapter
from src.datasets.funsd_types import FunsdEntity, FunsdPrediction

__all__ = ["DatasetAdapter", "FunsdAdapter", "FunsdEntity", "FunsdPrediction"]
