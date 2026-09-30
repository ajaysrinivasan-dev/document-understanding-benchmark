from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class OCRWord(BaseModel):
    text: str
    confidence: float | None = Field(default=None, ge=0, le=100)
    page_number: int = Field(ge=1)
    bbox: BoundingBox | None = None


class Table(BaseModel):
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    page_number: int | None = Field(default=None, ge=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


class DocumentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_type: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)
    tables: list[Table] = Field(default_factory=list)
    confidence: dict[str, float] = Field(default_factory=dict)
    page_references: dict[str, int] = Field(default_factory=dict)
    processing_time_ms: float | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    pipeline: str = "unknown"


class Annotation(BaseModel):
    document_id: str
    document_type: str
    fields: dict[str, Any] = Field(default_factory=dict)
    tables: list[Table] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractionResponse(BaseModel):
    document_id: str
    pipeline: str
    results: list[DocumentResult]
    processing_time_ms: float
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
