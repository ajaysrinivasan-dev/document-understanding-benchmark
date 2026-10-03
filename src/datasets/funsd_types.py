from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

FUNSD_LABELS = ("question", "answer", "header", "other")
FunsdLabel = Literal["question", "answer", "header", "other"]


class FunsdEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: int | str | None = None
    label: FunsdLabel
    text: str
    bbox: tuple[float, float, float, float] | None = None
    page_number: int = Field(default=1, ge=1)
    links: list[list[int]] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is not None and (
            value[0] < 0 or value[1] < 0 or value[2] < value[0] or value[3] < value[1]
        ):
            raise ValueError("bbox must be non-negative and ordered x0,y0,x1,y1")
        return value


class FunsdPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[FunsdEntity] = Field(default_factory=list)
