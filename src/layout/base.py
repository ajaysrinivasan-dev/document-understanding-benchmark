from dataclasses import dataclass, field

from src.schemas import BoundingBox


@dataclass
class LayoutBlock:
    text: str
    page_number: int
    bbox: BoundingBox | None = None
    block_type: str = "text"
    words: list[str] = field(default_factory=list)
