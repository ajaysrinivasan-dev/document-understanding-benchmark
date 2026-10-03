from collections.abc import Iterable

from src.layout.base import LayoutBlock
from src.schemas import BoundingBox, OCRWord


def group_words_into_lines(words: Iterable[OCRWord]) -> list[LayoutBlock]:
    """Group words into deterministic lines using page-local vertical proximity."""
    page_words: dict[int, list[OCRWord]] = {}
    for word in words:
        if word.bbox is not None and word.bbox.x1 > word.bbox.x0 and word.bbox.y1 > word.bbox.y0:
            page_words.setdefault(word.page_number, []).append(word)

    blocks: list[LayoutBlock] = []
    for page_number, words_on_page in sorted(page_words.items()):
        lines: list[list[OCRWord]] = []
        for word in sorted(words_on_page, key=_reading_order):
            matching_line = _find_line(word, lines)
            if matching_line is None:
                lines.append([word])
            else:
                matching_line.append(word)
        for line in sorted(lines, key=_line_top):
            ordered_words = sorted(line, key=_word_left)
            blocks.append(
                LayoutBlock(
                    text=" ".join(word.text for word in ordered_words),
                    page_number=page_number,
                    bbox=_union_bbox(ordered_words),
                    words=ordered_words,
                )
            )
    return blocks


def _reading_order(word: OCRWord) -> tuple[float, float]:
    bbox = word.bbox
    assert bbox is not None
    return bbox.y0, bbox.x0


def _line_top(words: list[OCRWord]) -> float:
    return min(_box(word).y0 for word in words)


def _word_left(word: OCRWord) -> float:
    return _box(word).x0


def _find_line(word: OCRWord, lines: list[list[OCRWord]]) -> list[OCRWord] | None:
    bbox = word.bbox
    assert bbox is not None
    center = (bbox.y0 + bbox.y1) / 2
    best_line: list[OCRWord] | None = None
    best_distance = float("inf")
    for line in lines:
        boxes = [item.bbox for item in line if item.bbox is not None]
        line_center = sum((item.y0 + item.y1) / 2 for item in boxes) / len(boxes)
        tolerance = max(bbox.y1 - bbox.y0, *(item.y1 - item.y0 for item in boxes)) * 0.6
        distance = abs(center - line_center)
        if distance <= tolerance and distance < best_distance:
            best_line = line
            best_distance = distance
    return best_line


def _union_bbox(words: list[OCRWord]) -> BoundingBox:
    boxes = [_box(word) for word in words]
    return BoundingBox(
        x0=min(box.x0 for box in boxes),
        y0=min(box.y0 for box in boxes),
        x1=max(box.x1 for box in boxes),
        y1=max(box.y1 for box in boxes),
    )


def _box(word: OCRWord) -> BoundingBox:
    assert word.bbox is not None
    return word.bbox
