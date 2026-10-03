from collections import Counter
from statistics import median

from src.layout.base import LayoutBlock
from src.layout.pipeline import group_words_into_lines
from src.schemas import BoundingBox, OCRWord, Table


def extract_tables(
    words: list[OCRWord], layout_blocks: list[LayoutBlock] | None = None
) -> list[Table]:
    """Extract table candidates using repeated geometric row and column alignment.

    This is heuristic table candidate extraction, not general table understanding.
    It returns no candidate when aligned multi-row evidence is insufficient.
    """
    blocks = layout_blocks if layout_blocks is not None else group_words_into_lines(words)
    candidates: list[Table] = []
    for page_number in sorted({block.page_number for block in blocks}):
        page_blocks = [block for block in blocks if block.page_number == page_number]
        row_candidates = [block.words for block in page_blocks if len(block.words) >= 2]
        if len(row_candidates) < 2:
            continue
        row_count = Counter(len(row) for row in row_candidates).most_common(1)[0]
        rows = [row for row in row_candidates if len(row) == row_count[0]]
        if len(rows) < 2 or row_count[0] < 2:
            continue
        anchors = [median(_box(row[index]).x0 for row in rows) for index in range(row_count[0])]
        tolerance = max(
            8.0, median(_box(word).y1 - _box(word).y0 for row in rows for word in row) * 2
        )
        if any(
            abs(_box(word).x0 - anchors[index]) > tolerance
            for row in rows
            for index, word in enumerate(row)
        ):
            continue
        columns = [word.text for word in rows[0]]
        data_rows = [[word.text for word in row] for row in rows[1:]]
        if not data_rows:
            continue
        confidence = min(1.0, 0.5 + 0.1 * len(rows) + 0.1 * row_count[0])
        candidates.append(
            Table(columns=columns, rows=data_rows, page_number=page_number, confidence=confidence)
        )
    return candidates


def _box(word: OCRWord) -> BoundingBox:
    assert word.bbox is not None
    return word.bbox
