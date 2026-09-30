from src.layout.base import LayoutBlock
from src.schemas import OCRWord


def group_words_into_lines(words: list[OCRWord]) -> list[LayoutBlock]:
    grouped: dict[int, list[OCRWord]] = {}
    for word in words:
        grouped.setdefault(word.page_number, []).append(word)
    return [
        LayoutBlock(text=" ".join(word.text for word in page_words), page_number=page)
        for page, page_words in sorted(grouped.items())
    ]
