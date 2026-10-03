from src.extraction.tables import extract_tables
from src.layout.pipeline import group_words_into_lines
from src.ocr.pipeline import OCRLayoutPipeline, build_ocr_word
from src.schemas import BoundingBox, DocumentResult, OCRWord


def word(text: str, x: float, y: float, page: int = 1, width: float = 30) -> OCRWord:
    return OCRWord(
        text=text,
        confidence=95,
        page_number=page,
        bbox=BoundingBox(x0=x, y0=y, x1=x + width, y1=y + 10),
    )


def test_tesseract_bbox_conversion():
    result = build_ocr_word(
        {
            "text": ["Total"],
            "conf": ["96.5"],
            "left": [12],
            "top": [20],
            "width": [40],
            "height": [11],
        },
        0,
        1,
    )
    assert result is not None
    assert result.text == "Total"
    assert result.confidence == 96.5
    assert result.bbox is not None
    assert result.bbox.model_dump() == {"x0": 12, "y0": 20, "x1": 52, "y1": 31}


def test_empty_and_invalid_tesseract_rows_are_rejected_or_unboxed():
    empty = build_ocr_word(
        {"text": ["   "], "conf": ["95"], "left": [1], "top": [1], "width": [10], "height": [10]},
        0,
        1,
    )
    invalid_box = build_ocr_word(
        {"text": ["word"], "conf": ["-1"], "left": [1], "top": [1], "width": [0], "height": [10]},
        0,
        1,
    )
    assert empty is None
    assert invalid_box is not None
    assert invalid_box.confidence is None
    assert invalid_box.bbox is None


def test_line_grouping_orders_words_and_separates_pages():
    blocks = group_words_into_lines(
        [
            word("world", 70, 11),
            word("page-two", 5, 10, page=2),
            word("hello", 5, 10),
            word("next", 5, 40),
        ]
    )
    assert [(block.page_number, block.text) for block in blocks] == [
        (1, "hello world"),
        (1, "next"),
        (2, "page-two"),
    ]
    assert [item.text for item in blocks[0].words] == ["hello", "world"]
    assert blocks[0].bbox is not None


def test_table_candidate_extraction_is_deterministic():
    words = [
        word("Item", 10, 10),
        word("Qty", 100, 10),
        word("Pen", 10, 40),
        word("2", 100, 40),
        word("Pencil", 10, 70),
        word("3", 100, 70),
    ]
    first = extract_tables(words)
    second = extract_tables(list(reversed(words)))
    assert [table.model_dump() for table in first] == [table.model_dump() for table in second]
    assert first[0].columns == ["Item", "Qty"]
    assert first[0].rows == [["Pen", "2"], ["Pencil", "3"]]


def test_table_detector_returns_no_table_without_repeated_rows():
    assert extract_tables([word("only", 10, 10), word("one", 100, 10)]) == []


def test_existing_field_extraction_and_result_contract_remain_valid():
    fields = OCRLayoutPipeline._candidate_fields(["Invoice", "No:", "A-10", "Total", "$12.00"])
    assert fields == {"invoice_number": "A-10", "total": "12.00"}
    result = DocumentResult(document_id="doc-1", fields=fields, tables=extract_tables([]))
    assert result.document_id == "doc-1"
    assert result.tables == []
