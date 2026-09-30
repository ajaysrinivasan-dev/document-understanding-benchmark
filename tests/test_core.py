from src.evaluation.metrics import field_metrics, table_metrics
from src.extraction.normalization import normalize_date, normalize_number, normalize_text
from src.schemas import Table
from src.vlm.pipeline import parse_json_object


def test_normalization_is_deterministic():
    assert normalize_text("  Vendor   Name ") == "vendor name"
    assert normalize_date("03/04/2024") == "2024-04-03"
    assert normalize_number("$1,200.00") == "1200"


def test_malformed_vlm_json_fails_safely():
    try:
        parse_json_object("not json")
    except ValueError as error:
        assert "malformed" in str(error)
    else:
        raise AssertionError("malformed JSON was accepted")


def test_field_and_table_metrics():
    fields = field_metrics({"total": "$10.00", "name": "Acme"}, {"total": "10", "name": "acme"})
    assert fields["f1"] == 1.0
    tables = table_metrics(
        [{"columns": ["item"], "rows": [["Pen"]]}], [{"columns": ["ITEM"], "rows": [["pen"]]}]
    )
    assert tables["cell_accuracy"] == 1.0


def test_table_schema_defaults():
    assert Table().rows == []
