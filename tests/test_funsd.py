import json

import pytest

from src.datasets.funsd import FunsdAdapter


def make_funsd_root(tmp_path, payloads):
    annotation_dir = tmp_path / "training_data" / "annotations"
    image_dir = tmp_path / "training_data" / "images"
    annotation_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    for document_id, payload in payloads.items():
        (annotation_dir / f"{document_id}.json").write_text(json.dumps(payload), encoding="utf-8")
        (image_dir / f"{document_id}.png").write_bytes(b"synthetic fixture placeholder")
    return tmp_path


def test_funsd_mapping_preserves_labels_entities_and_id(tmp_path):
    root = make_funsd_root(
        tmp_path,
        {
            "form-1": {
                "form": [
                    {
                        "id": 1,
                        "text": "Name",
                        "label": "question",
                        "box": [1, 2, 30, 12],
                        "linking": [[1, 2]],
                        "words": [{"text": "Name", "box": [1, 2, 30, 12]}],
                    },
                    {"id": 2, "text": "Ada", "label": "answer", "box": [40, 2, 60, 12]},
                ]
            }
        },
    )
    annotation = FunsdAdapter(root).load_annotations("training")[0]

    assert annotation.document_id == "form-1"
    assert annotation.document_type == "funsd_form"
    assert annotation.fields == {"question": ["Name"], "answer": ["Ada"]}
    assert annotation.tables == []
    assert annotation.metadata["dataset"] == "FUNSD"
    assert annotation.metadata["entities"][0]["linking"] == [[1, 2]]


def test_funsd_conversion_is_deterministic_and_manifest_is_local(tmp_path):
    root = make_funsd_root(tmp_path, {"b": {"form": []}, "a": {"form": []}})
    adapter = FunsdAdapter(root)

    first = [item.model_dump() for item in adapter.load_annotations()]
    second = [item.model_dump() for item in adapter.load_annotations()]
    manifest = adapter.manifest()

    assert first == second
    assert [item["document_id"] for item in first] == ["a", "b"]
    assert manifest["document_count"] == 2
    assert manifest["splits"] == {"training": 2}
    assert manifest["local_path"] == str(root)


def test_missing_form_and_entities_are_empty(tmp_path):
    root = make_funsd_root(tmp_path, {"empty": {}})
    annotation = FunsdAdapter(root).load_annotations()[0]
    assert annotation.fields == {}
    assert annotation.metadata["entities"] == []
    assert annotation.tables == []


def test_invalid_annotation_is_rejected(tmp_path):
    root = make_funsd_root(tmp_path, {"bad": {"form": "not-a-list"}})
    with pytest.raises(ValueError, match="form.*list"):
        FunsdAdapter(root).load_annotations()


def test_invalid_entity_is_rejected(tmp_path):
    root = make_funsd_root(tmp_path, {"bad": {"form": ["not-an-object"]}})
    with pytest.raises(ValueError, match="entity.*object"):
        FunsdAdapter(root).load_annotations()
