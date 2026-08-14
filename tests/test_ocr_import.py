from __future__ import annotations

import csv
import io
from pathlib import Path
import zipfile

from PIL import Image

from plate_dataset.ocr_import import import_existing_ocr, write_ocr_labels


def _jpeg_bytes(color: str) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (128, 64), color=color).save(stream, format="JPEG")
    return stream.getvalue()


def _csv_bytes(fieldnames: list[str], rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _make_ocr_archive(
    path: Path,
    accepted: dict[str, str],
    train: set[str],
    val: set[str],
    *,
    rejected: set[str] | None = None,
) -> Path:
    rejected = rejected or set()
    all_rows = [
        {
            "image_path": f"crops/{name}",
            "plate_text": text,
            "reviewed": "True",
            "rejected": str(name in rejected),
        }
        for name, text in accepted.items()
    ]
    row_by_name = {Path(row["image_path"]).name: row for row in all_rows}
    with zipfile.ZipFile(path, "w") as archive:
        for index, name in enumerate(accepted):
            archive.writestr(f"license_plate_dataset/crops/{name}", _jpeg_bytes(["red", "green", "blue"][index % 3]))
        fields = ["image_path", "plate_text", "reviewed", "rejected"]
        archive.writestr(
            "license_plate_dataset/annotations.csv", _csv_bytes(fields, all_rows)
        )
        archive.writestr(
            "license_plate_dataset/train_annotations.csv",
            _csv_bytes(fields, [row_by_name[name] for name in sorted(train)]),
        )
        archive.writestr(
            "license_plate_dataset/val_annotations.csv",
            _csv_bytes(fields, [row_by_name[name] for name in sorted(val)]),
        )
    return path


def test_import_preserves_splits_and_assigns_orphan_to_smaller_split(tmp_path: Path) -> None:
    """Catches the accepted crop missing from the supplied split CSVs being lost."""
    archive = _make_ocr_archive(
        tmp_path / "ocr.zip",
        {"a.jpg": "MH12AB1234", "b.jpg": "MH14CD5678", "c.jpg": "DL01AA0001"},
        train={"a.jpg", "b.jpg"},
        val=set(),
    )

    records = import_existing_ocr(archive, tmp_path / "out")

    assert len(records) == 3
    assert next(x for x in records if x.image_name == "a.jpg").split == "train"
    orphan = next(x for x in records if x.image_name == "c.jpg")
    assert orphan.split == "val"
    assert orphan.reconciliation == "assigned_orphan"
    assert all(record.image_path.exists() for record in records)


def test_import_excludes_rejected_and_empty_text_rows(tmp_path: Path) -> None:
    """Catches unusable OCR ground truth entering training labels."""
    archive = _make_ocr_archive(
        tmp_path / "ocr.zip",
        {"good.jpg": "MH12AB1234", "rejected.jpg": "MH14CD5678", "empty.jpg": ""},
        train={"good.jpg", "rejected.jpg", "empty.jpg"},
        val=set(),
        rejected={"rejected.jpg"},
    )

    records = import_existing_ocr(archive, tmp_path / "out")

    assert [record.image_name for record in records] == ["good.jpg"]


def test_import_normalizes_spacing_without_changing_characters(tmp_path: Path) -> None:
    """Catches spaces and punctuation leaking into the 37-symbol OCR vocabulary."""
    archive = _make_ocr_archive(
        tmp_path / "ocr.zip",
        {"a.jpg": "mh 12-ab 1234"},
        train={"a.jpg"},
        val=set(),
    )

    records = import_existing_ocr(archive, tmp_path / "out")

    assert records[0].plate_text == "MH12AB1234"


def test_write_ocr_labels_has_stable_schema(tmp_path: Path) -> None:
    """Catches OCR metadata columns changing between runs."""
    archive = _make_ocr_archive(
        tmp_path / "ocr.zip",
        {"a.jpg": "MH12AB1234"},
        train={"a.jpg"},
        val=set(),
    )
    records = import_existing_ocr(archive, tmp_path / "out")

    labels = tmp_path / "out" / "labels.csv"
    write_ocr_labels(records, labels)
    with labels.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == [
        "image_name",
        "image_path",
        "plate_text",
        "split",
        "source_id",
        "synthetic",
        "reconciliation",
    ]
    assert rows[0]["plate_text"] == "MH12AB1234"
    assert rows[0]["synthetic"] == "false"


def test_reimport_reuses_same_output_name(tmp_path: Path) -> None:
    """Catches resumable imports creating suffixed duplicate crop files."""
    archive = _make_ocr_archive(
        tmp_path / "ocr.zip",
        {"a.jpg": "MH12AB1234"},
        train={"a.jpg"},
        val=set(),
    )
    output = tmp_path / "out"

    first = import_existing_ocr(archive, output)
    second = import_existing_ocr(archive, output)

    assert second[0].image_name == first[0].image_name == "a.jpg"
    assert list((output / "images" / "train").glob("*.jpg")) == [first[0].image_path]
