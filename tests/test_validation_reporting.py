from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from plate_dataset.config import BuildConfig
from plate_dataset.manifests import sha256_file
from plate_dataset.reporting import make_contact_sheets, write_statistics
from plate_dataset.validate import validate_dataset


def _config(root: Path) -> BuildConfig:
    return BuildConfig(
        workspace=root,
        seed=20260814,
        target_images=10,
        max_images=10,
        mh_share=(0.60, 0.70),
        split=(0.80, 0.10, 0.10),
        negative_share=(0.10, 0.10),
        training_imgsz=512,
        min_box_at_training_size=(8, 4),
    )


def _write_fixture(root: Path) -> Path:
    rows: list[dict[str, str]] = []
    split_names = ["train"] * 8 + ["val", "test"]
    for index, split in enumerate(split_names):
        output_id = f"item-{index:02d}"
        image = root / "detection" / "images" / split / f"{output_id}.jpg"
        label = root / "detection" / "labels" / split / f"{output_id}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (200, 100), (30 + index * 10, 60, 90)).save(image)
        negative = index == 0
        label.write_text("" if negative else "0 0.5 0.5 0.30 0.20\n", encoding="utf-8")
        rows.append(
            {
                "output_id": output_id,
                "split": split,
                "origin": "real" if index < 4 or split != "train" else "synthetic",
                "source_id": "fixture",
                "source_family": f"family-{index}",
                "image_path": image.relative_to(root).as_posix(),
                "label_path": label.relative_to(root).as_posix(),
                "image_sha256": sha256_file(image),
                "label_sha256": sha256_file(label),
                "negative": str(negative).lower(),
                "state": "MH" if index not in {4, 8, 9} else "DL",
                "plate_text": "" if negative else f"MH12AB{index:04d}",
                "vehicle_type": "car",
                "viewpoint": "front" if index % 2 else "rear",
                "plate_style": "private",
                "plate_layout": "double" if index in {2, 6} else "single",
                "effect": "night" if index in {1, 5} else "day",
                "ocr_eligible": str(not negative).lower(),
            }
        )
    manifest = root / "metadata" / "generation_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_validator_reports_invalid_yolo_box(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)
    label = root / "detection" / "labels" / "train" / "item-01.txt"
    label.write_text("0 1.2 0.5 0.4 0.2\n", encoding="utf-8")

    report = validate_dataset(root, _config(root))

    assert any(issue.code == "coordinate_out_of_range" for issue in report.issues)


def test_validator_reports_cross_split_family(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)
    manifest = root / "metadata" / "generation_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[-1]["source_family"] = rows[0]["source_family"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = validate_dataset(root, _config(root))

    assert any(issue.code == "cross_split_family" for issue in report.issues)


def test_validator_reports_checksum_mismatch(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)
    image = root / "detection" / "images" / "train" / "item-02.jpg"
    Image.new("RGB", (200, 100), "red").save(image)

    report = validate_dataset(root, _config(root))

    assert any(issue.code == "checksum_mismatch" for issue in report.issues)


def test_clean_fixture_passes_and_writes_reports(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)

    report = validate_dataset(root, _config(root))
    statistics = root / "metadata" / "dataset_statistics.json"
    sheets = make_contact_sheets(root, report, samples_per_sheet=4)
    write_statistics(report, statistics)

    assert report.error_count == 0
    assert report.image_count == report.label_count == 10
    assert report.split_counts == {"train": 8, "val": 1, "test": 1}
    assert sheets and all(path.is_file() for path in sheets)
    assert statistics.is_file()
