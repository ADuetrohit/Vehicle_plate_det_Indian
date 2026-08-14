from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image
import pytest

from plate_dataset.config import BuildConfig
from plate_dataset.manifests import sha256_file
from plate_dataset.reporting import _caption, make_contact_sheets, write_statistics
from plate_dataset import reporting
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


@dataclass(frozen=True)
class SyntheticFixture:
    root: Path
    config: BuildConfig
    positive_crop: Path


def _manifest_rows(root: Path) -> list[dict[str, str]]:
    with (root / "metadata" / "generation_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return list(csv.DictReader(handle))


def _write_manifest_rows(root: Path, rows: list[dict[str, str]]) -> None:
    manifest = root / "metadata" / "generation_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def synthetic_fixture(tmp_path: Path) -> SyntheticFixture:
    root = tmp_path / "synthetic"
    config = replace(_config(root), synthetic_only=True)
    rows: list[dict[str, str]] = []
    specs = (
        ("train", True, "", "none", "removed", "night"),
        ("train", False, "MH", "single", "private", "day"),
        ("train", False, "MH", "double", "private", "night"),
        ("train", False, "MH", "single", "private", "night"),
        ("train", False, "MH", "double", "private", "rain"),
        ("train", False, "MH", "single", "private", "rain"),
        ("train", False, "DL", "single", "private", "day"),
        ("train", False, "DL", "single", "private", "day"),
        ("val", False, "DL", "single", "private", "day"),
        ("test", False, "MH", "single", "private", "day"),
    )
    positive_crop: Path | None = None
    for index, (split, negative, state, layout, style, condition) in enumerate(specs):
        output_id = f"syn-{index:02d}"
        image = root / "detection" / "images" / split / f"{output_id}.jpg"
        label = root / "detection" / "labels" / split / f"{output_id}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (200, 100), (30 + index * 10, 60, 90)).save(image)
        label.write_text("" if negative else "0 0.5 0.5 0.30 0.20\n", encoding="utf-8")
        crop = root / "ocr" / "images" / split / f"{output_id}.jpg"
        ocr_path = ""
        ocr_sha256 = ""
        if not negative:
            crop.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (256, 128), (80, 80, 80)).save(crop)
            ocr_path = crop.relative_to(root).as_posix()
            ocr_sha256 = sha256_file(crop)
            positive_crop = positive_crop or crop
        rows.append(
            {
                "output_id": output_id,
                "split": split,
                "origin": "synthetic",
                "source_id": f"source-{index}",
                "source_family": f"family-{index}",
                "image_path": image.relative_to(root).as_posix(),
                "label_path": label.relative_to(root).as_posix(),
                "image_sha256": sha256_file(image),
                "label_sha256": sha256_file(label),
                "negative": str(negative).lower(),
                "state": state,
                "plate_text": "" if negative else f"MH12AB{index:04d}",
                "vehicle_type": "car",
                "viewpoint": "front",
                "plate_style": style,
                "plate_layout": layout,
                "effect": condition,
                "ocr_eligible": str(not negative).lower(),
                "ocr_path": ocr_path,
                "ocr_sha256": ocr_sha256,
            }
        )
    _write_manifest_rows(root, rows)
    assert positive_crop is not None
    return SyntheticFixture(root, config, positive_crop)


def _first_row(rows: list[dict[str, str]], **wanted: str) -> dict[str, str]:
    return next(row for row in rows if all(row[key] == value for key, value in wanted.items()))


def test_synthetic_mode_does_not_require_real_holdout(synthetic_fixture: SyntheticFixture) -> None:
    """Catches synthetic validation mistakenly applying the real holdout gate."""
    report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)

    assert report.error_count == 0
    assert not any(issue.code == "real_holdout_share" for issue in report.issues)


def test_validator_requires_linked_crop_for_every_positive(synthetic_fixture: SyntheticFixture) -> None:
    """Catches a positive manifest row whose linked OCR crop was removed."""
    synthetic_fixture.positive_crop.unlink()

    report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)

    assert any(issue.code == "missing_ocr_crop" for issue in report.issues)


def test_validator_requires_linked_crop_checksum(synthetic_fixture: SyntheticFixture) -> None:
    """Catches a positive OCR crop that no longer matches its manifest checksum."""
    rows = _manifest_rows(synthetic_fixture.root)
    _first_row(rows, negative="false")["ocr_sha256"] = "incorrect"
    _write_manifest_rows(synthetic_fixture.root, rows)

    report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)

    assert any(issue.code == "ocr_checksum_mismatch" for issue in report.issues)


def test_validator_rejects_noncanonical_duplicate_ocr_crop_links(
    synthetic_fixture: SyntheticFixture,
) -> None:
    """Catches two positives sharing one otherwise checksum-valid OCR crop."""
    rows = _manifest_rows(synthetic_fixture.root)
    first, second = [row for row in rows if row["negative"] == "false"][:2]
    second["ocr_path"] = first["ocr_path"]
    second["ocr_sha256"] = first["ocr_sha256"]
    _write_manifest_rows(synthetic_fixture.root, rows)

    report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)

    assert any(issue.code == "invalid_ocr_path" for issue in report.issues)
    assert any(issue.code == "duplicate_ocr_crop" for issue in report.issues)


def test_validator_rejects_manifest_row_with_unknown_split(
    synthetic_fixture: SyntheticFixture,
) -> None:
    """Catches manifest rows silently omitted from the synthetic split quotas."""
    rows = _manifest_rows(synthetic_fixture.root)
    rows[0]["split"] = "unknown"
    _write_manifest_rows(synthetic_fixture.root, rows)

    report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)

    assert any(issue.code == "invalid_manifest_split" for issue in report.issues)


def test_validator_rejects_manifest_paths_outside_expected_pair(
    synthetic_fixture: SyntheticFixture,
) -> None:
    """Catches a manifest row whose stem matches but its declared pair path does not."""
    rows = _manifest_rows(synthetic_fixture.root)
    _first_row(rows, negative="false")["image_path"] = "other/syn-01.jpg"
    _write_manifest_rows(synthetic_fixture.root, rows)

    report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)

    assert any(issue.code == "invalid_manifest_pair_path" for issue in report.issues)


def test_validator_checks_ocr_for_positive_manifest_orphan(
    synthetic_fixture: SyntheticFixture,
) -> None:
    """Catches an orphaned positive row even when its missing crop has no file pair loop."""
    rows = _manifest_rows(synthetic_fixture.root)
    row = _first_row(rows, negative="false")
    (synthetic_fixture.root / row["image_path"]).unlink()
    (synthetic_fixture.root / row["label_path"]).unlink()
    (synthetic_fixture.root / row["ocr_path"]).unlink()

    report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)

    assert any(issue.code == "missing_manifest_pair" for issue in report.issues)
    assert any(issue.code == "missing_ocr_crop" for issue in report.issues)


def test_validator_rejects_duplicate_manifest_output_ids(
    synthetic_fixture: SyntheticFixture,
) -> None:
    """Catches duplicate manifest output IDs before they overwrite the pair lookup."""
    rows = _manifest_rows(synthetic_fixture.root)
    rows.append(rows[0].copy())
    _write_manifest_rows(synthetic_fixture.root, rows)

    report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)

    assert any(issue.code == "duplicate_manifest_output_id" for issue in report.issues)
    assert any(issue.code == "duplicate_manifest_pair" for issue in report.issues)


def test_validator_requires_default_ocr_crop_floor(synthetic_fixture: SyntheticFixture) -> None:
    """Catches a nominal 50,000-image build with fewer than 46,250 valid OCR crops."""
    default_config = replace(
        synthetic_fixture.config,
        target_images=50_000,
        max_images=50_000,
        negative_share=(0.075, 0.075),
    )

    report = validate_dataset(synthetic_fixture.root, default_config)

    assert any(issue.code == "ocr_crop_count_mismatch" for issue in report.issues)


def test_validator_enforces_exact_negative_quota(synthetic_fixture: SyntheticFixture) -> None:
    """Catches a positive label emptied into an unplanned hard negative."""
    label = synthetic_fixture.root / "detection" / "labels" / "train" / "syn-01.txt"
    label.write_text("", encoding="utf-8")

    report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)

    assert any(issue.code == "negative_count_mismatch" for issue in report.issues)


def test_validator_requires_synthetic_manifest_origins(synthetic_fixture: SyntheticFixture) -> None:
    """Catches a real-origin row admitted to a synthetic-only dataset."""
    rows = _manifest_rows(synthetic_fixture.root)
    rows[0]["origin"] = "real"
    _write_manifest_rows(synthetic_fixture.root, rows)

    report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)

    assert any(issue.code == "invalid_synthetic_origin" for issue in report.issues)


def test_validator_enforces_mh_positive_quota(synthetic_fixture: SyntheticFixture) -> None:
    """Catches Maharashtra-positive counts that differ from the planned split quota."""
    rows = _manifest_rows(synthetic_fixture.root)
    _first_row(rows, split="train", state="MH")["state"] = "DL"
    _write_manifest_rows(synthetic_fixture.root, rows)

    report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)

    assert any(issue.code == "mh_positive_count_mismatch" for issue in report.issues)


def test_validator_enforces_double_layout_quota(synthetic_fixture: SyntheticFixture) -> None:
    """Catches double-row plate counts that differ from the planned split quota."""
    rows = _manifest_rows(synthetic_fixture.root)
    _first_row(rows, split="train", plate_layout="double")["plate_layout"] = "single"
    _write_manifest_rows(synthetic_fixture.root, rows)

    report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)

    assert any(issue.code == "double_row_count_mismatch" for issue in report.issues)


def test_validator_enforces_camera_condition_quota(synthetic_fixture: SyntheticFixture) -> None:
    """Catches low-light rows reassigned outside their planned per-split quota."""
    rows = _manifest_rows(synthetic_fixture.root)
    _first_row(rows, split="train", effect="night")["effect"] = "day"
    _write_manifest_rows(synthetic_fixture.root, rows)

    report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)

    assert any(issue.code == "low_light_count_mismatch" for issue in report.issues)


def test_validator_enforces_adverse_condition_quota(synthetic_fixture: SyntheticFixture) -> None:
    """Catches rainy rows reassigned outside their planned per-split quota."""
    rows = _manifest_rows(synthetic_fixture.root)
    _first_row(rows, split="train", effect="rain")["effect"] = "day"
    _write_manifest_rows(synthetic_fixture.root, rows)

    report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)

    assert any(issue.code == "adverse_condition_count_mismatch" for issue in report.issues)


def test_contact_sheet_caption_uses_safe_sample_metadata() -> None:
    """Catches captions that reveal a registration number instead of safe QA metadata."""
    caption = _caption(
        {
            "output_id": "syn-42",
            "plate_style": "commercial",
            "effect": "rain",
            "plate_text": "MH12AB9999",
        }
    )

    assert caption == "syn-42 | commercial | rain"
    assert "MH12AB9999" not in caption


def test_contact_sheet_caption_lines_keep_condition_for_long_output_id() -> None:
    """Catches the condition being clipped when a QA sample has a long output ID."""
    output_id = "synthetic-sample-" + "x" * 80
    lines = getattr(reporting, "_caption_lines", lambda _: ())(
        {
            "output_id": output_id,
            "plate_style": "commercial",
            "effect": "rain",
            "plate_text": "MH12AB9999",
        }
    )

    assert "".join(lines[:-2]) == output_id
    assert lines[-2:] == ("commercial", "rain")
    assert all("MH12AB9999" not in line for line in lines)


def test_contact_sheets_group_by_split_style_layout_and_condition(
    synthetic_fixture: SyntheticFixture,
) -> None:
    """Catches QA sheets that mix distinct synthetic plate strata together."""
    report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)

    sheets = make_contact_sheets(synthetic_fixture.root, report, samples_per_sheet=4)

    assert {path.name for path in sheets} >= {
        "train-private-double-night.jpg",
        "train-private-single-rain.jpg",
        "val-private-single-day.jpg",
        "test-private-single-day.jpg",
    }


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
