from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from plate_dataset.config import BuildConfig
from plate_dataset.manifests import MANIFEST_FIELDS, sha256_file, write_manifest
from plate_dataset.records import Box
from plate_dataset.storage import (
    InsufficientStorage,
    StorageEstimate,
    estimate_storage,
    require_storage,
)
from plate_dataset.ocr_export import export_ocr_crop, merge_ocr_labels


@pytest.fixture
def config(tmp_path: Path) -> BuildConfig:
    return BuildConfig(
        workspace=tmp_path,
        seed=1,
        target_images=50_000,
        max_images=50_000,
        mh_share=(0.60, 0.70),
        split=(0.80, 0.10, 0.10),
        negative_share=(0.075, 0.075),
        training_imgsz=512,
        min_box_at_training_size=(8, 4),
    )


@pytest.fixture
def sample_jpegs(tmp_path: Path) -> list[Path]:
    paths = []
    for index, size in enumerate(((1920, 1080), (640, 480))):
        path = tmp_path / f"source-{index}.jpg"
        Image.new("RGB", size, color=(30 + index, 80, 120)).save(path, quality=88)
        paths.append(path)
    return paths


@pytest.fixture
def scene() -> np.ndarray:
    image = np.full((100, 220, 3), 255, dtype=np.uint8)
    image[20:70, 20:180] = (20, 60, 100)
    return image


@pytest.fixture
def existing_labels(tmp_path: Path) -> Path:
    path = tmp_path / "existing.csv"
    rows = [
        {
            "image_name": "existing-a.jpg",
            "image_path": "images/train/existing-a.jpg",
            "plate_text": "MH12AB1234",
            "split": "train",
            "source_id": "existing_archive",
            "synthetic": "false",
            "reconciliation": "preserved",
        },
        {
            "image_name": "existing-b.jpg",
            "image_path": "images/val/existing-b.jpg",
            "plate_text": "DL01AA0001",
            "split": "val",
            "source_id": "existing_archive",
            "synthetic": "false",
            "reconciliation": "preserved",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def synthetic_row() -> dict[str, str]:
    return {
        "image_name": "syn-b.jpg",
        "image_path": "images/train/syn-b.jpg",
        "plate_text": "MH14CD5678",
        "split": "train",
        "source_id": "synthetic",
        "synthetic": "true",
        "reconciliation": "generated",
        "output_id": "syn-002",
    }


def test_storage_preflight_rejects_less_than_required_headroom() -> None:
    """Catches a build beginning when final output would consume its required reserve."""
    estimate = StorageEstimate(projected_bytes=10_000, reserve_bytes=5_000)

    with pytest.raises(InsufficientStorage, match="required_bytes=15000"):
        require_storage(estimate, free_bytes=14_999)


def test_storage_estimate_uses_measured_source_average(
    sample_jpegs: list[Path], config: BuildConfig
) -> None:
    """Catches a fixed output guess that ignores actual source JPEG density."""
    result = estimate_storage(config, sample_jpegs)

    assert result.projected_bytes > sum(path.stat().st_size for path in sample_jpegs)


def test_storage_estimate_scales_source_pixels_to_configured_scene_edge(
    sample_jpegs: list[Path], config: BuildConfig
) -> None:
    """Catches estimating full-size scenes after the configured edge limit is applied."""
    smaller = estimate_storage(config, sample_jpegs)
    larger = estimate_storage(replace(config, max_scene_edge=1_920), sample_jpegs)

    assert larger.projected_bytes > smaller.projected_bytes

def test_storage_estimate_reserves_crop_for_every_target_scene(
    sample_jpegs: list[Path], config: BuildConfig
) -> None:
    """Catches preflight shrinking OCR capacity from an assumed negative-rate midpoint."""
    all_positive = estimate_storage(config, sample_jpegs)
    mostly_negative = estimate_storage(
        replace(config, negative_share=(0.50, 0.50)), sample_jpegs
    )

    assert mostly_negative.projected_bytes == all_positive.projected_bytes


def test_exported_crop_is_256_by_128_and_checksum_valid(
    scene: np.ndarray, tmp_path: Path
) -> None:
    """Catches OCR crops that have an incorrect training canvas or stale checksum."""
    output = tmp_path / "crop.jpg"

    checksum = export_ocr_crop(
        scene, Box(0, 20, 20, 180, 70), output, (256, 128), 92
    )

    assert Image.open(output).size == (256, 128)
    assert checksum == sha256_file(output)


def test_export_preserves_aspect_ratio_and_pads_to_canvas(
    scene: np.ndarray, tmp_path: Path
) -> None:
    """Catches stretching an OCR plate crop instead of centering it on the canvas."""
    output = tmp_path / "crop.jpg"

    export_ocr_crop(scene, Box(0, 20, 20, 180, 70), output, (256, 128), 92)

    with Image.open(output) as crop:
        pixels = np.asarray(crop.convert("RGB"))
    assert pixels[0, 0].mean() < 200
    assert pixels[64, 128].mean() < 150


def test_merge_preserves_existing_and_adds_synthetic(
    existing_labels: Path, synthetic_row: dict[str, str], tmp_path: Path
) -> None:
    """Catches replacement of the imported OCR corpus when synthetic labels are merged."""
    output = tmp_path / "labels.csv"

    count = merge_ocr_labels(existing_labels, [synthetic_row], output)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert count == 3
    assert [row["image_name"] for row in rows] == [
        "existing-a.jpg",
        "existing-b.jpg",
        "syn-b.jpg",
    ]


def test_merge_sorts_new_rows_by_split_and_output_id(
    existing_labels: Path, synthetic_row: dict[str, str], tmp_path: Path
) -> None:
    """Catches nondeterministic synthetic OCR label order between resumed builds."""
    output = tmp_path / "labels.csv"
    first = dict(synthetic_row, image_name="syn-z.jpg", split="val", output_id="syn-010")
    second = dict(synthetic_row, image_name="syn-a.jpg", split="train", output_id="syn-001")

    merge_ocr_labels(existing_labels, [first, second], output)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["image_name"] for row in rows[-2:]] == ["syn-a.jpg", "syn-z.jpg"]


def test_merge_rejects_conflicting_synthetic_rows_with_same_stable_key(
    existing_labels: Path, synthetic_row: dict[str, str], tmp_path: Path
) -> None:
    """Catches a resumed build silently choosing one of two conflicting OCR labels."""
    conflict = dict(synthetic_row, plate_text="KA01ZZ9999")

    with pytest.raises(ValueError, match="conflicting synthetic OCR rows.*synthetic:syn-b.jpg"):
        merge_ocr_labels(existing_labels, [synthetic_row, conflict], tmp_path / "labels.csv")

def test_manifest_writes_ocr_linkage_columns(tmp_path: Path) -> None:
    """Catches crop paths or checksums being silently discarded from generation metadata."""
    output = tmp_path / "generation_manifest.csv"

    write_manifest(
        [{"output_id": "syn-001", "ocr_path": "ocr/images/train/syn-001.jpg", "ocr_sha256": "abc"}],
        output,
    )

    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert "ocr_path" in MANIFEST_FIELDS
    assert row["ocr_path"] == "ocr/images/train/syn-001.jpg"
    assert row["ocr_sha256"] == "abc"
