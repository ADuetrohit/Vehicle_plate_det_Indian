from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import replace
from pathlib import Path

from PIL import Image
import pytest

from plate_dataset.builder import InsufficientSourceData, build_dataset
from plate_dataset.config import BuildConfig
from plate_dataset.records import Box, ImageRecord
from plate_dataset.split import assign_splits


def _config(workspace: Path, target: int = 100) -> BuildConfig:
    return BuildConfig(
        workspace=workspace,
        seed=20260814,
        target_images=target,
        max_images=target,
        mh_share=(0.60, 0.70),
        split=(0.80, 0.10, 0.10),
        negative_share=(0.10, 0.10) if target <= 10 else (0.05, 0.10),
        training_imgsz=512,
        min_box_at_training_size=(8, 4),
    )


def _record(index: int, *, is_real: bool = True, family: str | None = None) -> ImageRecord:
    return ImageRecord(
        record_id=f"record-{index:03d}",
        image_path=Path(f"unused-{index:03d}.jpg"),
        width=320,
        height=180,
        boxes=(Box(0, 100, 120, 220, 155),),
        source_id="fixture",
        source_family=family or f"family-{index:03d}",
        is_real=is_real,
        plate_text=None,
        tags={"vehicle_type": "car", "viewpoint": "rear"},
    )


def test_split_matches_ratios_and_real_holdout_minimum(tmp_path: Path) -> None:
    """Catches skewed splits or synthetic-heavy validation/test sets."""
    records = [_record(i, is_real=i < 90) for i in range(100)]

    assignments = assign_splits(records, _config(tmp_path))

    counts = Counter(item.split for item in assignments.values())
    assert counts == {"train": 80, "val": 10, "test": 10}
    for split in ("val", "test"):
        chosen = [r for r in records if assignments[r.record_id].split == split]
        assert sum(record.is_real for record in chosen) / len(chosen) >= 0.80


def test_split_keeps_source_family_together(tmp_path: Path) -> None:
    """Catches variants of one scene leaking into separate splits."""
    records = [_record(i) for i in range(30)]
    records[1] = replace(records[1], source_family=records[0].source_family)

    assignments = assign_splits(records, _config(tmp_path, target=30))
    by_family: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_family[record.source_family].add(assignments[record.record_id].split)

    assert all(len(splits) == 1 for splits in by_family.values())


def test_split_fails_when_real_holdout_requirement_is_impossible(tmp_path: Path) -> None:
    """Catches silently weakening the real validation/test requirement."""
    records = [_record(0, is_real=True)] + [
        _record(i, is_real=False) for i in range(1, 10)
    ]

    with pytest.raises(InsufficientSourceData, match="required_real=2 available_real=1"):
        assign_splits(records, _config(tmp_path, target=10))


def _write_real_records(root: Path, count: int) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for index in range(count):
        path = root / "source" / f"scene-{index}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (320, 180), color=(40 + index * 20, 70, 100)).save(path)
        records.append(
            replace(
                _record(index),
                image_path=path,
                plate_text=f"MH12AB{index + 1:04d}",
                tags={"vehicle_type": "car", "viewpoint": "rear", "state": "MH"},
            )
        )
    return records


def test_builder_fills_synthetic_deficit_and_writes_valid_pairs(tmp_path: Path) -> None:
    """Catches a build stopping at the real-source count or emitting orphan labels."""
    records = _write_real_records(tmp_path, 4)
    output = tmp_path / "dataset"

    manifest = build_dataset(_config(output, target=10), records, output)

    assert manifest.generated_count == 10
    assert manifest.reused_count == 0
    images = list((output / "detection" / "images").glob("*/*.jpg"))
    labels = list((output / "detection" / "labels").glob("*/*.txt"))
    assert len(images) == len(labels) == 10
    split_counts = Counter(path.parent.name for path in images)
    assert split_counts == {"train": 8, "val": 1, "test": 1}
    assert sum(path.read_text(encoding="utf-8") == "" for path in labels) == 1

    with manifest.manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert Counter(row["origin"] for row in rows) == {"real": 4, "synthetic": 6}
    synthetic_positive = [
        row for row in rows if row["origin"] == "synthetic" and row["negative"] == "false"
    ]
    mh_share = sum(row["state"] == "MH" for row in synthetic_positive) / len(synthetic_positive)
    assert 0.60 <= mh_share <= 0.70


def test_builder_rerun_reuses_checksum_valid_outputs(tmp_path: Path) -> None:
    """Catches a resumed build rewriting deterministic outputs."""
    records = _write_real_records(tmp_path, 4)
    output = tmp_path / "dataset"
    config = _config(output, target=10)

    first = build_dataset(config, records, output)
    second = build_dataset(config, records, output)

    assert first.generated_count == second.generated_count == 10
    assert second.reused_count == 10
    assert len(list((output / "detection" / "images").glob("*/*.jpg"))) == 10
