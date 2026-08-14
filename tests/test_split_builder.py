from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import replace
from pathlib import Path

from PIL import Image
import pytest

from plate_dataset.builder import InsufficientSourceData, _synthetic_specs, build_dataset
from plate_dataset.config import BuildConfig, load_config
from plate_dataset.manifests import sha256_file, write_manifest
from plate_dataset.quotas import generation_quotas
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

def test_synthetic_only_assigns_all_source_families_without_real_holdout(tmp_path: Path) -> None:
    config = replace(_config(tmp_path, target=50), synthetic_only=True)
    records = [_record(i, is_real=True) for i in range(20)]
    assignments = assign_splits(records, config)
    assert set(assignments) == {record.record_id for record in records}
    assert {item.split for item in assignments.values()} == {"train", "val", "test"}
    assert Counter(item.split for item in assignments.values()) == {"train": 16, "val": 2, "test": 2}
    by_family = defaultdict(set)
    for record in records:
        by_family[record.source_family].add(assignments[record.record_id].split)
    assert all(len(values) == 1 for values in by_family.values())

def test_synthetic_only_rejects_too_few_source_families(tmp_path: Path) -> None:
    config = replace(_config(tmp_path, target=50), synthetic_only=True)
    records = [_record(0, family="shared"), _record(1, family="shared")]
    with pytest.raises(InsufficientSourceData, match="at least three source families"):
        assign_splits(records, config)


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
def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _synthetic_only_config(workspace: Path) -> BuildConfig:
    return replace(
        _config(workspace, target=20),
        synthetic_only=True,
        negative_share=(0.10, 0.10),
        max_scene_edge=160,
        ocr_canvas=(256, 128),
        min_free_gb=0,
    )


def test_synthetic_only_builder_writes_only_synthetic_outputs_and_crops(tmp_path: Path) -> None:
    """Catches synthetic-only builds copying source scenes or omitting positive OCR crops."""
    config = _synthetic_only_config(tmp_path / "dataset")

    result = build_dataset(config, _write_real_records(tmp_path, 8), config.workspace)

    rows = _manifest_rows(result.manifest_path)
    assert len(rows) == 20
    assert {row["origin"] for row in rows} == {"synthetic"}
    assert sum(row["negative"] == "true" for row in rows) == 2
    assert sum(bool(row["ocr_path"]) for row in rows) == 18
    assert all(max(Image.open(config.workspace / row["image_path"]).size) <= 160 for row in rows)


def test_synthetic_only_resume_checks_image_label_and_crop_checksums(tmp_path: Path) -> None:
    """Catches reuse accepting a stale OCR crop alongside an otherwise valid pair."""
    config = _synthetic_only_config(tmp_path / "dataset")
    records = _write_real_records(tmp_path, 8)

    first = build_dataset(config, records, config.workspace)
    second = build_dataset(config, records, config.workspace)
    assert second.reused_count == 20

    positive = next(row for row in _manifest_rows(first.manifest_path) if row["ocr_path"])
    crop_path = config.workspace / positive["ocr_path"]
    crop_path.write_bytes(b"stale crop")

    resumed = build_dataset(config, records, config.workspace)
    refreshed = next(row for row in _manifest_rows(resumed.manifest_path) if row["output_id"] == positive["output_id"])
    assert resumed.reused_count == 19
    assert sha256_file(crop_path) == refreshed["ocr_sha256"]


def test_rejected_output_is_replaced_without_changing_total(tmp_path: Path) -> None:
    """Catches rejected synthetic IDs remaining in a resumed manifest or shrinking its quota."""
    config = _synthetic_only_config(tmp_path / "dataset")
    records = _write_real_records(tmp_path, 8)

    first = build_dataset(config, records, config.workspace)
    rejected = _manifest_rows(first.manifest_path)[0]["output_id"]
    second = build_dataset(config, records, config.workspace, rejected_ids={rejected})
    second_ids = {row["output_id"] for row in _manifest_rows(second.manifest_path)}

    assert rejected not in second_ids
    assert len(second_ids) == 20

def test_synthetic_specs_match_every_default_split_quota_without_rendering() -> None:
    """Catches global-first allocation that violates per-split planner quotas."""
    config = load_config(Path("config/default.yaml"))
    records = [_record(index, family=f"default-family-{index}") for index in range(3)]

    specs = _synthetic_specs(config, records, frozenset())
    quotas = generation_quotas(config)
    for split, quota in quotas.items():
        chosen = [spec for spec in specs if spec.split == split]
        positives = [spec for spec in chosen if not spec.negative]
        assert len(chosen) == quota.total
        assert sum(spec.negative for spec in chosen) == quota.negatives
        assert sum(spec.force_mh for spec in positives) == quota.mh_positives
        assert sum(spec.layout == "double" for spec in positives) == quota.double_row_positives
        assert sum(spec.condition == "night" for spec in chosen) == quota.low_light
        assert sum(spec.condition == "rain" for spec in chosen) == quota.adverse


def test_synthetic_only_manifest_matches_small_split_quotas(tmp_path: Path) -> None:
    """Catches rendered rows drifting from their deterministic split specifications."""
    config = _synthetic_only_config(tmp_path / "dataset")
    records = _write_real_records(tmp_path, 8)

    result = build_dataset(config, records, config.workspace)
    rows = _manifest_rows(result.manifest_path)
    quotas = generation_quotas(config)
    for split, quota in quotas.items():
        chosen = [row for row in rows if row["split"] == split]
        positives = [row for row in chosen if row["negative"] == "false"]
        assert len(chosen) == quota.total
        assert sum(row["negative"] == "true" for row in chosen) == quota.negatives
        assert sum(row["state"] == "MH" for row in positives) == quota.mh_positives
        assert sum(row["plate_layout"] == "double" for row in positives) == quota.double_row_positives
        assert sum(row["effect"] == "night" for row in chosen) == quota.low_light
        assert sum(row["effect"] == "rain" for row in chosen) == quota.adverse


def test_synthetic_only_resume_requires_crop_linkage_for_positive(tmp_path: Path) -> None:
    """Catches a positive manifest row being reused after its required crop linkage is removed."""
    config = _synthetic_only_config(tmp_path / "dataset")
    records = _write_real_records(tmp_path, 8)
    first = build_dataset(config, records, config.workspace)
    rows = _manifest_rows(first.manifest_path)
    positive = next(row for row in rows if row["ocr_path"])
    positive["ocr_path"] = ""
    positive["ocr_sha256"] = ""
    write_manifest(rows, first.manifest_path)

    resumed = build_dataset(config, records, config.workspace)
    restored = next(row for row in _manifest_rows(resumed.manifest_path) if row["output_id"] == positive["output_id"])

    assert resumed.reused_count == 19
    assert restored["ocr_path"]
    assert restored["ocr_sha256"]


def test_synthetic_negative_records_apply_and_report_assigned_conditions(tmp_path: Path) -> None:
    """Catches inpainted negatives bypassing their assigned camera-condition profile."""
    config = _synthetic_only_config(tmp_path / "dataset")
    records = _write_real_records(tmp_path, 8)
    expected_conditions = {
        spec.output_id: spec.condition
        for spec in _synthetic_specs(config, records, frozenset())
        if spec.negative
    }

    result = build_dataset(config, records, config.workspace)
    negatives = [row for row in _manifest_rows(result.manifest_path) if row["negative"] == "true"]

    assert {row["output_id"] for row in negatives} == set(expected_conditions)
    assert all(row["effect"] == expected_conditions[row["output_id"]] for row in negatives)
    assert all(not row["ocr_path"] and not row["ocr_sha256"] for row in negatives)
    assert all((config.workspace / row["label_path"]).read_text(encoding="utf-8") == "" for row in negatives)

def test_synthetic_only_resume_rebuilds_checksum_valid_stale_spec_metadata(tmp_path: Path) -> None:
    """Catches resume preserving prior quota metadata after the current spec changes."""
    config = _synthetic_only_config(tmp_path / "dataset")
    records = _write_real_records(tmp_path, 8)
    specs = {spec.output_id: spec for spec in _synthetic_specs(config, records, frozenset())}
    first = build_dataset(config, records, config.workspace)
    rows = _manifest_rows(first.manifest_path)
    stale_negative = next(row for row in rows if row["negative"] == "true")
    stale_positive = next(
        row for row in rows if row["negative"] == "false" and specs[row["output_id"]].force_mh
    )
    stale_negative["negative"] = "false"
    stale_negative["effect"] = "plate_removed"
    stale_positive["plate_style"] = "commercial"
    stale_positive["plate_layout"] = "single"
    stale_positive["state"] = "DL"
    stale_positive["effect"] = "day"
    write_manifest(rows, first.manifest_path)

    resumed = build_dataset(config, records, config.workspace)
    final_rows = {row["output_id"]: row for row in _manifest_rows(resumed.manifest_path)}
    quotas = generation_quotas(config)

    assert resumed.reused_count == 18
    assert final_rows[stale_negative["output_id"]]["negative"] == "true"
    assert final_rows[stale_negative["output_id"]]["effect"] == specs[stale_negative["output_id"]].condition
    assert final_rows[stale_positive["output_id"]]["plate_style"] == specs[stale_positive["output_id"]].category
    assert final_rows[stale_positive["output_id"]]["plate_layout"] == specs[stale_positive["output_id"]].layout
    assert final_rows[stale_positive["output_id"]]["state"] == "MH"
    for split, quota in quotas.items():
        assert sum(
            row["negative"] == "true" for row in final_rows.values() if row["split"] == split
        ) == quota.negatives
