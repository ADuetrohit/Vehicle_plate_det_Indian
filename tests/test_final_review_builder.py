from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytest

import plate_dataset.builder as builder
from plate_dataset.augment import EffectProfile
from plate_dataset.builder import build_dataset
from plate_dataset.config import BuildConfig
from plate_dataset.manifests import sha256_file
from plate_dataset.records import Box, ImageRecord


def _config(workspace: Path, *, target: int = 10) -> BuildConfig:
    return BuildConfig(
        workspace=workspace,
        seed=20260814,
        target_images=target,
        max_images=target,
        mh_share=(0.60, 0.70),
        split=(0.80, 0.10, 0.10),
        negative_share=(0.10, 0.10),
        training_imgsz=512,
        min_box_at_training_size=(8, 4),
        synthetic_only=True,
        max_scene_edge=320,
        jpeg_quality=88,
        ocr_canvas=(256, 128),
        min_free_gb=0,
        workers=1,
    )


def _write_records(root: Path, count: int = 3) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for index in range(count):
        image_path = root / "source" / f"scene-{index}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (320, 180), (25 + index * 5, 65, 105))
        draw = ImageDraw.Draw(image)
        draw.rectangle((100, 100, 240, 150), fill=(250, 250, 250))
        draw.rectangle((20, 20, 80, 50), fill=(250, 250, 250))
        image.save(image_path, quality=95, subsampling=0)
        records.append(
            ImageRecord(
                record_id=f"record-{index}",
                image_path=image_path,
                width=320,
                height=180,
                boxes=(Box(0, 100, 100, 240, 150), Box(0, 20, 20, 80, 50)),
                source_id=f"fixture/source-{index}",
                source_family=f"family-{index}",
                is_real=True,
                plate_text=f"MH12AB{index + 1:04d}",
                tags={"vehicle_type": "car", "viewpoint": "rear"},
            )
        )
    return records


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _neutralize_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        builder,
        "named_effect_profile",
        lambda name, rng: EffectProfile(name=name, jpeg_quality=100),
    )


def _secondary_anchor_mean(workspace: Path, row: dict[str, str]) -> float:
    with Image.open(workspace / row["image_path"]) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return float(array[22:48, 22:78].mean())


def test_positive_output_erases_every_secondary_source_plate_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a positive output retaining a second unmodified real plate."""
    workspace = tmp_path / "dataset"
    _neutralize_effects(monkeypatch)

    result = build_dataset(_config(workspace), _write_records(tmp_path), workspace)
    positives = [row for row in _rows(result.manifest_path) if row["negative"] == "false"]

    assert positives
    assert all(_secondary_anchor_mean(workspace, row) < 180 for row in positives)
    assert all(
        len((workspace / row["label_path"]).read_text(encoding="utf-8").splitlines()) == 1
        for row in positives
    )


def test_negative_output_erases_every_source_plate_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a hard negative retaining an unmodified secondary real plate."""
    workspace = tmp_path / "dataset"
    _neutralize_effects(monkeypatch)

    result = build_dataset(_config(workspace), _write_records(tmp_path), workspace)
    negatives = [row for row in _rows(result.manifest_path) if row["negative"] == "true"]

    assert negatives
    assert all(_secondary_anchor_mean(workspace, row) < 180 for row in negatives)
    assert all(
        (workspace / row["label_path"]).read_text(encoding="utf-8") == ""
        for row in negatives
    )


@pytest.mark.parametrize(
    ("field", "changed_value"),
    (
        ("max_scene_edge", 280),
        ("jpeg_quality", 73),
        ("ocr_canvas", (192, 96)),
        ("mh_share", (0.50, 0.50)),
    ),
)
def test_resume_rebuilds_all_rows_when_output_profile_changes(
    tmp_path: Path, field: str, changed_value: object
) -> None:
    """Catches checksum-valid outputs being reused under an incompatible profile."""
    workspace = tmp_path / "dataset"
    config = _config(workspace)
    records = _write_records(tmp_path)
    first = build_dataset(config, records, workspace)
    first_rows = _rows(first.manifest_path)

    resumed = build_dataset(
        replace(config, **{field: changed_value}), records, workspace
    )
    resumed_rows = _rows(resumed.manifest_path)

    assert resumed.reused_count == 0
    assert all(row["generation_profile_sha256"] for row in resumed_rows)
    assert {
        row["generation_profile_sha256"] for row in resumed_rows
    }.isdisjoint({row["generation_profile_sha256"] for row in first_rows})


def test_resume_rebuilds_only_rows_whose_source_image_contents_changed(
    tmp_path: Path,
) -> None:
    """Catches a source image edit being hidden by valid output checksums."""
    workspace = tmp_path / "dataset"
    config = _config(workspace)
    records = _write_records(tmp_path)
    first = build_dataset(config, records, workspace)
    first_rows = _rows(first.manifest_path)
    changed_source = records[0]
    with Image.open(changed_source.image_path) as original:
        changed = Image.new("RGB", original.size, (150, 30, 40))
    draw = ImageDraw.Draw(changed)
    draw.rectangle((100, 100, 240, 150), fill=(250, 250, 250))
    draw.rectangle((20, 20, 80, 50), fill=(250, 250, 250))
    changed.save(changed_source.image_path, quality=95, subsampling=0)

    resumed = build_dataset(config, records, workspace)
    resumed_rows = _rows(resumed.manifest_path)
    affected = [
        row for row in first_rows if row["source_id"] == changed_source.source_id
    ]

    assert affected
    assert resumed.reused_count == config.target_images - len(affected)
    assert all(
        row["source_image_sha256"] == sha256_file(changed_source.image_path)
        for row in resumed_rows
        if row["source_id"] == changed_source.source_id
    )


def test_resume_rebuilds_all_rows_when_source_membership_changes(
    tmp_path: Path,
) -> None:
    """Catches a changed source pool reusing rows selected under the old membership."""
    workspace = tmp_path / "dataset"
    config = _config(workspace)
    records = _write_records(tmp_path)
    first = build_dataset(config, records, workspace)
    first_profile = {
        row.get("generation_profile_sha256", "") for row in _rows(first.manifest_path)
    }
    extra = replace(
        _write_records(tmp_path / "extra", 1)[0],
        record_id="record-extra",
        source_id="fixture/source-extra",
        source_family=records[0].source_family,
    )

    resumed = build_dataset(config, [*records, extra], workspace)
    resumed_profile = {
        row.get("generation_profile_sha256", "") for row in _rows(resumed.manifest_path)
    }

    assert resumed.reused_count == 0
    assert resumed_profile.isdisjoint(first_profile)


def test_full_reuse_run_does_not_generate_any_new_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches identity preparation doing quadratic work before the reuse check."""
    workspace = tmp_path / "dataset"
    config = _config(workspace)
    records = _write_records(tmp_path)
    build_dataset(config, records, workspace)

    def fail_identity_generation(*args: object, **kwargs: object):
        raise AssertionError("a checksum-compatible reuse must not generate identities")

    monkeypatch.setattr(builder, "generate_identity", fail_identity_generation)

    resumed = build_dataset(config, records, workspace)

    assert resumed.reused_count == config.target_images


def test_existing_ocr_registration_is_never_emitted(
    tmp_path: Path,
) -> None:
    """Catches imported OCR text being omitted from the identity forbidden set."""
    workspace = tmp_path / "dataset"
    config = _config(workspace)
    records = _write_records(tmp_path)
    first_positive = next(
        spec
        for spec in builder._synthetic_specs(config, records, frozenset())
        if not spec.negative
    )
    generator = builder._rng(
        config.seed,
        first_positive.split,
        first_positive.source.source_family,
        first_positive.variant_index,
    )
    forbidden_identity = builder.generate_identity(
        generator,
        mh_probability=1.0 if first_positive.force_mh else 0.0,
        forbidden=set(),
        forbidden_is_normalized=True,
    ).compact_text

    result = build_dataset(
        config,
        records,
        workspace,
        forbidden_plate_texts=frozenset({forbidden_identity}),
    )
    emitted = {
        row["plate_text"]
        for row in _rows(result.manifest_path)
        if row["negative"] == "false"
    }

    assert forbidden_identity not in emitted


def test_rejecting_one_candidate_changes_only_its_logical_slot(
    tmp_path: Path,
) -> None:
    """Catches one rejection shifting variants and metadata for later family slots."""
    workspace = tmp_path / "dataset"
    config = _config(workspace)
    records = _write_records(tmp_path)
    first = build_dataset(config, records, workspace)
    before = {row["output_id"]: row for row in _rows(first.manifest_path)}
    rejected = next(
        row["output_id"]
        for row in before.values()
        if row["split"] == "train" and row["negative"] == "false"
    )

    resumed = build_dataset(
        config, records, workspace, rejected_ids=frozenset({rejected})
    )
    after = {row["output_id"]: row for row in _rows(resumed.manifest_path)}
    shared = before.keys() & after.keys()

    assert len(shared) == config.target_images - 1
    assert all(after[output_id] == before[output_id] for output_id in shared)
    assert len(after.keys() - before.keys()) == 1
    tombstone_path = workspace / "metadata" / "rejected_generation_slots.json"
    tombstones = json.loads(tombstone_path.read_text(encoding="utf-8"))
    matching = [
        (logical_slot, entry)
        for logical_slot, entries in tombstones["slots"].items()
        for entry in entries
        if entry["candidate_id"] == rejected
    ]
    assert len(matching) == 1
    assert matching[0][0].startswith("train:")
    assert matching[0][1]["attempt"] == 0


def test_reject_tombstone_is_durable_before_replacement_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an interrupted reject pass forgetting which logical slot advanced."""
    workspace = tmp_path / "dataset"
    config = _config(workspace)
    records = _write_records(tmp_path)
    first = build_dataset(config, records, workspace)
    original_ids = {row["output_id"] for row in _rows(first.manifest_path)}
    rejected = next(iter(original_ids))
    original_renderer = builder._render_synthetic_spec

    def fail_replacement(*args: object, **kwargs: object):
        spec = args[1]
        if spec.output_id not in original_ids:
            raise RuntimeError("injected replacement failure")
        return original_renderer(*args, **kwargs)

    monkeypatch.setattr(builder, "_render_synthetic_spec", fail_replacement)

    with pytest.raises(RuntimeError, match="failed with RuntimeError"):
        build_dataset(config, records, workspace, rejected_ids=frozenset({rejected}))

    tombstone_path = workspace / "metadata" / "rejected_generation_slots.json"
    assert tombstone_path.is_file()
    assert not tombstone_path.with_name(f".{tombstone_path.name}.tmp").exists()

    monkeypatch.setattr(builder, "_render_synthetic_spec", original_renderer)
    resumed = build_dataset(
        config, records, workspace, rejected_ids=frozenset({rejected})
    )
    final_ids = {row["output_id"] for row in _rows(resumed.manifest_path)}

    assert rejected not in final_ids
    assert len(final_ids) == config.target_images


def test_manifest_detector_eligibility_matches_emitted_labels(
    tmp_path: Path,
) -> None:
    """Catches negatives or training-undersized positives being marked detector eligible."""
    workspace = tmp_path / "dataset"
    config = _config(workspace)

    result = build_dataset(config, _write_records(tmp_path), workspace)
    rows = _rows(result.manifest_path)

    for row in rows:
        label_text = (workspace / row["label_path"]).read_text(encoding="utf-8")
        if row["negative"] == "true":
            assert row["detector_eligible"] == "false"
            assert label_text == ""
        else:
            assert row["detector_eligible"] == "true"
            assert len(label_text.splitlines()) == 1
