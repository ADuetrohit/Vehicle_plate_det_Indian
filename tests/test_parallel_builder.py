from __future__ import annotations

import csv
from dataclasses import replace
import os
from pathlib import Path

from PIL import Image
import pytest

import plate_dataset.builder as builder
from plate_dataset.builder import BuildManifest, build_dataset
from plate_dataset.config import BuildConfig
from plate_dataset.manifests import write_manifest
from plate_dataset.records import Box, ImageRecord


def _config(workspace: Path) -> BuildConfig:
    return BuildConfig(
        workspace=workspace,
        seed=20260814,
        target_images=10,
        max_images=10,
        mh_share=(0.60, 0.70),
        split=(0.80, 0.10, 0.10),
        negative_share=(0.10, 0.10),
        training_imgsz=512,
        min_box_at_training_size=(8, 4),
        synthetic_only=True,
        max_scene_edge=160,
        ocr_canvas=(256, 128),
        min_free_gb=0,
    )


def _write_records(root: Path, count: int = 8) -> list[ImageRecord]:
    records = []
    for index in range(count):
        image_path = root / "source" / f"scene-{index}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (320, 180), color=(40 + index * 20, 70, 100)).save(image_path)
        records.append(
            ImageRecord(
                record_id=f"record-{index:03d}",
                image_path=image_path,
                width=320,
                height=180,
                boxes=(Box(0, 100, 120, 220, 155),),
                source_id="fixture",
                source_family=f"family-{index:03d}",
                is_real=True,
                plate_text=f"MH12AB{index + 1:04d}",
                tags={"vehicle_type": "car", "viewpoint": "rear", "state": "MH"},
            )
        )
    return records


def _manifest_projection(manifest: BuildManifest) -> list[dict[str, str]]:
    with manifest.manifest_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_parallel_and_serial_builds_have_identical_ids_and_labels(tmp_path: Path) -> None:
    """Catches worker scheduling changing output metadata or invoking progress off-coordinator."""
    records = _write_records(tmp_path)
    config = _config(tmp_path / "unused")
    coordinator_pid = os.getpid()
    serial_progress: list[tuple[int, int, int]] = []
    parallel_progress: list[tuple[int, int, int]] = []

    serial = build_dataset(
        replace(config, workers=1),
        records,
        tmp_path / "serial",
        progress=lambda completed, total: serial_progress.append(
            (completed, total, os.getpid())
        ),
    )
    parallel = build_dataset(
        replace(config, workers=2),
        records,
        tmp_path / "parallel",
        progress=lambda completed, total: parallel_progress.append(
            (completed, total, os.getpid())
        ),
    )

    assert _manifest_projection(serial) == _manifest_projection(parallel)
    expected_progress = [
        (completed, config.target_images, coordinator_pid)
        for completed in range(1, config.target_images + 1)
    ]
    assert serial_progress == expected_progress
    assert parallel_progress == expected_progress


class InjectedRenderFailure(RuntimeError):
    pass


def test_failed_build_checkpoints_five_rows_and_resumes_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches completed outputs being lost when a renderer fails between checkpoints."""
    records = _write_records(tmp_path)
    output = tmp_path / "dataset"
    config = replace(_config(output), workers=1)
    original_renderer = builder._render_synthetic_spec
    completed = 0
    failed_output_id = ""

    def fail_after_seven_results(*args: object, **kwargs: object):
        nonlocal completed, failed_output_id
        spec = args[1]
        if completed == 7:
            failed_output_id = spec.output_id
            raise InjectedRenderFailure("injected failure after seven results")
        result = original_renderer(*args, **kwargs)
        completed += 1
        return result

    monkeypatch.setattr(builder, "_CHECKPOINT_INTERVAL", 5, raising=False)
    monkeypatch.setattr(builder, "_render_synthetic_spec", fail_after_seven_results)

    with pytest.raises(RuntimeError) as caught:
        build_dataset(config, records, output)

    checkpoint_path = output / "metadata" / "generation_manifest.csv"
    assert checkpoint_path.is_file()
    with checkpoint_path.open(newline="", encoding="utf-8") as handle:
        checkpoint_rows = list(csv.DictReader(handle))
    assert len(checkpoint_rows) == 5
    assert {row["output_id"] for row in checkpoint_rows} == {
        spec.output_id
        for spec in builder._synthetic_specs(config, records, frozenset())[:5]
    }
    assert failed_output_id in str(caught.value)
    assert "InjectedRenderFailure" in str(caught.value)

    monkeypatch.setattr(builder, "_render_synthetic_spec", original_renderer)
    resumed = build_dataset(config, records, output)
    final_rows = _manifest_projection(resumed)

    assert resumed.generated_count == config.target_images
    assert resumed.reused_count == 5
    assert {row["output_id"] for row in checkpoint_rows} <= {
        row["output_id"] for row in final_rows
    }


def test_repeated_resume_preserves_non_prefix_durable_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a later partial checkpoint dropping valid rows from an earlier checkpoint."""
    records = _write_records(tmp_path)
    output = tmp_path / "dataset"
    config = replace(_config(output), workers=1)
    initial = build_dataset(config, records, output)
    initial_by_id = {
        row["output_id"]: row for row in _manifest_projection(initial)
    }
    specs = builder._synthetic_specs(config, records, frozenset())
    prior_ids = {spec.output_id for spec in specs[5:]}
    write_manifest(
        [initial_by_id[output_id] for output_id in prior_ids],
        initial.manifest_path,
    )

    original_renderer = builder._render_synthetic_spec
    completed = 0

    def fail_after_seven_results(*args: object, **kwargs: object):
        nonlocal completed
        if completed == 7:
            raise InjectedRenderFailure("second interruption")
        result = original_renderer(*args, **kwargs)
        completed += 1
        return result

    monkeypatch.setattr(builder, "_CHECKPOINT_INTERVAL", 5)
    monkeypatch.setattr(builder, "_render_synthetic_spec", fail_after_seven_results)

    with pytest.raises(RuntimeError, match="InjectedRenderFailure"):
        build_dataset(config, records, output)

    checkpoint_rows = _manifest_projection(initial)
    expected_ids = prior_ids | {spec.output_id for spec in specs[:5]}
    assert {row["output_id"] for row in checkpoint_rows} == expected_ids

    monkeypatch.setattr(builder, "_render_synthetic_spec", original_renderer)
    final = build_dataset(config, records, output)

    assert final.generated_count == config.target_images
    assert final.reused_count == config.target_images
