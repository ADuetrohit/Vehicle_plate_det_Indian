from __future__ import annotations

import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

from PIL import Image
import pytest
import yaml

from plate_dataset.builder import BuildManifest, build_dataset
from plate_dataset.config import load_config
from plate_dataset.manifests import write_manifest
from plate_dataset.storage import StorageEstimate
from scripts import generate_synthetic
from scripts import validate_dataset as validation_cli


def _write_config(
    path: Path,
    workspace: Path,
    *,
    target_images: int = 50_000,
    min_free_gb: float = 5.0,
) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "workspace": workspace.as_posix(),
                "seed": 20260814,
                "target_images": target_images,
                "max_images": target_images,
                "mh_share": [0.60, 0.70],
                "split": [0.80, 0.10, 0.10],
                "negative_share": [0.075, 0.075],
                "training_imgsz": 512,
                "min_box_at_training_size": [8, 4],
                "synthetic_only": True,
                "max_scene_edge": 960,
                "jpeg_quality": 88,
                "ocr_canvas": [256, 128],
                "min_free_gb": min_free_gb,
                "workers": 0,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _write_normalized_record(workspace: Path, *, count: int = 1) -> Path:
    manifest = workspace / "metadata" / "normalized_records.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(count):
        image_path = workspace / "raw" / f"source-{index:03d}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(
            "RGB", (320, 180), color=(40 + index * 10, 80, 120)
        ).save(image_path, quality=88)
        rows.append(
            {
                "record_id": f"source-{index:03d}",
                "image_path": image_path.relative_to(workspace).as_posix(),
                "width": 320,
                "height": 180,
                "boxes": [[0, 100, 120, 220, 155]],
                "source_id": "fixture/source",
                "source_family": f"fixture-family-{index:03d}",
                "is_real": True,
                "plate_text": None,
                "tags": {"viewpoint": "rear"},
            }
        )
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return workspace / "raw" / "source-000.jpg"


def _jpeg_payload(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (200, 100), color=color).save(output, format="JPEG")
    return output.getvalue()


def _write_yolo_archive(path: Path, image_name: str, *, valid: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label = b"0 0.5 0.5 0.4 0.2\n" if valid else b"1 0.5 0.5 0.4 0.2\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"dataset/images/{image_name}.jpg", _jpeg_payload((50, 80, 110)))
        archive.writestr(f"dataset/labels/{image_name}.txt", label)


@pytest.mark.parametrize(
    "script",
    [
        "download_sources.py",
        "convert_annotations.py",
        "generate_synthetic.py",
        "validate_dataset.py",
    ],
)
def test_cli_help_exits_zero(script: str) -> None:
    result = subprocess.run(
        [sys.executable, f"scripts/{script}", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_data_yaml_is_portable() -> None:
    data = yaml.safe_load(Path("detection/data.yaml").read_text(encoding="utf-8"))
    assert data == {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "number_plate"},
    }


def test_download_dry_run_does_not_create_workspace(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    workspace = tmp_path / "never-created"
    config.write_text(
        "\n".join(
            (
                f"workspace: '{workspace.as_posix()}'",
                "seed: 7",
                "target_images: 10",
                "max_images: 10",
                "mh_share: [0.60, 0.70]",
                "split: [0.80, 0.10, 0.10]",
                "negative_share: [0.05, 0.10]",
                "training_imgsz: 512",
                "min_box_at_training_size: [8, 4]",
            )
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/download_sources.py", "--config", str(config), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not workspace.exists()
    assert "credential_values_never_printed" in result.stdout


def test_generation_help_exposes_parallel_reject_and_resume_controls() -> None:
    """Catches required generation controls disappearing from the public CLI."""
    result = subprocess.run(
        [sys.executable, "scripts/generate_synthetic.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for flag in ("--workers", "--reject-file", "--resume", "--no-resume"):
        assert flag in result.stdout


def test_generation_dry_run_reports_exact_50000_plan_without_detector_files(
    tmp_path: Path,
) -> None:
    """Catches a dry-run mutating detector output or hiding quota/storage projections."""
    workspace = tmp_path / "workspace"
    config = _write_config(tmp_path / "config.yaml", workspace)
    _write_normalized_record(workspace)

    result = subprocess.run(
        [sys.executable, "scripts/generate_synthetic.py", "--config", str(config), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "target_images=50000" in result.stdout
    assert "split_train=40000" in result.stdout
    assert "split_val=5000" in result.stdout
    assert "split_test=5000" in result.stdout
    assert "projected_storage_bytes=" in result.stdout
    assert "storage_reserve_bytes=5368709120" in result.stdout
    assert not (workspace / "detection").exists()


def test_conversion_excludes_unapproved_source_archives(tmp_path: Path) -> None:
    """Catches a registered but unapproved source leaking into normalized records."""
    workspace = tmp_path / "workspace"
    config = _write_config(tmp_path / "config.yaml", workspace, target_images=10)
    allowed = workspace / "raw" / "kedarsai__indian-license-plates-with-labels" / "a.zip"
    unapproved = workspace / "raw" / "saisirishan__indian-vehicle-dataset" / "b.zip"
    _write_yolo_archive(allowed, "allowed")
    _write_yolo_archive(unapproved, "unapproved")

    result = subprocess.run(
        [sys.executable, "scripts/convert_annotations.py", "--config", str(config)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = workspace / "metadata" / "normalized_records.jsonl"
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {row["source_id"] for row in rows} == {
        "kedarsai/indian-license-plates-with-labels"
    }
    assert not output.with_name(f".{output.name}.tmp").exists()


def test_failed_conversion_preserves_previous_normalized_manifest(tmp_path: Path) -> None:
    """Catches partial conversion replacing the last complete normalized manifest."""
    workspace = tmp_path / "workspace"
    config = _write_config(tmp_path / "config.yaml", workspace, target_images=10)
    first = workspace / "raw" / "kedarsai__indian-license-plates-with-labels" / "a.zip"
    second = (
        workspace
        / "raw"
        / "deepakat002__indian-vehicle-number-plate-yolo-annotation"
        / "b.zip"
    )
    _write_yolo_archive(first, "valid")
    _write_yolo_archive(second, "invalid", valid=False)
    output = workspace / "metadata" / "normalized_records.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("previous-complete-manifest\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/convert_annotations.py", "--config", str(config)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert output.read_text(encoding="utf-8") == "previous-complete-manifest\n"
    assert not output.with_name(f".{output.name}.tmp").exists()


def test_generation_preflight_refuses_before_output_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches output directories being created before the 5 GB reserve is enforced."""
    workspace = tmp_path / "workspace"
    config = _write_config(tmp_path / "config.yaml", workspace)
    _write_normalized_record(workspace)

    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=0))

    def forbidden_build(*args: object, **kwargs: object) -> BuildManifest:
        raise AssertionError("builder must not run after failed storage preflight")

    monkeypatch.setattr(generate_synthetic, "build_dataset", forbidden_build)

    result = generate_synthetic.main(["--config", str(config)])

    assert result == 2
    assert not (workspace / "detection").exists()
    assert not (workspace / "ocr").exists()


def test_generation_wires_workers_rejects_progress_and_merged_ocr_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches CLI options or generated OCR labels being dropped at the builder boundary."""
    workspace = tmp_path / "workspace"
    config = _write_config(
        tmp_path / "config.yaml", workspace, target_images=1_000, min_free_gb=0
    )
    _write_normalized_record(workspace, count=8)
    rejects = tmp_path / "rejects.txt"
    rejected_id = "syn-aaaaaaaaaaaaaaaaaaaa"
    rejects.write_text(rejected_id + "\n", encoding="utf-8")
    stale_image = workspace / "detection" / "images" / "train" / f"{rejected_id}.jpg"
    stale_label = workspace / "detection" / "labels" / "train" / f"{rejected_id}.txt"
    stale_crop = workspace / "ocr" / "images" / "train" / f"{rejected_id}.jpg"
    for path in (stale_image, stale_label, stale_crop):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale")
    write_manifest(
        [
            {
                "output_id": rejected_id,
                "split": "train",
                "image_path": f"detection/images/train/{rejected_id}.jpg",
                "label_path": f"detection/labels/train/{rejected_id}.txt",
                "ocr_path": f"ocr/images/train/{rejected_id}.jpg",
                "negative": "false",
            }
        ],
        workspace / "metadata" / "generation_manifest.csv",
    )
    existing = workspace / "ocr" / "labels.csv"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(
        "image_name,image_path,plate_text,split,source_id,synthetic,reconciliation\n"
        "existing.jpg,images/train/existing.jpg,MH12AB1234,train,existing_archive,false,preserved\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        generate_synthetic,
        "estimate_storage",
        lambda config, samples: StorageEstimate(projected_bytes=1, reserve_bytes=0),
        raising=False,
    )

    def fake_build(config, records, output, rejected_ids=frozenset(), progress=None):
        assert config.workers == 3
        assert rejected_ids == frozenset({rejected_id})
        assert progress is not None
        for completed in (499, 500, 999, 1000):
            progress(completed, 1000)
        manifest_path = output / "metadata" / "generation_manifest.csv"
        write_manifest(
            [
                {
                    "output_id": "syn-001",
                    "split": "train",
                    "origin": "synthetic",
                    "source_id": "fixture/source",
                    "source_family": "fixture-family",
                    "image_path": "detection/images/train/syn-001.jpg",
                    "label_path": "detection/labels/train/syn-001.txt",
                    "negative": "false",
                    "plate_text": "MH14CD5678",
                    "ocr_path": "ocr/images/train/syn-001.jpg",
                }
            ],
            manifest_path,
        )
        return BuildManifest(20260814, 1000, 0, 2, manifest_path)

    monkeypatch.setattr(generate_synthetic, "build_dataset", fake_build)

    result = generate_synthetic.main(
        ["--config", str(config), "--workers", "3", "--reject-file", str(rejects)]
    )

    assert result == 0
    stdout = capsys.readouterr().out
    assert "progress=500/1000" in stdout
    assert "progress=1000/1000" in stdout
    assert "progress=499/1000" not in stdout
    with (workspace / "ocr" / "labels.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["image_name"] for row in rows] == ["existing.jpg", "syn-001.jpg"]
    assert not stale_image.exists()
    assert not stale_label.exists()
    assert not stale_crop.exists()


def test_generation_reject_file_must_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an explicit but misspelled visual-QA reject file being ignored."""
    workspace = tmp_path / "workspace"
    config = _write_config(tmp_path / "config.yaml", workspace, target_images=10)
    _write_normalized_record(workspace)

    def forbidden_build(*args: object, **kwargs: object) -> BuildManifest:
        raise AssertionError("builder must not run for a missing reject file")

    monkeypatch.setattr(generate_synthetic, "build_dataset", forbidden_build)

    result = generate_synthetic.main(
        ["--config", str(config), "--reject-file", str(tmp_path / "missing.txt")]
    )

    assert result == 2


def test_generation_reject_ids_must_exist_in_previous_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches unknown visual-QA IDs reporting success without replacing an output."""
    workspace = tmp_path / "workspace"
    config = _write_config(tmp_path / "config.yaml", workspace, target_images=10)
    _write_normalized_record(workspace)
    rejects = tmp_path / "rejects.txt"
    rejects.write_text("unknown-001\n", encoding="utf-8")
    write_manifest(
        [{"output_id": "known-001"}],
        workspace / "metadata" / "generation_manifest.csv",
    )

    def forbidden_build(*args: object, **kwargs: object) -> BuildManifest:
        raise AssertionError("builder must not run for unknown reject IDs")

    monkeypatch.setattr(generate_synthetic, "build_dataset", forbidden_build)

    result = generate_synthetic.main(
        ["--config", str(config), "--reject-file", str(rejects)]
    )

    assert result == 2


def test_generation_refuses_rejected_artifact_path_outside_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a crafted prior manifest turning reject cleanup into path traversal."""
    workspace = tmp_path / "workspace"
    config = _write_config(tmp_path / "config.yaml", workspace, target_images=10)
    _write_normalized_record(workspace)
    output_id = "../../../../victim"
    rejects = tmp_path / "rejects.txt"
    rejects.write_text(output_id + "\n", encoding="utf-8")
    write_manifest(
        [
            {
                "output_id": output_id,
                "split": "train",
                "image_path": f"detection/images/train/{output_id}.jpg",
                "label_path": f"detection/labels/train/{output_id}.txt",
                "negative": "true",
            }
        ],
        workspace / "metadata" / "generation_manifest.csv",
    )

    def forbidden_build(*args: object, **kwargs: object) -> BuildManifest:
        raise AssertionError("builder must not run for an unsafe cleanup path")

    monkeypatch.setattr(generate_synthetic, "build_dataset", forbidden_build)

    result = generate_synthetic.main(
        ["--config", str(config), "--reject-file", str(rejects)]
    )

    assert result == 2


def test_generation_resume_preflight_projects_only_missing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a completed resume requiring free space for a second full dataset."""
    workspace = tmp_path / "workspace"
    config = _write_config(
        tmp_path / "config.yaml", workspace, target_images=10, min_free_gb=0
    )
    _write_normalized_record(workspace, count=8)
    loaded = load_config(config)
    records = generate_synthetic._load_records(
        workspace / "metadata" / "normalized_records.jsonl", workspace
    )
    built = build_dataset(replace(loaded, workers=1), records, workspace)
    monkeypatch.setattr(
        generate_synthetic,
        "estimate_storage",
        lambda config, samples: StorageEstimate(
            projected_bytes=10_000, reserve_bytes=5_000
        ),
    )
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=6_000))

    result = generate_synthetic.main(["--config", str(config), "--dry-run"])

    assert result == 0
    assert "projected_storage_bytes=0" in capsys.readouterr().out

    with built.manifest_path.open(newline="", encoding="utf-8") as handle:
        first = next(csv.DictReader(handle))
    (workspace / first["image_path"]).write_bytes(b"corrupt")

    result = generate_synthetic.main(["--config", str(config), "--dry-run"])

    assert result == 0
    assert "projected_storage_bytes=1000" in capsys.readouterr().out


def test_validation_returns_nonzero_when_report_has_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches validation findings being reported with a successful process status."""
    config = _write_config(tmp_path / "config.yaml", tmp_path / "workspace", target_images=10)
    report = SimpleNamespace(image_count=10, label_count=10, error_count=1, warning_count=0)
    monkeypatch.setattr(validation_cli, "validate_dataset", lambda root, cfg: report)
    monkeypatch.setattr(validation_cli, "write_validation_report", lambda *args: None)
    monkeypatch.setattr(validation_cli, "write_statistics", lambda *args: None)
    monkeypatch.setattr(validation_cli, "make_contact_sheets", lambda *args: None)

    assert validation_cli.main(["--config", str(config)]) == 1
