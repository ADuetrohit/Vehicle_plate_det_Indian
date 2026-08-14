from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest
import yaml

from plate_dataset.builder import BuildManifest
from plate_dataset.manifests import write_manifest
from plate_dataset.storage import StorageEstimate
from scripts import generate_synthetic


def _write_config(path: Path, workspace: Path, *, target: int = 10) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "workspace": workspace.as_posix(),
                "seed": 20260814,
                "target_images": target,
                "max_images": target,
                "mh_share": [0.60, 0.70],
                "split": [0.80, 0.10, 0.10],
                "negative_share": [0.10, 0.10],
                "training_imgsz": 512,
                "min_box_at_training_size": [8, 4],
                "synthetic_only": True,
                "max_scene_edge": 320,
                "jpeg_quality": 88,
                "ocr_canvas": [256, 128],
                "min_free_gb": 0,
                "workers": 1,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _write_normalized_records(workspace: Path) -> None:
    rows = []
    for index in range(3):
        image = workspace / "raw" / f"source-{index}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (320, 180), (40 + index * 20, 80, 120)).save(image)
        rows.append(
            {
                "record_id": f"record-{index}",
                "image_path": image.relative_to(workspace).as_posix(),
                "width": 320,
                "height": 180,
                "boxes": [[0, 100, 120, 220, 155]],
                "source_id": "fixture/source",
                "source_family": f"family-{index}",
                "is_real": True,
                "plate_text": None,
                "tags": {"viewpoint": "rear"},
            }
        )
    manifest = workspace / "metadata" / "normalized_records.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _allow_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        generate_synthetic,
        "estimate_storage",
        lambda config, samples: StorageEstimate(projected_bytes=1, reserve_bytes=0),
    )


def _write_ocr_archive(path: Path, plate_text: str = "mh 12-ab 1234") -> Path:
    image_bytes = io.BytesIO()
    Image.new("RGB", (128, 64), "white").save(image_bytes, format="JPEG")
    row = {
        "image_path": "crops/existing.jpg",
        "plate_text": plate_text,
        "reviewed": "True",
        "rejected": "False",
    }
    csv_bytes = io.StringIO()
    writer = csv.DictWriter(csv_bytes, fieldnames=list(row), lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("license_plate_dataset/crops/existing.jpg", image_bytes.getvalue())
        archive.writestr("license_plate_dataset/annotations.csv", csv_bytes.getvalue())
        archive.writestr("license_plate_dataset/train_annotations.csv", csv_bytes.getvalue())
        empty = io.StringIO()
        empty_writer = csv.DictWriter(empty, fieldnames=list(row), lineterminator="\n")
        empty_writer.writeheader()
        archive.writestr("license_plate_dataset/val_annotations.csv", empty.getvalue())
    return path


def test_generation_requires_preserved_ocr_before_detector_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catches a missing OCR archive and labels CSV being noticed only after rendering."""
    workspace = tmp_path / "workspace"
    config_path = _write_config(tmp_path / "config.yaml", workspace)
    _write_normalized_records(workspace)
    _allow_storage(monkeypatch)
    build_called = False

    def fake_build(*args: object, **kwargs: object) -> BuildManifest:
        nonlocal build_called
        build_called = True
        manifest = workspace / "metadata" / "generation_manifest.csv"
        write_manifest([], manifest)
        return BuildManifest(20260814, 10, 0, 0, manifest)

    monkeypatch.setattr(generate_synthetic, "build_dataset", fake_build)

    result = generate_synthetic.main(["--config", str(config_path)])

    assert result == 2
    assert not build_called
    assert not (workspace / "detection").exists()
    assert "preserved OCR" in capsys.readouterr().err


def test_generation_imports_existing_ocr_and_forbids_its_text_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches the existing corpus being imported only after identities were rendered."""
    workspace = tmp_path / "workspace"
    config_path = _write_config(tmp_path / "config.yaml", workspace)
    _write_normalized_records(workspace)
    archive = _write_ocr_archive(tmp_path / "ocr.zip")
    _allow_storage(monkeypatch)

    def fake_build(*args: object, **kwargs: object) -> BuildManifest:
        labels = workspace / "ocr" / "labels.csv"
        assert labels.is_file()
        with labels.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["plate_text"] for row in rows] == ["MH12AB1234"]
        assert kwargs["forbidden_plate_texts"] == frozenset({"MH12AB1234"})
        manifest = workspace / "metadata" / "generation_manifest.csv"
        write_manifest([], manifest)
        return BuildManifest(20260814, 10, 0, 0, manifest)

    monkeypatch.setattr(generate_synthetic, "build_dataset", fake_build)

    result = generate_synthetic.main(
        ["--config", str(config_path), "--ocr-archive", str(archive)]
    )

    assert result == 0


def test_resumed_reject_uses_tombstone_and_cleans_obsolete_crop_after_ocr_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches checkpointed reject retries failing unknown-ID checks or leaking crops."""
    workspace = tmp_path / "workspace"
    config_path = _write_config(tmp_path / "config.yaml", workspace)
    _write_normalized_records(workspace)
    _allow_storage(monkeypatch)
    labels = workspace / "ocr" / "labels.csv"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text(
        "image_name,image_path,plate_text,split,source_id,synthetic,reconciliation\n"
        "existing.jpg,images/train/existing.jpg,MH12AB1234,train,existing_archive,false,preserved\n",
        encoding="utf-8",
    )
    rejected_id = "syn-aaaaaaaaaaaaaaaaaaaa"
    reject_file = tmp_path / "rejects.txt"
    reject_file.write_text(rejected_id + "\n", encoding="utf-8")
    tombstone = workspace / "metadata" / "rejected_generation_slots.json"
    tombstone.parent.mkdir(parents=True, exist_ok=True)
    tombstone.write_text(
        json.dumps(
            {
                "version": 1,
                "slots": {
                    "train:00000000": [
                        {"candidate_id": rejected_id, "attempt": 0}
                    ]
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    stale = (
        workspace / "detection" / "images" / "train" / f"{rejected_id}.jpg",
        workspace / "detection" / "labels" / "train" / f"{rejected_id}.txt",
        workspace / "ocr" / "images" / "train" / f"{rejected_id}.jpg",
    )
    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"obsolete")
    manifest_path = workspace / "metadata" / "generation_manifest.csv"
    write_manifest([], manifest_path)

    def fake_build(*args: object, **kwargs: object) -> BuildManifest:
        assert kwargs["rejected_ids"] == frozenset({rejected_id})
        write_manifest([], manifest_path)
        return BuildManifest(20260814, 10, 9, 1, manifest_path)

    real_merge = generate_synthetic.merge_ocr_labels

    def merge_while_obsolete_files_are_still_durable(*args: object, **kwargs: object):
        assert all(path.is_file() for path in stale)
        return real_merge(*args, **kwargs)

    monkeypatch.setattr(generate_synthetic, "build_dataset", fake_build)
    monkeypatch.setattr(
        generate_synthetic,
        "merge_ocr_labels",
        merge_while_obsolete_files_are_still_durable,
    )

    result = generate_synthetic.main(
        ["--config", str(config_path), "--reject-file", str(reject_file)]
    )

    assert result == 0
    assert all(not path.exists() for path in stale)
