from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plate_dataset.archives import safe_extract_zip
from plate_dataset.config import BuildConfig, load_config
from plate_dataset.dedupe import perceptual_family
from plate_dataset.ingest import RejectedSourceRecord, ingest_source
from plate_dataset.records import ImageRecord
from plate_dataset.sources import SourceSpec, source_registry


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Extract licensed sources and normalize annotations."
    )
    result.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    result.add_argument("--dry-run", action="store_true")
    return result


def _record_json(record, workspace: Path) -> dict[str, object]:
    try:
        image_path = record.image_path.relative_to(workspace).as_posix()
    except ValueError:
        image_path = record.image_path.as_posix()
    return {
        "record_id": record.record_id,
        "image_path": image_path,
        "width": record.width,
        "height": record.height,
        "boxes": [
            [box.class_id, box.x_min, box.y_min, box.x_max, box.y_max]
            for box in record.boxes
        ],
        "source_id": record.source_id,
        "source_family": record.source_family,
        "is_real": record.is_real,
        "plate_text": record.plate_text,
        "tags": dict(record.tags),
    }


def _license_is_allowed(spec: SourceSpec, license_dir: Path) -> bool:
    if spec.license_status == "allowed":
        return True
    if spec.license_status != "verify":
        return False
    decision_path = license_dir / f"{spec.slug.replace('/', '__')}.yaml"
    if not decision_path.is_file():
        return False
    try:
        decision = yaml.safe_load(decision_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(decision, dict) and decision.get("decision") == "allowed"


def _resized_geometry(
    record: ImageRecord, config: BuildConfig
) -> tuple[float, int, int]:
    resize_scale = min(
        1.0, config.max_scene_edge / max(record.width, record.height)
    )
    resized_width = max(1, round(record.width * resize_scale))
    resized_height = max(1, round(record.height * resize_scale))
    return resize_scale, resized_width, resized_height


def _scaled_primary_size(
    record: ImageRecord, config: BuildConfig
) -> tuple[float, float] | None:
    resize_scale, resized_width, resized_height = _resized_geometry(record, config)
    candidates: list[tuple[float, float]] = []
    for box in record.boxes:
        scaled_width = (
            min(box.x_max * resize_scale, resized_width)
            - max(box.x_min * resize_scale, 0.0)
        )
        scaled_height = (
            min(box.y_max * resize_scale, resized_height)
            - max(box.y_min * resize_scale, 0.0)
        )
        if scaled_width >= 8 and scaled_height >= 4:
            candidates.append((scaled_width, scaled_height))
    if not candidates:
        return None
    return max(candidates, key=lambda size: size[0] * size[1])


def _training_anchor_rejection_reason(
    record: ImageRecord, config: BuildConfig
) -> str | None:
    primary_size = _scaled_primary_size(record, config)
    if primary_size is None:
        return (
            "no primary anchor survives the 8x4-pixel compositing gate after "
            f"max-scene-edge {config.max_scene_edge} resize"
        )
    _, resized_width, resized_height = _resized_geometry(record, config)
    training_scale = config.training_imgsz / max(resized_width, resized_height)
    minimum_width, minimum_height = config.min_box_at_training_size
    if (
        primary_size[0] * training_scale >= minimum_width
        and primary_size[1] * training_scale >= minimum_height
    ):
        return None
    return (
        f"no primary anchor reaches {minimum_width}x{minimum_height} pixels at "
        f"{config.training_imgsz}-pixel training size after max-scene-edge "
        f"{config.max_scene_edge} resize"
    )


def _relative_path(path: Path, workspace: Path) -> str:
    try:
        return path.relative_to(workspace).as_posix()
    except ValueError:
        return path.as_posix()


def _write_rejected_records(
    rejected_records: list[RejectedSourceRecord], output: Path, workspace: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    rows = {
        (
            rejected.source_id,
            _relative_path(rejected.image_path, workspace),
            rejected.reason,
        )
        for rejected in rejected_records
    }
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("source_id", "image_path", "reason")
            )
            writer.writeheader()
            for source_id, image_path, reason in sorted(
                rows, key=lambda row: (row[1], row[0], row[2])
            ):
                writer.writerow(
                    {
                        "source_id": source_id,
                        "image_path": image_path,
                        "reason": reason,
                    }
                )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    raw = config.workspace / "raw"
    license_dir = config.workspace / "metadata" / "licenses"
    specs = {
        spec.slug.replace("/", "__"): spec
        for spec in source_registry()
        if _license_is_allowed(spec, license_dir)
    }
    discovered = sorted(raw.glob("*/*.zip")) if raw.exists() else []
    archives = [archive for archive in discovered if archive.parent.name in specs]
    if args.dry_run:
        print(f"archives={len(archives)}")
        print(f"excluded_archives={len(discovered) - len(archives)}")
        print(
            "normalized_manifest="
            f"{config.workspace / 'metadata' / 'normalized_records.jsonl'}"
        )
        return 0
    if not archives:
        print(
            "No downloaded source archives found. Run download_sources.py first.",
            file=sys.stderr,
        )
        return 2

    records: list[ImageRecord] = []
    rejected_records: list[RejectedSourceRecord] = []
    for archive in archives:
        spec = specs.get(archive.parent.name)
        if spec is None:
            continue
        destination = raw / "extracted" / archive.parent.name
        safe_extract_zip(archive, destination)
        records.extend(
            ingest_source(
                destination, spec, rejected_records=rejected_records
            )
        )
    eligible_records: list[ImageRecord] = []
    for record in records:
        rejection_reason = _training_anchor_rejection_reason(record, config)
        if rejection_reason is None:
            eligible_records.append(record)
        else:
            rejected_records.append(
                RejectedSourceRecord(
                    source_id=record.source_id,
                    image_path=record.image_path,
                    reason=rejection_reason,
                )
            )
    records = eligible_records
    families = perceptual_family(records)
    records = [
        replace(record, source_family=families[record.record_id])
        for record in records
    ]
    rejected_output = (
        config.workspace / "metadata" / "rejected_source_records.csv"
    )
    _write_rejected_records(rejected_records, rejected_output, config.workspace)
    if not records:
        print("No valid image-label pairs were normalized.", file=sys.stderr)
        return 2

    output = config.workspace / "metadata" / "normalized_records.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in sorted(records, key=lambda item: item.record_id):
                handle.write(
                    json.dumps(_record_json(record, config.workspace), sort_keys=True)
                    + "\n"
                )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"normalized_records={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
