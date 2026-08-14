from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plate_dataset.builder import (
    _compatible_previous_rows,
    _load_rejection_tombstones,
    _synthetic_specs,
    build_dataset,
)
from plate_dataset.config import BuildConfig, load_config
from plate_dataset.manifests import load_manifest
from plate_dataset.ocr_export import merge_ocr_labels
from plate_dataset.ocr_import import import_existing_ocr, write_ocr_labels
from plate_dataset.quotas import generation_quotas
from plate_dataset.records import Box, ImageRecord
from plate_dataset.registration import normalize_registration
from plate_dataset.storage import (
    InsufficientStorage,
    StorageEstimate,
    estimate_storage,
    require_storage,
)


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build the deterministic hybrid plate dataset."
    )
    result.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    result.add_argument("--dry-run", action="store_true")
    result.add_argument(
        "--workers",
        type=_nonnegative_int,
        help="Parallel render workers; 0 selects a bounded automatic count.",
    )
    resume = result.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true", default=True)
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    result.add_argument(
        "--reject-file", type=Path, help="Output IDs rejected during visual QA."
    )
    result.add_argument("--ocr-archive", type=Path)
    return result


def _load_records(path: Path, workspace: Path) -> list[ImageRecord]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = workspace / image_path
        records.append(
            ImageRecord(
                record_id=row["record_id"],
                image_path=image_path,
                width=int(row["width"]),
                height=int(row["height"]),
                boxes=tuple(Box(*values) for values in row["boxes"]),
                source_id=row["source_id"],
                source_family=row["source_family"],
                is_real=bool(row["is_real"]),
                plate_text=row.get("plate_text"),
                tags=row.get("tags", {}),
            )
        )
    return records


def _storage_preflight(
    config: BuildConfig, records: list[ImageRecord], pending_outputs: int
) -> tuple[StorageEstimate, int]:
    samples = sorted({record.image_path for record in records})
    full_estimate = estimate_storage(config, samples)
    estimate = StorageEstimate(
        projected_bytes=math.ceil(
            full_estimate.projected_bytes
            * min(config.target_images, pending_outputs)
            / config.target_images
        ),
        reserve_bytes=full_estimate.reserve_bytes,
    )
    free_bytes = shutil.disk_usage(config.workspace).free
    require_storage(estimate, free_bytes)
    return estimate, free_bytes


def _artifact_path(
    workspace: Path, row: dict[str, str], field: str
) -> Path:
    output_id = row.get("output_id", "")
    split = row.get("split", "")
    if (
        re.fullmatch(r"(?:syn|real)-[0-9a-f]{20}", output_id) is None
        or split not in {"train", "val", "test"}
    ):
        raise ValueError("rejected output ID or split is not canonical")
    if field == "image_path":
        expected = PurePosixPath("detection", "images", split, f"{output_id}.jpg")
    elif field == "label_path":
        expected = PurePosixPath("detection", "labels", split, f"{output_id}.txt")
    elif field == "ocr_path":
        expected = PurePosixPath("ocr", "images", split, f"{output_id}.jpg")
    else:
        raise ValueError(f"unsupported artifact field: {field}")
    relative = PurePosixPath(row.get(field, ""))
    if relative.is_absolute() or ".." in relative.parts or relative != expected:
        raise ValueError(
            f"refusing noncanonical rejected artifact path: {row.get(field, '')}"
        )
    return workspace.joinpath(*relative.parts)


def _required_artifacts(
    workspace: Path, row: dict[str, str]
) -> tuple[Path, ...]:
    fields = ["image_path", "label_path"]
    if row.get("negative", "").lower() != "true":
        fields.append("ocr_path")
    return tuple(_artifact_path(workspace, row, field) for field in fields)


def _pending_output_count(
    config: BuildConfig,
    records: list[ImageRecord],
    previous: dict[str, dict[str, str]],
    rejected: frozenset[str],
    tombstones: dict[str, list[dict[str, object]]],
    forbidden_plate_texts: frozenset[str],
) -> int:
    if not previous:
        return config.target_images
    specs = _synthetic_specs(
        config,
        records,
        rejected,
        forbidden_plate_texts=forbidden_plate_texts,
        tombstones=tombstones,
    )
    compatible = _compatible_previous_rows(specs, previous, config.workspace)
    return config.target_images - len(compatible)


def _rejected_artifacts(
    workspace: Path,
    previous: dict[str, dict[str, str]],
    rejected: frozenset[str],
) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for output_id in sorted(rejected):
        paths.update(_required_artifacts(workspace, previous[output_id]))
        row = previous[output_id]
        split = row.get("split", "")
        if split in {"train", "val", "test"}:
            paths.add(workspace / "ocr" / "images" / split / f"{output_id}.jpg")
    return tuple(sorted(paths))


def _tombstoned_rejected_artifacts(
    workspace: Path,
    tombstones: dict[str, list[dict[str, object]]],
    rejected: frozenset[str],
) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for logical_slot, entries in tombstones.items():
        split = logical_slot.partition(":")[0]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"invalid rejection tombstone slot: {logical_slot}")
        for entry in entries:
            output_id = str(entry["candidate_id"])
            if output_id not in rejected:
                continue
            if re.fullmatch(r"syn-[0-9a-f]{20}", output_id) is None:
                raise ValueError(f"invalid rejection tombstone candidate: {output_id}")
            paths.update(
                {
                    workspace
                    / "detection"
                    / "images"
                    / split
                    / f"{output_id}.jpg",
                    workspace
                    / "detection"
                    / "labels"
                    / split
                    / f"{output_id}.txt",
                    workspace / "ocr" / "images" / split / f"{output_id}.jpg",
                }
            )
    return tuple(sorted(paths))


def _synthetic_ocr_rows(manifest_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ocr_path = PurePosixPath(row.get("ocr_path", ""))
            if not ocr_path.name or not row.get("plate_text"):
                continue
            try:
                relative_path = ocr_path.relative_to("ocr").as_posix()
            except ValueError as error:
                raise ValueError(f"invalid manifest OCR path: {ocr_path}") from error
            rows.append(
                {
                    "image_name": ocr_path.name,
                    "image_path": relative_path,
                    "plate_text": row["plate_text"],
                    "split": row["split"],
                    "source_id": "synthetic",
                    "synthetic": "true",
                    "reconciliation": "generated",
                    "output_id": row["output_id"],
                }
            )
    return rows


def _write_existing_only(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [
            row
            for row in reader
            if row.get("synthetic", "").strip().lower() != "true"
        ]
    if not fieldnames:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _merge_ocr_corpora(config, archive: Path, manifest_path: Path) -> int:
    labels = config.workspace / "ocr" / "labels.csv"
    temporary_existing = labels.with_name(f".{labels.stem}.existing.tmp.csv")
    try:
        _write_existing_only(labels, temporary_existing)
        return merge_ocr_labels(
            temporary_existing,
            _synthetic_ocr_rows(manifest_path),
            labels,
        )
    finally:
        if temporary_existing.exists():
            temporary_existing.unlink()


def _existing_ocr_texts(labels: Path) -> frozenset[str]:
    with labels.open(newline="", encoding="utf-8") as handle:
        try:
            rows = list(csv.DictReader(handle))
        except csv.Error as error:
            raise ValueError(f"cannot read preserved OCR labels: {error}") from error
    return frozenset(
        normalized
        for row in rows
        if row.get("synthetic", "").strip().lower() != "true"
        and (normalized := normalize_registration(row.get("plate_text", "")))
    )


def _prepare_existing_ocr_corpus(config: BuildConfig, archive: Path) -> frozenset[str]:
    labels = config.workspace / "ocr" / "labels.csv"
    if not labels.is_file():
        records = import_existing_ocr(archive, config.workspace / "ocr")
        write_ocr_labels(records, labels)
    texts = _existing_ocr_texts(labels)
    if not texts:
        raise ValueError("preserved OCR labels contain no accepted registrations")
    return texts


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if args.workers is not None:
        config = replace(config, workers=args.workers)
    record_path = config.workspace / "metadata" / "normalized_records.jsonl"
    if not record_path.is_file():
        print(
            "Normalized records are missing. Run convert_annotations.py first.",
            file=sys.stderr,
        )
        return 2
    records = _load_records(record_path, config.workspace)
    manifest_path = config.workspace / "metadata" / "generation_manifest.csv"
    previous = load_manifest(manifest_path)
    tombstone_path = config.workspace / "metadata" / "rejected_generation_slots.json"
    try:
        tombstones = _load_rejection_tombstones(tombstone_path)
    except ValueError as error:
        print(f"Rejected output tombstones are invalid: {error}", file=sys.stderr)
        return 2
    rejected: frozenset[str] = frozenset()
    rejected_artifacts: tuple[Path, ...] = ()
    if args.reject_file:
        if not args.reject_file.is_file():
            print(f"Reject file not found: {args.reject_file}", file=sys.stderr)
            return 2
        try:
            rejected = frozenset(
                line.strip()
                for line in args.reject_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except OSError as error:
            print(f"Reject file could not be read: {error}", file=sys.stderr)
            return 2
        tombstoned_ids = {
            str(entry["candidate_id"])
            for entries in tombstones.values()
            for entry in entries
        }
        unknown = sorted(rejected - previous.keys() - tombstoned_ids)
        if unknown:
            print(f"Unknown rejected output IDs: {','.join(unknown)}", file=sys.stderr)
            return 2
        try:
            rejected_artifacts = _rejected_artifacts(
                config.workspace, previous, rejected & previous.keys()
            )
            rejected_artifacts = tuple(
                sorted(
                    set(rejected_artifacts)
                    | set(
                        _tombstoned_rejected_artifacts(
                            config.workspace, tombstones, rejected
                        )
                    )
                )
            )
        except ValueError as error:
            print(f"Rejected artifact cleanup refused: {error}", file=sys.stderr)
            return 2
    ocr_labels_path = config.workspace / "ocr" / "labels.csv"
    try:
        pending_forbidden_texts = (
            _existing_ocr_texts(ocr_labels_path)
            if ocr_labels_path.is_file()
            else frozenset()
        )
    except (OSError, ValueError) as error:
        print(f"Preserved OCR preparation failed: {error}", file=sys.stderr)
        return 2
    pending_outputs = _pending_output_count(
        config,
        records,
        previous,
        rejected,
        tombstones,
        pending_forbidden_texts,
    )
    try:
        estimate, free_bytes = _storage_preflight(
            config, records, pending_outputs
        )
    except (InsufficientStorage, OSError, ValueError) as error:
        print(f"Storage preflight failed: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        quotas = generation_quotas(config)
        print(f"source_records={len(records)}")
        print(f"target_images={config.target_images}")
        for split in ("train", "val", "test"):
            print(f"split_{split}={quotas[split].total}")
        print(f"projected_storage_bytes={estimate.projected_bytes}")
        print(f"storage_reserve_bytes={estimate.reserve_bytes}")
        print(f"storage_required_bytes={estimate.required_bytes}")
        print(f"free_storage_bytes={free_bytes}")
        print(f"rejected_ids={len(rejected)}")
        return 0
    if not args.resume and manifest_path.exists():
        print(
            "Existing generation manifest found; use --resume or a new workspace.",
            file=sys.stderr,
        )
        return 2
    archive = args.ocr_archive or config.workspace.parent / "archive.zip"
    if not ocr_labels_path.is_file() and not archive.is_file():
        print(
            "Required preserved OCR data is unavailable: provide --ocr-archive or an existing "
            "ocr/labels.csv before generation.",
            file=sys.stderr,
        )
        return 2
    if rejected:
        print("Rejected IDs will be replaced during this resumable build.")

    try:
        forbidden_plate_texts = _prepare_existing_ocr_corpus(config, archive)
    except (OSError, ValueError) as error:
        print(f"Preserved OCR preparation failed: {error}", file=sys.stderr)
        return 2

    def progress(completed: int, total: int) -> None:
        if completed % 500 == 0:
            print(f"progress={completed}/{total}", flush=True)

    manifest = build_dataset(
        config,
        records,
        config.workspace,
        rejected_ids=rejected,
        progress=progress,
        forbidden_plate_texts=forbidden_plate_texts,
    )

    try:
        ocr_labels = _merge_ocr_corpora(config, archive, manifest.manifest_path)
    except (OSError, ValueError) as error:
        print(f"OCR label merge failed: {error}", file=sys.stderr)
        return 2
    current_ids = load_manifest(manifest.manifest_path).keys()
    retained_rejections = sorted(rejected & current_ids)
    if retained_rejections:
        print(
            f"Builder retained rejected output IDs: {','.join(retained_rejections)}",
            file=sys.stderr,
        )
        return 2
    try:
        for path in rejected_artifacts:
            if path.is_file():
                path.unlink()
    except OSError as error:
        print(f"Rejected artifact cleanup failed: {error}", file=sys.stderr)
        return 2
    print(f"ocr_labels={ocr_labels}")
    print(
        f"dataset_images={manifest.generated_count} reused={manifest.reused_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
