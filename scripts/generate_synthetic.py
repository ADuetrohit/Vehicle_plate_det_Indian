from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plate_dataset.builder import build_dataset
from plate_dataset.config import load_config
from plate_dataset.ocr_import import import_existing_ocr, write_ocr_labels
from plate_dataset.records import Box, ImageRecord


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build the deterministic hybrid plate dataset."
    )
    result.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    result.add_argument("--dry-run", action="store_true")
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


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    record_path = config.workspace / "metadata" / "normalized_records.jsonl"
    if not record_path.is_file():
        print(
            "Normalized records are missing. Run convert_annotations.py first.",
            file=sys.stderr,
        )
        return 2
    records = _load_records(record_path, config.workspace)
    rejected = set()
    if args.reject_file and args.reject_file.is_file():
        rejected = {
            line.strip()
            for line in args.reject_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    if args.dry_run:
        print(f"source_records={len(records)}")
        print(f"target_images={config.target_images}")
        print(f"rejected_ids={len(rejected)}")
        return 0
    manifest_path = config.workspace / "metadata" / "generation_manifest.csv"
    if not args.resume and manifest_path.exists():
        print(
            "Existing generation manifest found; use --resume or a new workspace.",
            file=sys.stderr,
        )
        return 2
    if rejected:
        print("Rejected IDs will be replaced during this resumable build.")
    manifest = build_dataset(config, records, config.workspace)

    archive = args.ocr_archive or config.workspace.parent / "archive.zip"
    labels = config.workspace / "ocr" / "labels.csv"
    if archive.is_file() and not labels.is_file():
        ocr_records = import_existing_ocr(archive, config.workspace / "ocr")
        write_ocr_labels(ocr_records, labels)
        print(f"existing_ocr_crops={len(ocr_records)}")
    print(
        f"dataset_images={manifest.generated_count} reused={manifest.reused_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
