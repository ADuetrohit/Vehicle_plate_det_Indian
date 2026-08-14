from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plate_dataset.archives import safe_extract_zip
from plate_dataset.config import load_config
from plate_dataset.ingest import ingest_source
from plate_dataset.sources import source_registry


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


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    raw = config.workspace / "raw"
    specs = {spec.slug.replace("/", "__"): spec for spec in source_registry()}
    archives = sorted(raw.glob("*/*.zip")) if raw.exists() else []
    if args.dry_run:
        print(f"archives={len(archives)}")
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

    records = []
    for archive in archives:
        spec = specs.get(archive.parent.name)
        if spec is None:
            continue
        destination = raw / "extracted" / archive.parent.name
        safe_extract_zip(archive, destination)
        try:
            records.extend(ingest_source(destination, spec))
        except ValueError as error:
            print(f"{spec.slug}: {error}", file=sys.stderr)
            return 2
    if not records:
        print("No valid image-label pairs were normalized.", file=sys.stderr)
        return 2

    output = config.workspace / "metadata" / "normalized_records.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item.record_id):
            handle.write(
                json.dumps(_record_json(record, config.workspace), sort_keys=True)
                + "\n"
            )
    os.replace(temporary, output)
    print(f"normalized_records={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
