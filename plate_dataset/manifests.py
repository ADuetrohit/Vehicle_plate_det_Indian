from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Iterable, Mapping


MANIFEST_FIELDS = (
    "output_id",
    "split",
    "origin",
    "source_id",
    "source_family",
    "image_path",
    "label_path",
    "image_sha256",
    "label_sha256",
    "negative",
    "state",
    "plate_text",
    "vehicle_type",
    "viewpoint",
    "plate_style",
    "plate_layout",
    "effect",
    "ocr_eligible",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["output_id"]: row for row in csv.DictReader(handle)}


def write_manifest(rows: Iterable[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: str(item["output_id"])):
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
    os.replace(temporary, path)
