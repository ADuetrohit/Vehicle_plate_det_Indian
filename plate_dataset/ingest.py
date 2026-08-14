from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .adapters.voc import parse_voc
from .adapters.yolo import parse_yolo
from .dedupe import perceptual_family
from .records import ImageRecord
from .sources import SourceSpec


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ingest_source(source_root: Path, spec: SourceSpec) -> list[ImageRecord]:
    images = sorted(
        path for path in source_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    records: list[ImageRecord] = []
    for image_path in images:
        label_path = _matching_label(source_root, image_path, spec.annotation_format)
        if label_path is None:
            continue
        if spec.annotation_format == "yolo":
            record = parse_yolo(label_path, image_path, spec.slug)
        else:
            record = parse_voc(label_path, image_path, spec.slug)
        records.append(record)
    if not records:
        raise ValueError(f"no matching image-label pairs in {source_root}")
    families = perceptual_family(records)
    return [replace(record, source_family=families[record.record_id]) for record in records]


def _matching_label(
    source_root: Path, image_path: Path, annotation_format: str
) -> Path | None:
    suffix = ".txt" if annotation_format == "yolo" else ".xml"
    candidates = [image_path.with_suffix(suffix)]
    relative = image_path.relative_to(source_root)
    parts = list(relative.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels" if suffix == ".txt" else "annotations"
        candidates.append((source_root / Path(*parts)).with_suffix(suffix))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
