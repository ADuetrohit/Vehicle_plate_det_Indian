from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .adapters.voc import parse_voc
from .adapters.yolo import parse_yolo
from .records import ImageRecord
from .sources import SourceSpec


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class RejectedSourceRecord:
    source_id: str
    image_path: Path
    reason: str


def _relative_reason(error: Exception, source_root: Path, *paths: Path) -> str:
    reason = str(error)
    for path in paths:
        try:
            relative = path.relative_to(source_root).as_posix()
        except ValueError:
            continue
        reason = reason.replace(str(path), relative).replace(path.as_posix(), relative)
    return reason


def ingest_source(
    source_root: Path,
    spec: SourceSpec,
    *,
    rejected_records: list[RejectedSourceRecord] | None = None,
) -> list[ImageRecord]:
    images = sorted(
        path for path in source_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    records: list[ImageRecord] = []
    for image_path in images:
        label_path = _matching_label(source_root, image_path, spec.annotation_format)
        if label_path is None:
            continue
        try:
            if spec.annotation_format == "yolo":
                record = parse_yolo(label_path, image_path, spec.slug)
            else:
                record = parse_voc(label_path, image_path, spec.slug)
            with Image.open(image_path) as image:
                image.convert("RGB").load()
        except (OSError, SyntaxError, ValueError) as error:
            if rejected_records is not None:
                rejected_records.append(
                    RejectedSourceRecord(
                        source_id=spec.slug,
                        image_path=image_path,
                        reason=_relative_reason(
                            error, source_root, label_path, image_path
                        ),
                    )
                )
            continue
        records.append(record)
    return records


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
