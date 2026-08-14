from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image

from .manifests import sha256_file
from .records import Box


_DEFAULT_LABEL_FIELDS = (
    "image_name",
    "image_path",
    "plate_text",
    "split",
    "source_id",
    "synthetic",
    "reconciliation",
    "output_id",
)
_NEUTRAL_RGB = (128, 128, 128)


def export_ocr_crop(
    image: np.ndarray | Image.Image,
    box: Box,
    output_path: Path,
    canvas: tuple[int, int],
    quality: int,
) -> str:
    """Export an aspect-preserving, padded OCR crop and return its file checksum."""
    canvas_width, canvas_height = canvas
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("OCR canvas dimensions must be positive")
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")

    source = _rgb_image(image)
    width, height = source.size
    crop = _padded_crop(source, box, width, height)
    scale = min(canvas_width / crop.width, canvas_height / crop.height)
    resized = crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
        Image.Resampling.LANCZOS,
    )
    result = Image.new("RGB", canvas, _NEUTRAL_RGB)
    offset = ((canvas_width - resized.width) // 2, (canvas_height - resized.height) // 2)
    result.paste(resized, offset)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    try:
        result.save(temporary, format="JPEG", quality=quality, subsampling=0)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(output_path)


def merge_ocr_labels(
    existing_csv: Path,
    synthetic_rows: Iterable[Mapping[str, object]],
    output_csv: Path,
) -> int:
    """Preserve imported labels and append deterministic, non-duplicated synthetic rows."""
    existing_fields, existing_rows = _read_rows(Path(existing_csv))
    synthetic = [{key: str(value) for key, value in row.items()} for row in synthetic_rows]
    fieldnames = _fieldnames(existing_fields, synthetic)
    seen = {_stable_key(row) for row in existing_rows}
    new_rows: list[dict[str, str]] = []
    for row in sorted(synthetic, key=_synthetic_sort_key):
        key = _stable_key(row)
        if key in seen:
            continue
        seen.add(key)
        new_rows.append(row)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(f".{output_csv.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in [*existing_rows, *new_rows]:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
        os.replace(temporary, output_csv)
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(existing_rows) + len(new_rows)


def _rgb_image(image: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("scene must be an RGB uint8 array")
    return Image.fromarray(array, mode="RGB")


def _padded_crop(image: Image.Image, box: Box, width: int, height: int) -> Image.Image:
    x_padding = (box.x_max - box.x_min) * 0.10
    y_padding = (box.y_max - box.y_min) * 0.10
    left = max(0, math.floor(box.x_min - x_padding))
    top = max(0, math.floor(box.y_min - y_padding))
    right = min(width, math.ceil(box.x_max + x_padding))
    bottom = min(height, math.ceil(box.y_max + y_padding))
    if right <= left or bottom <= top:
        raise ValueError("OCR crop does not overlap the scene")
    return image.crop((left, top, right, bottom))


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _fieldnames(existing_fields: list[str], synthetic_rows: list[dict[str, str]]) -> list[str]:
    fields = list(existing_fields) or list(_DEFAULT_LABEL_FIELDS)
    for name in _DEFAULT_LABEL_FIELDS:
        if name not in fields and any(name in row for row in synthetic_rows):
            fields.append(name)
    for name in sorted({key for row in synthetic_rows for key in row} - set(fields)):
        fields.append(name)
    return fields


def _stable_key(row: Mapping[str, str]) -> tuple[str, str]:
    return row.get("source_id", ""), row.get("image_name", "")


def _synthetic_sort_key(row: Mapping[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("split", ""),
        row.get("output_id", ""),
        row.get("source_id", ""),
        row.get("image_name", ""),
    )
