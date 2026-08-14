from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from . import record_id
from ..records import Box, ImageRecord


def parse_yolo(
    label_path: Path,
    image_path: Path,
    source_id: str,
    *,
    allow_negative: bool = False,
) -> ImageRecord:
    with Image.open(image_path) as image:
        width, height = image.size
    boxes: list[Box] = []
    rows = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows and not allow_negative:
        raise ValueError(f"empty label without explicit negative flag: {label_path}")
    for row in rows:
        parts = row.split()
        try:
            if len(parts) != 5 or int(parts[0]) != 0:
                raise ValueError
            center_x, center_y, box_width, box_height = map(float, parts[1:])
            values = (center_x, center_y, box_width, box_height)
            if not all(math.isfinite(value) for value in values):
                raise ValueError
            if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
                raise ValueError
            if not (0.0 < box_width <= 1.0 and 0.0 < box_height <= 1.0):
                raise ValueError
            half_width = box_width * width / 2.0
            half_height = box_height * height / 2.0
            box = Box(
                0,
                center_x * width - half_width,
                center_y * height - half_height,
                center_x * width + half_width,
                center_y * height + half_height,
            ).clip(width, height)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid YOLO label row in {label_path}: {row}") from exc
        boxes.append(box)
    identifier = record_id(source_id, image_path)
    tags = {"annotation_format": "yolo"}
    if not boxes:
        tags["negative"] = "true"
    return ImageRecord(
        record_id=identifier,
        image_path=image_path,
        width=width,
        height=height,
        boxes=tuple(boxes),
        source_id=source_id,
        source_family=identifier,
        is_real=True,
        plate_text=None,
        tags=tags,
    )

