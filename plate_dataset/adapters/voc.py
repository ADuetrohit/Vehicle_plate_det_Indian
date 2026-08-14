from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image

from . import is_plate_label, plate_text_from_label, record_id
from ..records import Box, ImageRecord


def parse_voc(xml_path: Path, image_path: Path, source_id: str) -> ImageRecord:
    with Image.open(image_path) as image:
        width, height = image.size
    root = ET.parse(xml_path).getroot()
    boxes: list[Box] = []
    recognized_text: str | None = None
    for item in root.findall("object"):
        name = (item.findtext("name") or "").strip()
        if not is_plate_label(name):
            continue
        bounds = item.find("bndbox")
        if bounds is None:
            raise ValueError(f"plate object has no bndbox: {xml_path}")
        try:
            box = Box(
                0,
                float(bounds.findtext("xmin", "")),
                float(bounds.findtext("ymin", "")),
                float(bounds.findtext("xmax", "")),
                float(bounds.findtext("ymax", "")),
            ).clip(width, height)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid VOC plate box: {xml_path}") from exc
        boxes.append(box)
        recognized_text = recognized_text or plate_text_from_label(name)
    if not boxes:
        raise ValueError(f"no number-plate boxes in {xml_path}")
    identifier = record_id(source_id, image_path)
    return ImageRecord(
        record_id=identifier,
        image_path=image_path,
        width=width,
        height=height,
        boxes=tuple(boxes),
        source_id=source_id,
        source_family=identifier,
        is_real=True,
        plate_text=recognized_text,
        tags={"annotation_format": "voc"},
    )

