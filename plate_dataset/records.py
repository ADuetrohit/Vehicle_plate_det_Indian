from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Box:
    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.class_id != 0:
            raise ValueError("number_plate must use class ID 0")
        coordinates = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("box coordinates must be finite")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("box must have positive area")

    def clip(self, width: int, height: int) -> "Box":
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        return Box(
            class_id=self.class_id,
            x_min=max(0.0, min(self.x_min, float(width))),
            y_min=max(0.0, min(self.y_min, float(height))),
            x_max=max(0.0, min(self.x_max, float(width))),
            y_max=max(0.0, min(self.y_max, float(height))),
        )

    def to_yolo(
        self, width: int, height: int
    ) -> tuple[int, float, float, float, float]:
        box = self.clip(width, height)
        box_width = box.x_max - box.x_min
        box_height = box.y_max - box.y_min
        return (
            box.class_id,
            (box.x_min + box.x_max) / (2.0 * width),
            (box.y_min + box.y_max) / (2.0 * height),
            box_width / width,
            box_height / height,
        )


@dataclass(frozen=True)
class ImageRecord:
    record_id: str
    image_path: Path
    width: int
    height: int
    boxes: tuple[Box, ...]
    source_id: str
    source_family: str
    is_real: bool
    plate_text: str | None
    tags: Mapping[str, str]
