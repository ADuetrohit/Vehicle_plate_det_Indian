from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .registration import PlateIdentity


PlateCategory = Literal[
    "private",
    "commercial",
    "electric_private",
    "electric_commercial",
    "temporary",
]
PlateLayout = Literal["single", "double"]


@dataclass(frozen=True)
class PlateStyle:
    category: PlateCategory
    layout: PlateLayout
    wear: float = 0.12

    def __post_init__(self) -> None:
        if self.category not in {
            "private",
            "commercial",
            "electric_private",
            "electric_commercial",
            "temporary",
        }:
            raise ValueError(f"unsupported plate category: {self.category}")
        if self.layout not in {"single", "double"}:
            raise ValueError(f"unsupported plate layout: {self.layout}")
        if not 0.0 <= self.wear <= 1.0:
            raise ValueError("wear must be between zero and one")


@dataclass(frozen=True)
class RenderedPlate:
    image: Image.Image
    identity: PlateIdentity
    style: PlateStyle
    ocr_eligible: bool


COLORS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "private": ((245, 245, 245), (10, 10, 10)),
    "commercial": ((255, 204, 0), (10, 10, 10)),
    "electric_private": ((0, 140, 70), (250, 250, 250)),
    "electric_commercial": ((0, 140, 70), (255, 220, 0)),
    "temporary": ((255, 220, 0), (180, 0, 0)),
}


def discover_font_paths() -> list[Path]:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\bahnschrift.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ]
    found = [path for path in candidates if path.is_file()]
    if not found:
        raise FileNotFoundError(
            "no supported bold Latin font found; configure an Arial, Bahnschrift, "
            "DejaVu Sans, or Liberation Sans bold font"
        )
    return found


def render_plate(
    identity: PlateIdentity,
    style: PlateStyle,
    font_paths: Sequence[Path],
    rng: np.random.Generator,
) -> RenderedPlate:
    if not font_paths:
        raise ValueError("at least one font path is required")
    size = (520, 110) if style.layout == "single" else (280, 200)
    background, foreground = COLORS[style.category]
    image = Image.new("RGB", size, color=background)
    draw = ImageDraw.Draw(image)
    width, height = size
    draw.rounded_rectangle(
        (8, 8, width - 9, height - 9),
        radius=5,
        outline=foreground,
        width=max(2, width // 150),
    )
    lines = (
        identity.display_lines
        if style.layout == "single"
        else (f"{identity.state} {identity.district}", f"{identity.series} {identity.number}")
    )
    font_path = Path(font_paths[int(rng.integers(0, len(font_paths)))])
    _draw_centered_lines(draw, size, lines, font_path, foreground)
    screw_radius = max(3, height // 28)
    for center_x in (22, width - 23):
        center_y = height // 2
        draw.ellipse(
            (
                center_x - screw_radius,
                center_y - screw_radius,
                center_x + screw_radius,
                center_y + screw_radius,
            ),
            fill=(105, 105, 105),
            outline=(35, 35, 35),
        )
    if style.wear > 0:
        _apply_wear(draw, size, style.wear, rng, foreground)
    return RenderedPlate(
        image=image,
        identity=identity,
        style=style,
        ocr_eligible=style.wear <= 0.65,
    )


def _draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    size: tuple[int, int],
    lines: Sequence[str],
    font_path: Path,
    color: tuple[int, int, int],
) -> None:
    width, height = size
    available_width = width - 64
    available_height = height - 28
    max_font = 72 if len(lines) == 1 else 70
    font = _fit_font(draw, lines, font_path, max_font, available_width, available_height)
    boxes = [draw.textbbox((0, 0), line, font=font, stroke_width=1) for line in lines]
    heights = [box[3] - box[1] for box in boxes]
    spacing = 8 if len(lines) > 1 else 0
    total_height = sum(heights) + spacing * (len(lines) - 1)
    y = (height - total_height) / 2
    for line, box, line_height in zip(lines, boxes, heights):
        line_width = box[2] - box[0]
        x = (width - line_width) / 2
        draw.text(
            (x, y - box[1]),
            line,
            font=font,
            fill=color,
            stroke_width=1,
            stroke_fill=color,
        )
        y += line_height + spacing


def _fit_font(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[str],
    font_path: Path,
    maximum: int,
    available_width: int,
    available_height: int,
) -> ImageFont.FreeTypeFont:
    for size in range(maximum, 13, -1):
        font = ImageFont.truetype(str(font_path), size=size)
        boxes = [draw.textbbox((0, 0), line, font=font, stroke_width=1) for line in lines]
        widest = max(box[2] - box[0] for box in boxes)
        total_height = sum(box[3] - box[1] for box in boxes) + 8 * (len(lines) - 1)
        if widest <= available_width and total_height <= available_height:
            return font
    raise ValueError("registration text does not fit plate template")


def _apply_wear(
    draw: ImageDraw.ImageDraw,
    size: tuple[int, int],
    wear: float,
    rng: np.random.Generator,
    foreground: tuple[int, int, int],
) -> None:
    width, height = size
    mark_count = int(4 + wear * 45)
    muted = tuple(int(channel * 0.55 + 105) for channel in foreground)
    for _ in range(mark_count):
        x = int(rng.integers(10, max(11, width - 10)))
        y = int(rng.integers(10, max(11, height - 10)))
        radius_x = int(rng.integers(1, max(2, int(2 + wear * 12))))
        radius_y = int(rng.integers(1, max(2, int(2 + wear * 5))))
        draw.ellipse((x - radius_x, y - radius_y, x + radius_x, y + radius_y), fill=muted)
