from __future__ import annotations

from dataclasses import asdict
import csv
import json
import math
import os
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont

from .validate import ValidationReport


def _write_json_atomic(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_statistics(report: ValidationReport, path: Path) -> None:
    payload = asdict(report)
    payload["issues"] = [asdict(issue) for issue in report.issues]
    _write_json_atomic(payload, path)


def write_validation_report(report: ValidationReport, path: Path) -> None:
    write_statistics(report, path)


def _read_boxes(path: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        try:
            _, x, y, width, height = map(float, values)
        except ValueError:
            continue
        boxes.append((x, y, width, height))
    return boxes


def _caption(row: dict[str, str]) -> str:
    """Return the safe QA caption without registration text or source filenames."""
    return " | ".join(
        (row.get("output_id", "?"), row.get("plate_style", "?"), row.get("effect", "?"))
    )


def _caption_lines(row: dict[str, str], width: int = 36) -> tuple[str, ...]:
    """Wrap each safe caption field without dropping category or condition."""
    fields = (row.get("output_id", "?"), row.get("plate_style", "?"), row.get("effect", "?"))
    return tuple(
        line
        for field in fields
        for line in (wrap(field, width=width, break_on_hyphens=False) or ["?"])
    )


def _draw_sample(root: Path, row: dict[str, str], size: tuple[int, int]) -> Image.Image:
    image_path = root / row["image_path"]
    label_path = root / row["label_path"]
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        original_width, original_height = source.size
        caption_lines = _caption_lines(row)
        caption_height = len(caption_lines) * 11 + 5
        max_width, max_height = size[0], max(1, size[1] - caption_height)
        scale = min(max_width / original_width, max_height / original_height)
        resized = source.resize(
            (max(1, round(original_width * scale)), max(1, round(original_height * scale))),
            Image.Resampling.LANCZOS,
        )
    canvas = Image.new("RGB", size, "#1b1b1b")
    offset_x = (max_width - resized.width) // 2
    offset_y = (max_height - resized.height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)
    for x, y, width, height in _read_boxes(label_path):
        x1 = offset_x + (x - width / 2) * resized.width
        y1 = offset_y + (y - height / 2) * resized.height
        x2 = offset_x + (x + width / 2) * resized.width
        y2 = offset_y + (y + height / 2) * resized.height
        draw.rectangle((x1, y1, x2, y2), outline="#00ff5a", width=2)
    caption_top = size[1] - caption_height
    for index, caption_line in enumerate(caption_lines):
        draw.text((5, caption_top + index * 11), caption_line, fill="white", font=ImageFont.load_default())
    return canvas


def make_contact_sheets(
    root: Path, report: ValidationReport, samples_per_sheet: int = 16
) -> list[Path]:
    del report  # The validated manifest is the source of deterministic sample metadata.
    if samples_per_sheet <= 0:
        raise ValueError("samples_per_sheet must be positive")
    root = Path(root)
    manifest = root / "metadata" / "generation_manifest.csv"
    if not manifest.is_file():
        return []
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output_dir = root / "reports" / "contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    columns = max(1, math.ceil(math.sqrt(samples_per_sheet)))
    rows_per_sheet = math.ceil(samples_per_sheet / columns)
    cell_size = (320, 208)
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(
            row.get(field, "unknown") or "unknown"
            for field in ("split", "plate_style", "plate_layout", "effect")
        )
        groups.setdefault(key, []).append(row)
    for (split, style, layout, condition), group in sorted(groups.items()):
        selected = sorted(group, key=lambda row: row.get("output_id", ""))[:samples_per_sheet]
        if not selected:
            continue
        sheet = Image.new(
            "RGB",
            (columns * cell_size[0], rows_per_sheet * cell_size[1]),
            "#111111",
        )
        for index, row in enumerate(selected):
            sample = _draw_sample(root, row, cell_size)
            sheet.paste(sample, ((index % columns) * cell_size[0], (index // columns) * cell_size[1]))
        path = output_dir / f"{split}-{style}-{layout}-{condition}.jpg"
        sheet.save(path, format="JPEG", quality=90)
        paths.append(path)
    return paths
