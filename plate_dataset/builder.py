from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from .augment import apply_camera_effects, named_effect_profile
from .composite import CompositeResult, composite_plate, erase_plate
from .config import BuildConfig
from .manifests import load_manifest, sha256_file, write_manifest
from .ocr_export import export_ocr_crop
from .records import Box, ImageRecord
from .registration import generate_identity
from .render import PlateStyle, discover_font_paths, render_plate
from .split import SplitName, assign_splits, split_target_counts


class InsufficientSourceData(ValueError):
    pass


@dataclass(frozen=True)
class BuildManifest:
    seed: int
    generated_count: int
    reused_count: int
    rejected_count: int
    manifest_path: Path


_CATEGORIES = (
    "private",
    "commercial",
    "electric_private",
    "electric_commercial",
    "temporary",
)
_EFFECTS = ("day", "night", "rain", "fog", "glare", "shadow", "motion", "compression")


def _stable_hex(*values: object) -> str:
    return hashlib.sha256(":".join(map(str, values)).encode("utf-8")).hexdigest()


def _rng(seed: int, *values: object) -> np.random.Generator:
    return np.random.default_rng(int(_stable_hex(seed, *values)[:16], 16))


def _format_labels(boxes: Sequence[Box], width: int, height: int) -> str:
    lines = []
    for box in boxes:
        class_id, x, y, w, h = box.to_yolo(width, height)
        lines.append(f"{class_id} {x:.8f} {y:.8f} {w:.8f} {h:.8f}")
    return "\n".join(lines) + ("\n" if lines else "")


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _save_jpeg_atomic(image: np.ndarray | Image.Image, path: Path, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.jpg")
    converted = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    converted.convert("RGB").save(temporary, format="JPEG", quality=quality, subsampling=0)
    os.replace(temporary, path)


def _paths(output: Path, split: SplitName, output_id: str) -> tuple[Path, Path]:
    return (
        output / "detection" / "images" / split / f"{output_id}.jpg",
        output / "detection" / "labels" / split / f"{output_id}.txt",
    )


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _can_reuse(row: Mapping[str, str] | None, output: Path) -> bool:
    if not row:
        return False
    image = output / row.get("image_path", "")
    label = output / row.get("label_path", "")
    valid_pair = (
        image.is_file()
        and label.is_file()
        and sha256_file(image) == row.get("image_sha256")
        and sha256_file(label) == row.get("label_sha256")
    )
    ocr_path = row.get("ocr_path", "")
    if not ocr_path:
        return valid_pair
    crop = output / ocr_path
    return valid_pair and crop.is_file() and sha256_file(crop) == row.get("ocr_sha256")


def _base_row(
    *,
    output_id: str,
    split: SplitName,
    origin: str,
    source: ImageRecord,
    image_path: Path,
    label_path: Path,
    output: Path,
    negative: bool,
) -> dict[str, object]:
    return {
        "output_id": output_id,
        "split": split,
        "origin": origin,
        "source_id": source.source_id,
        "source_family": source.source_family,
        "image_path": _relative(image_path, output),
        "label_path": _relative(label_path, output),
        "negative": str(negative).lower(),
        "state": source.tags.get("state", ""),
        "plate_text": source.plate_text or "",
        "vehicle_type": source.tags.get("vehicle_type", "unknown"),
        "viewpoint": source.tags.get("viewpoint", "unknown"),
        "plate_style": "original" if not negative else "removed",
        "plate_layout": source.tags.get("plate_layout", "unknown"),
        "effect": "original",
        "ocr_eligible": str(bool(source.plate_text and not negative)).lower(),
    }


def _finish_row(row: dict[str, object], image_path: Path, label_path: Path) -> dict[str, object]:
    row["image_sha256"] = sha256_file(image_path)
    row["label_sha256"] = sha256_file(label_path)
    return row


@dataclass(frozen=True)
class GenerationSpec:
    output_id: str
    split: SplitName
    source: ImageRecord
    global_index: int
    variant_index: int
    negative: bool
    force_mh: bool
    layout: str
    category: str
    condition: str


@dataclass(frozen=True)
class GenerationResult:
    row: dict[str, object]
    reused: bool


def _synthetic_specs(
    config: BuildConfig,
    records: Sequence[ImageRecord],
    rejected_ids: frozenset[str],
) -> list[GenerationSpec]:
    assignments = assign_splits(records, config)
    grouped: dict[SplitName, dict[str, list[ImageRecord]]] = {
        "train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)
    }
    for record in records:
        assignment = assignments[record.record_id]
        grouped[assignment.split][assignment.source_family].append(record)
    for families in grouped.values():
        for sources in families.values():
            sources.sort(key=lambda source: source.record_id)

    targets = split_target_counts(config.target_images, config.split)
    negative_count = min(
        config.target_images,
        int(round(config.target_images * sum(config.negative_share) / 2.0)),
    )
    positive_count = config.target_images - negative_count
    mh_count = int(round(positive_count * sum(config.mh_share) / 2.0))
    specs: list[GenerationSpec] = []
    global_index = 0
    for split in ("train", "val", "test"):
        family_names = sorted(grouped[split])
        if targets[split] and not family_names:
            raise InsufficientSourceData(
                f"split {split} has no source scene from which to create variants"
            )
        next_variant = {family: 0 for family in family_names}
        for slot in range(targets[split]):
            family = family_names[slot % len(family_names)]
            variant_index = next_variant[family]
            output_id = f"syn-{_stable_hex(config.seed, split, family, variant_index)[:20]}"
            while output_id in rejected_ids:
                variant_index += 1
                output_id = f"syn-{_stable_hex(config.seed, split, family, variant_index)[:20]}"
            next_variant[family] = variant_index + 1
            source = grouped[split][family][variant_index % len(grouped[split][family])]
            negative = global_index < negative_count
            positive_index = global_index - negative_count
            specs.append(
                GenerationSpec(
                    output_id=output_id,
                    split=split,
                    source=source,
                    global_index=global_index,
                    variant_index=variant_index,
                    negative=negative,
                    force_mh=not negative and positive_index < mh_count,
                    layout="double" if global_index % 5 == 4 else "single",
                    category=_CATEGORIES[global_index % len(_CATEGORIES)],
                    condition=_EFFECTS[global_index % len(_EFFECTS)],
                )
            )
            global_index += 1
    return specs


def _load_scaled_anchor(source: ImageRecord, max_scene_edge: int) -> tuple[np.ndarray, Box]:
    with Image.open(source.image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, max_scene_edge / max(width, height))
        resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        if resized_size != image.size:
            image = image.resize(resized_size, Image.Resampling.LANCZOS)
        scene = np.asarray(image, dtype=np.uint8).copy()
    candidates = []
    for box in source.boxes:
        scaled = Box(
            box.class_id,
            box.x_min * scale,
            box.y_min * scale,
            box.x_max * scale,
            box.y_max * scale,
        ).clip(scene.shape[1], scene.shape[0])
        if scaled.x_max - scaled.x_min >= 8 and scaled.y_max - scaled.y_min >= 4:
            candidates.append(scaled)
    if not candidates:
        raise InsufficientSourceData(f"source scene {source.record_id} has no valid plate anchor")
    return scene, max(candidates, key=lambda box: (box.x_max - box.x_min) * (box.y_max - box.y_min))


def _render_synthetic_spec(
    config: BuildConfig,
    spec: GenerationSpec,
    output: Path,
    previous: Mapping[str, Mapping[str, str]],
    forbidden: set[str],
    fonts: Sequence[Path],
) -> GenerationResult:
    old_row = previous.get(spec.output_id)
    if _can_reuse(old_row, output):
        return GenerationResult(dict(old_row), True)

    scene, anchor = _load_scaled_anchor(spec.source, config.max_scene_edge)
    generator = _rng(config.seed, spec.split, spec.source.source_family, spec.variant_index)
    final_boxes: list[Box] = []
    plate_text = ""
    state = ""
    ocr_eligible = False
    ocr_path: Path | None = None
    ocr_sha256 = ""
    if spec.negative:
        scene = erase_plate(scene, anchor, generator)
    else:
        identity = generate_identity(
            generator,
            mh_probability=1.0 if spec.force_mh else 0.0,
            forbidden=forbidden,
        )
        forbidden.add(identity.compact_text)
        rendered = render_plate(
            identity,
            PlateStyle(
                category=spec.category,
                layout=spec.layout,
                wear=float(generator.uniform(0, 0.48)),
            ),
            fonts,
            generator,
        )
        composite = composite_plate(scene, anchor, rendered, generator)
        effected = apply_camera_effects(
            composite,
            named_effect_profile(spec.condition, generator),
            generator,
        )
        scene = effected.image
        final_boxes.append(effected.box)
        plate_text = identity.compact_text
        state = identity.state
        ocr_eligible = effected.ocr_eligible

    image_path, label_path = _paths(output, spec.split, spec.output_id)
    _save_jpeg_atomic(scene, image_path, config.jpeg_quality)
    _write_text_atomic(label_path, _format_labels(final_boxes, scene.shape[1], scene.shape[0]))
    if final_boxes:
        ocr_path = output / "ocr" / "images" / spec.split / f"{spec.output_id}.jpg"
        ocr_sha256 = export_ocr_crop(
            scene, final_boxes[0], ocr_path, config.ocr_canvas, config.jpeg_quality
        )
    row = _base_row(
        output_id=spec.output_id,
        split=spec.split,
        origin="synthetic",
        source=spec.source,
        image_path=image_path,
        label_path=label_path,
        output=output,
        negative=spec.negative,
    )
    row.update(
        {
            "state": state,
            "plate_text": plate_text,
            "plate_style": spec.category if not spec.negative else "removed",
            "plate_layout": spec.layout if not spec.negative else "none",
            "effect": spec.condition if not spec.negative else "plate_removed",
            "ocr_eligible": str(ocr_eligible and not spec.negative).lower(),
            "ocr_path": _relative(ocr_path, output) if ocr_path else "",
            "ocr_sha256": ocr_sha256,
        }
    )
    return GenerationResult(_finish_row(row, image_path, label_path), False)


def _build_synthetic_only(
    config: BuildConfig,
    records: Sequence[ImageRecord],
    output: Path,
    rejected_ids: frozenset[str],
) -> BuildManifest:
    manifest_path = output / "metadata" / "generation_manifest.csv"
    previous = load_manifest(manifest_path)
    specs = _synthetic_specs(config, records, rejected_ids)
    forbidden = {record.plate_text for record in records if record.plate_text}
    fonts = discover_font_paths()
    rows: list[dict[str, object]] = []
    reused = 0
    for spec in specs:
        result = _render_synthetic_spec(config, spec, output, previous, forbidden, fonts)
        if result.reused:
            reused += 1
            text = str(result.row.get("plate_text", ""))
            if text:
                forbidden.update(part for part in text.split("|") if part)
        rows.append(result.row)
    if len(rows) != config.target_images:
        raise RuntimeError(f"builder emitted {len(rows)} records, expected {config.target_images}")
    write_manifest(rows, manifest_path)
    return BuildManifest(
        seed=config.seed,
        generated_count=len(rows),
        reused_count=reused,
        rejected_count=len(rejected_ids),
        manifest_path=manifest_path,
    )

def build_dataset(
    config: BuildConfig,
    records: Sequence[ImageRecord],
    output: Path | None = None,
    rejected_ids: frozenset[str] = frozenset(),
    progress: object = None,
) -> BuildManifest:
    output = Path(output or config.workspace)
    if not records:
        raise InsufficientSourceData("at least one annotated source scene is required")
    if len(records) > config.target_images:
        ordered = sorted(
            records,
            key=lambda record: _stable_hex(config.seed, record.source_family, record.record_id),
        )
        records = ordered[: config.target_images]

    if config.synthetic_only:
        return _build_synthetic_only(config, records, output, frozenset(rejected_ids))

    assignments = assign_splits(records, config)
    targets = split_target_counts(config.target_images, config.split)
    existing_by_split: dict[SplitName, list[ImageRecord]] = defaultdict(list)
    for record in records:
        existing_by_split[assignments[record.record_id].split].append(record)
    for split in existing_by_split:
        existing_by_split[split].sort(key=lambda record: record.record_id)

    manifest_path = output / "metadata" / "generation_manifest.csv"
    previous = load_manifest(manifest_path)
    rows: list[dict[str, object]] = []
    reused = 0

    # Preserve each source scene once; synthetic variants are added only to deficits.
    for split in ("train", "val", "test"):
        for source in existing_by_split[split]:
            output_id = f"real-{_stable_hex(config.seed, source.record_id)[:20]}"
            image_path, label_path = _paths(output, split, output_id)
            if _can_reuse(previous.get(output_id), output):
                rows.append(dict(previous[output_id]))
                reused += 1
                continue
            with Image.open(source.image_path) as image:
                image = image.convert("RGB")
                width, height = image.size
                _save_jpeg_atomic(image, image_path)
            _write_text_atomic(label_path, _format_labels(source.boxes, width, height))
            row = _base_row(
                output_id=output_id,
                split=split,
                origin="real" if source.is_real else "synthetic",
                source=source,
                image_path=image_path,
                label_path=label_path,
                output=output,
                negative=False,
            )
            rows.append(_finish_row(row, image_path, label_path))

    deficits = {
        split: targets[split] - len(existing_by_split[split])
        for split in ("train", "val", "test")
    }
    if any(value < 0 for value in deficits.values()):
        raise InsufficientSourceData("source assignment exceeds a requested split target")
    synthetic_total = sum(deficits.values())
    negative_rate = sum(config.negative_share) / 2.0
    negative_count = min(synthetic_total, int(round(config.target_images * negative_rate)))
    positive_count = synthetic_total - negative_count
    mh_rate = sum(config.mh_share) / 2.0
    mh_count = int(round(positive_count * mh_rate))
    forbidden = {record.plate_text for record in records if record.plate_text}
    fonts = discover_font_paths() if positive_count else []

    specs: list[tuple[SplitName, ImageRecord, int]] = []
    for split in ("train", "val", "test"):
        pool = existing_by_split[split]
        if deficits[split] and not pool:
            # Borrowing a base would leak a source family across splits.
            raise InsufficientSourceData(
                f"split {split} has no source scene from which to create variants"
            )
        for index in range(deficits[split]):
            specs.append((split, pool[index % len(pool)], index))

    for global_index, (split, source, variant_index) in enumerate(specs):
        negative = global_index < negative_count
        positive_index = global_index - negative_count
        output_id = f"syn-{_stable_hex(config.seed, split, source.source_family, variant_index)[:20]}"
        image_path, label_path = _paths(output, split, output_id)
        if _can_reuse(previous.get(output_id), output):
            rows.append(dict(previous[output_id]))
            reused += 1
            if not negative:
                text = previous[output_id].get("plate_text", "")
                if text:
                    forbidden.add(text)
            continue

        generator = _rng(config.seed, split, source.source_family, variant_index)
        with Image.open(source.image_path) as image:
            scene = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        final_boxes: list[Box] = []
        plate_texts: list[str] = []
        states: list[str] = []
        category = _CATEGORIES[global_index % len(_CATEGORIES)]
        layout = "double" if global_index % 5 == 4 else "single"
        effect = _EFFECTS[global_index % len(_EFFECTS)]
        ocr_eligible = False

        if negative:
            for anchor in source.boxes:
                scene = erase_plate(scene, anchor, generator)
        else:
            force_mh = positive_index < mh_count
            for anchor in source.boxes:
                identity = generate_identity(
                    generator, mh_probability=1.0 if force_mh else 0.0, forbidden=forbidden
                )
                forbidden.add(identity.compact_text)
                rendered = render_plate(
                    identity,
                    PlateStyle(category=category, layout=layout, wear=float(generator.uniform(0, 0.48))),
                    fonts,
                    generator,
                )
                composite = composite_plate(scene, anchor, rendered, generator)
                scene = composite.image
                final_boxes.append(composite.box)
                plate_texts.append(identity.compact_text)
                states.append(identity.state)
                ocr_eligible = ocr_eligible or composite.ocr_eligible
            effected = apply_camera_effects(
                CompositeResult(
                    image=scene,
                    box=final_boxes[0],
                    tags={},
                    keep_detection=True,
                    ocr_eligible=ocr_eligible,
                ),
                named_effect_profile(effect, generator),
                generator,
            )
            scene = effected.image
            ocr_eligible = effected.ocr_eligible

        _save_jpeg_atomic(scene, image_path)
        _write_text_atomic(
            label_path,
            _format_labels(final_boxes, scene.shape[1], scene.shape[0]),
        )
        row = _base_row(
            output_id=output_id,
            split=split,
            origin="synthetic",
            source=source,
            image_path=image_path,
            label_path=label_path,
            output=output,
            negative=negative,
        )
        row.update(
            {
                "state": states[0] if states else "",
                "plate_text": "|".join(plate_texts),
                "plate_style": category if not negative else "removed",
                "plate_layout": layout if not negative else "none",
                "effect": effect if not negative else "plate_removed",
                "ocr_eligible": str(ocr_eligible and not negative).lower(),
            }
        )
        rows.append(_finish_row(row, image_path, label_path))

    if len(rows) != config.target_images:
        raise RuntimeError(
            f"builder emitted {len(rows)} records, expected {config.target_images}"
        )
    write_manifest(rows, manifest_path)
    return BuildManifest(
        seed=config.seed,
        generated_count=len(rows),
        reused_count=reused,
        rejected_count=0,
        manifest_path=manifest_path,
    )
