from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import AbstractSet, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from .augment import apply_camera_effects, named_effect_profile
from .composite import CompositeResult, composite_plate, erase_plate
from .config import BuildConfig
from .manifests import load_manifest, sha256_file, write_manifest
from .ocr_export import export_ocr_crop
from .quotas import generation_quotas, generation_slot_plan
from .records import Box, ImageRecord
from .registration import PlateIdentity, generate_identity, normalize_registrations
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
_CHECKPOINT_INTERVAL = 100
_GENERATION_PROFILE_VERSION = "synthetic-v2"
_REJECTION_TOMBSTONE_VERSION = 1
_MIN_COMPOSITE_WIDTH_RATIO = 0.89
_MIN_COMPOSITE_HEIGHT_RATIO = 0.71


def _stable_hex(*values: object) -> str:
    return hashlib.sha256(":".join(map(str, values)).encode("utf-8")).hexdigest()


def _rng(seed: int, *values: object) -> np.random.Generator:
    return np.random.default_rng(int(_stable_hex(seed, *values)[:16], 16))


def _generation_profile_sha256(
    config: BuildConfig,
    records: Sequence[ImageRecord],
    forbidden_plate_texts: AbstractSet[str],
) -> str:
    source_membership = [
        {
            "record_id": record.record_id,
            "source_id": record.source_id,
            "source_family": record.source_family,
            "width": record.width,
            "height": record.height,
            "boxes": [
                [box.class_id, box.x_min, box.y_min, box.x_max, box.y_max]
                for box in record.boxes
            ],
            "is_real": record.is_real,
            "plate_text": record.plate_text or "",
            "tags": dict(sorted(record.tags.items())),
        }
        for record in sorted(records, key=lambda item: item.record_id)
    ]
    payload = {
        "profile_version": _GENERATION_PROFILE_VERSION,
        "config": {
            "seed": config.seed,
            "target_images": config.target_images,
            "max_images": config.max_images,
            "mh_share": list(config.mh_share),
            "split": list(config.split),
            "negative_share": list(config.negative_share),
            "training_imgsz": config.training_imgsz,
            "min_box_at_training_size": list(config.min_box_at_training_size),
            "synthetic_only": config.synthetic_only,
            "max_scene_edge": config.max_scene_edge,
            "jpeg_quality": config.jpeg_quality,
            "ocr_canvas": list(config.ocr_canvas),
        },
        "source_membership": source_membership,
        "forbidden_plate_texts": sorted(forbidden_plate_texts),
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _load_rejection_tombstones(path: Path) -> dict[str, list[dict[str, object]]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read rejection tombstones: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("version") != _REJECTION_TOMBSTONE_VERSION
        or not isinstance(payload.get("slots"), dict)
    ):
        raise ValueError("rejection tombstones have an unsupported schema")
    result: dict[str, list[dict[str, object]]] = {}
    for logical_slot, raw_entries in payload["slots"].items():
        if not isinstance(logical_slot, str) or not isinstance(raw_entries, list):
            raise ValueError("rejection tombstones contain an invalid slot entry")
        entries: list[dict[str, object]] = []
        for raw_entry in raw_entries:
            if (
                not isinstance(raw_entry, dict)
                or not isinstance(raw_entry.get("candidate_id"), str)
                or not isinstance(raw_entry.get("attempt"), int)
                or int(raw_entry["attempt"]) < 0
            ):
                raise ValueError("rejection tombstones contain an invalid candidate")
            entries.append(
                {
                    "candidate_id": str(raw_entry["candidate_id"]),
                    "attempt": int(raw_entry["attempt"]),
                }
            )
        result[logical_slot] = sorted(entries, key=lambda entry: int(entry["attempt"]))
    return result


def _write_rejection_tombstones(
    tombstones: Mapping[str, Sequence[Mapping[str, object]]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _REJECTION_TOMBSTONE_VERSION,
        "slots": {
            logical_slot: [
                {
                    "candidate_id": str(entry["candidate_id"]),
                    "attempt": int(entry["attempt"]),
                }
                for entry in sorted(entries, key=lambda item: int(item["attempt"]))
            ]
            for logical_slot, entries in sorted(tombstones.items())
        },
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def _can_reuse(
    row: Mapping[str, str] | None,
    output: Path,
    *,
    requires_ocr: bool = False,
) -> bool:
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
        return valid_pair and not requires_ocr
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
        "detector_eligible": str(not negative).lower(),
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
    generation_profile_sha256: str
    source_image_sha256: str
    logical_slot_id: str
    candidate_attempt: int


@dataclass(frozen=True)
class GenerationResult:
    row: dict[str, object]
    reused: bool


@dataclass(frozen=True)
class GenerationJob:
    spec: GenerationSpec
    identity: PlateIdentity | None
    rng_state: dict[str, object] | None


def _synthetic_specs(
    config: BuildConfig,
    records: Sequence[ImageRecord],
    rejected_ids: frozenset[str],
    *,
    forbidden_plate_texts: AbstractSet[str] = frozenset(),
    tombstones: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
) -> list[GenerationSpec]:
    records = [
        record
        for record in records
        if _source_can_render_detector_eligible_plate(record, config)
    ]
    if not records:
        raise InsufficientSourceData(
            "no source scene contains a plate anchor large enough for detector training"
        )
    generation_profile = _generation_profile_sha256(
        config, records, forbidden_plate_texts
    )
    source_hashes = {
        record.record_id: (
            sha256_file(record.image_path) if record.image_path.is_file() else ""
        )
        for record in records
    }
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

    quotas = generation_quotas(config)
    specs: list[GenerationSpec] = []
    global_index = 0
    for split in ("train", "val", "test"):
        quota = quotas[split]
        family_names = sorted(grouped[split])
        if quota.total and not family_names:
            raise InsufficientSourceData(
                f"split {split} has no source scene from which to create variants"
            )
        next_variant = {family: 0 for family in family_names}
        slot_plan = generation_slot_plan(config, split)
        for slot in range(quota.total):
            family = family_names[slot % len(family_names)]
            base_variant = next_variant[family]
            next_variant[family] = base_variant + 1
            logical_slot_id = f"{split}:{slot:08d}"
            entries = (tombstones or {}).get(logical_slot_id, ())
            candidate_attempt = max(
                (int(entry["attempt"]) for entry in entries), default=-1
            ) + 1
            variant_index = base_variant + candidate_attempt * max(1, quota.total)
            output_id = f"syn-{_stable_hex(config.seed, split, family, variant_index)[:20]}"
            while output_id in rejected_ids:
                candidate_attempt += 1
                variant_index = base_variant + candidate_attempt * max(1, quota.total)
                output_id = f"syn-{_stable_hex(config.seed, split, family, variant_index)[:20]}"
            source = grouped[split][family][variant_index % len(grouped[split][family])]
            negative = slot in slot_plan.negative_slots
            specs.append(
                GenerationSpec(
                    output_id=output_id,
                    split=split,
                    source=source,
                    global_index=global_index,
                    variant_index=variant_index,
                    negative=negative,
                    force_mh=slot in slot_plan.mh_positive_slots,
                    layout=(
                        "double"
                        if slot in slot_plan.double_row_positive_slots
                        else "single"
                    ),
                    category=_CATEGORIES[global_index % len(_CATEGORIES)],
                    condition=slot_plan.conditions[slot],
                    generation_profile_sha256=generation_profile,
                    source_image_sha256=source_hashes[source.record_id],
                    logical_slot_id=logical_slot_id,
                    candidate_attempt=candidate_attempt,
                )
            )
            global_index += 1
    return specs


def _persist_rejected_candidates(
    config: BuildConfig,
    records: Sequence[ImageRecord],
    rejected_ids: AbstractSet[str],
    forbidden_plate_texts: AbstractSet[str],
    tombstones: dict[str, list[dict[str, object]]],
    path: Path,
) -> None:
    if not rejected_ids:
        return
    current_specs = _synthetic_specs(
        config,
        records,
        frozenset(),
        forbidden_plate_texts=forbidden_plate_texts,
        tombstones=tombstones,
    )
    current_by_id = {spec.output_id: spec for spec in current_specs}
    already_rejected = {
        str(entry["candidate_id"])
        for entries in tombstones.values()
        for entry in entries
    }
    unknown = sorted(set(rejected_ids) - current_by_id.keys() - already_rejected)
    if unknown:
        raise ValueError(f"unknown rejected output IDs: {','.join(unknown)}")
    changed = False
    for output_id in sorted(rejected_ids):
        if output_id in already_rejected:
            continue
        spec = current_by_id[output_id]
        tombstones.setdefault(spec.logical_slot_id, []).append(
            {"candidate_id": output_id, "attempt": spec.candidate_attempt}
        )
        changed = True
    if changed:
        _write_rejection_tombstones(tombstones, path)


def _load_scaled_anchor(
    source: ImageRecord, config: BuildConfig
) -> tuple[np.ndarray, Box, tuple[Box, ...]]:
    with Image.open(source.image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, config.max_scene_edge / max(width, height))
        resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        if resized_size != image.size:
            image = image.resize(resized_size, Image.Resampling.LANCZOS)
        scene = np.asarray(image, dtype=np.uint8).copy()
    anchors: list[Box] = []
    candidates: list[Box] = []
    for box in source.boxes:
        scaled = Box(
            box.class_id,
            box.x_min * scale,
            box.y_min * scale,
            box.x_max * scale,
            box.y_max * scale,
        ).clip(scene.shape[1], scene.shape[0])
        anchors.append(scaled)
        if _composite_box_meets_training_minimum(
            scaled, scene.shape[1], scene.shape[0], config
        ):
            candidates.append(scaled)
    if not candidates:
        raise InsufficientSourceData(
            f"source scene {source.record_id} has no detector-eligible plate anchor"
        )
    primary = max(
        candidates,
        key=lambda box: (box.x_max - box.x_min) * (box.y_max - box.y_min),
    )
    return scene, primary, tuple(anchors)


def _source_can_render_detector_eligible_plate(
    source: ImageRecord, config: BuildConfig
) -> bool:
    if source.width <= 0 or source.height <= 0:
        return False
    scale = min(1.0, config.max_scene_edge / max(source.width, source.height))
    width = max(1, round(source.width * scale))
    height = max(1, round(source.height * scale))
    for box in source.boxes:
        scaled = Box(
            box.class_id,
            box.x_min * scale,
            box.y_min * scale,
            box.x_max * scale,
            box.y_max * scale,
        ).clip(width, height)
        if _composite_box_meets_training_minimum(scaled, width, height, config):
            return True
    return False


def _composite_box_meets_training_minimum(
    box: Box, image_width: int, image_height: int, config: BuildConfig
) -> bool:
    scale = config.training_imgsz / max(image_width, image_height)
    minimum_width, minimum_height = config.min_box_at_training_size
    return (
        (box.x_max - box.x_min) * _MIN_COMPOSITE_WIDTH_RATIO * scale
        >= minimum_width
        and (box.y_max - box.y_min) * _MIN_COMPOSITE_HEIGHT_RATIO * scale
        >= minimum_height
    )


def _matches_synthetic_spec(row: Mapping[str, str] | None, spec: GenerationSpec) -> bool:
    if not row:
        return False
    if (
        row.get("origin") != "synthetic"
        or row.get("split") != spec.split
        or row.get("source_id") != spec.source.source_id
        or row.get("source_family") != spec.source.source_family
        or row.get("image_path")
        != f"detection/images/{spec.split}/{spec.output_id}.jpg"
        or row.get("label_path")
        != f"detection/labels/{spec.split}/{spec.output_id}.txt"
        or row.get("generation_profile_sha256")
        != spec.generation_profile_sha256
        or row.get("source_image_sha256") != spec.source_image_sha256
        or row.get("logical_slot_id") != spec.logical_slot_id
        or row.get("candidate_attempt") != str(spec.candidate_attempt)
        or row.get("negative") != str(spec.negative).lower()
        or row.get("effect") != spec.condition
    ):
        return False
    if spec.negative:
        return (
            row.get("plate_style") == "removed"
            and row.get("plate_layout") == "none"
            and not row.get("state")
            and not row.get("plate_text")
            and not row.get("ocr_path")
            and row.get("detector_eligible") == "false"
            and row.get("ocr_eligible") == "false"
            and not row.get("ocr_sha256")
        )
    return (
        bool(row.get("plate_text"))
        and row.get("ocr_path") == f"ocr/images/{spec.split}/{spec.output_id}.jpg"
        and row.get("plate_style") == spec.category
        and row.get("plate_layout") == spec.layout
        and row.get("detector_eligible") == "true"
        and (row.get("state") == "MH") == spec.force_mh
    )


def _meets_training_minimum(
    box: Box, image_width: int, image_height: int, config: BuildConfig
) -> bool:
    scale = config.training_imgsz / max(image_width, image_height)
    minimum_width, minimum_height = config.min_box_at_training_size
    return (
        (box.x_max - box.x_min) * scale >= minimum_width
        and (box.y_max - box.y_min) * scale >= minimum_height
    )


def _prepare_synthetic_jobs(
    config: BuildConfig,
    specs: Sequence[GenerationSpec],
    records: Sequence[ImageRecord],
    output: Path,
    previous: Mapping[str, Mapping[str, str]],
    forbidden_plate_texts: AbstractSet[str],
) -> list[GenerationJob]:
    forbidden = normalize_registrations(
        [
            *(record.plate_text for record in records if record.plate_text),
            *forbidden_plate_texts,
        ]
    )
    for spec in specs:
        old_row = previous.get(spec.output_id)
        if old_row and not spec.negative:
            forbidden.update(
                normalize_registrations(
                    part
                    for part in str(old_row.get("plate_text", "")).split("|")
                    if part
                )
            )
    jobs = []
    for spec in specs:
        if spec.negative or spec.output_id in previous:
            jobs.append(GenerationJob(spec=spec, identity=None, rng_state=None))
            continue
        generator = _rng(
            config.seed, spec.split, spec.source.source_family, spec.variant_index
        )
        identity = generate_identity(
            generator,
            mh_probability=1.0 if spec.force_mh else 0.0,
            forbidden=forbidden,
            forbidden_is_normalized=True,
        )
        jobs.append(
            GenerationJob(
                spec=spec,
                identity=identity,
                rng_state=generator.bit_generator.state,
            )
        )
        forbidden.add(identity.compact_text)
    return jobs


def _compatible_previous_rows(
    specs: Sequence[GenerationSpec],
    previous: Mapping[str, Mapping[str, str]],
    output: Path,
) -> dict[str, dict[str, str]]:
    compatible = {}
    for spec in specs:
        row = previous.get(spec.output_id)
        if _matches_synthetic_spec(row, spec) and _can_reuse(
            row, output, requires_ocr=not spec.negative
        ):
            compatible[spec.output_id] = dict(row)
    return compatible


def _render_synthetic_spec(
    config: BuildConfig,
    spec: GenerationSpec,
    output: Path,
    previous: Mapping[str, Mapping[str, str]],
    forbidden: set[str],
    fonts: Sequence[Path],
    prepared_identity: PlateIdentity | None = None,
    prepared_rng_state: dict[str, object] | None = None,
) -> GenerationResult:
    old_row = previous.get(spec.output_id)
    if _matches_synthetic_spec(old_row, spec) and _can_reuse(
        old_row, output, requires_ocr=not spec.negative
    ):
        return GenerationResult(dict(old_row), True)

    scene, anchor, source_anchors = _load_scaled_anchor(spec.source, config)
    generator = _rng(config.seed, spec.split, spec.source.source_family, spec.variant_index)
    for source_anchor in source_anchors:
        scene = erase_plate(scene, source_anchor, generator)
    final_boxes: list[Box] = []
    plate_text = ""
    state = ""
    ocr_eligible = False
    detector_eligible = False
    ocr_path: Path | None = None
    ocr_sha256 = ""
    if spec.negative:
        scene = apply_camera_effects(
            CompositeResult(
                image=scene,
                box=anchor,
                tags={},
                keep_detection=False,
                ocr_eligible=False,
            ),
            named_effect_profile(spec.condition, generator),
            generator,
        ).image
    else:
        if prepared_identity is None:
            identity = generate_identity(
                generator,
                mh_probability=1.0 if spec.force_mh else 0.0,
                forbidden=forbidden,
            )
            forbidden.add(identity.compact_text)
        else:
            if prepared_rng_state is None:
                raise RuntimeError("prepared identity is missing its RNG state")
            identity = prepared_identity
            generator.bit_generator.state = prepared_rng_state
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

    if final_boxes:
        detector_eligible = all(
            _meets_training_minimum(
                box, scene.shape[1], scene.shape[0], config
            )
            for box in final_boxes
        )
        if not detector_eligible:
            raise InsufficientSourceData(
                f"synthetic output {spec.output_id} has no detector-eligible plate box"
            )

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
            "effect": spec.condition,
            "generation_profile_sha256": spec.generation_profile_sha256,
            "source_image_sha256": spec.source_image_sha256,
            "logical_slot_id": spec.logical_slot_id,
            "candidate_attempt": spec.candidate_attempt,
            "detector_eligible": str(detector_eligible).lower(),
            "ocr_eligible": str(ocr_eligible and not spec.negative).lower(),
            "ocr_path": _relative(ocr_path, output) if ocr_path else "",
            "ocr_sha256": ocr_sha256,
        }
    )
    return GenerationResult(_finish_row(row, image_path, label_path), False)


_WORKER_CONTEXT: tuple[
    BuildConfig,
    Path,
    Mapping[str, Mapping[str, str]],
    Sequence[Path],
] | None = None


def _initialize_synthetic_worker(
    config: BuildConfig,
    output: Path,
    previous: Mapping[str, Mapping[str, str]],
    fonts: Sequence[Path],
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = (config, output, previous, fonts)


def _render_synthetic_job_worker(job: GenerationJob) -> GenerationResult:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("synthetic worker context was not initialized")
    config, output, previous, fonts = _WORKER_CONTEXT
    return _render_synthetic_spec(
        config,
        job.spec,
        output,
        previous,
        set(),
        fonts,
        job.identity,
        job.rng_state,
    )


def _resolved_worker_count(requested: int) -> int:
    if requested:
        return requested
    return max(1, min((os.cpu_count() or 1) - 1, 8))


def _render_failure(job: GenerationJob, error: Exception) -> RuntimeError:
    return RuntimeError(
        f"synthetic output {job.spec.output_id} failed with {type(error).__name__}"
    )


def _build_synthetic_only(
    config: BuildConfig,
    records: Sequence[ImageRecord],
    output: Path,
    rejected_ids: frozenset[str],
    progress: Callable[[int, int], object] | None = None,
    forbidden_plate_texts: AbstractSet[str] = frozenset(),
) -> BuildManifest:
    manifest_path = output / "metadata" / "generation_manifest.csv"
    tombstone_path = output / "metadata" / "rejected_generation_slots.json"
    previous = load_manifest(manifest_path)
    tombstones = _load_rejection_tombstones(tombstone_path)
    _persist_rejected_candidates(
        config,
        records,
        rejected_ids,
        forbidden_plate_texts,
        tombstones,
        tombstone_path,
    )
    specs = _synthetic_specs(
        config,
        records,
        frozenset(),
        forbidden_plate_texts=forbidden_plate_texts,
        tombstones=tombstones,
    )
    durable_previous = _compatible_previous_rows(specs, previous, output)
    jobs = _prepare_synthetic_jobs(
        config,
        specs,
        records,
        output,
        durable_previous,
        forbidden_plate_texts,
    )
    fonts = discover_font_paths()
    rows: list[dict[str, object]] = []
    reused = 0

    def record_result(result: GenerationResult) -> None:
        nonlocal reused
        if result.reused:
            reused += 1
        rows.append(result.row)
        if len(rows) % _CHECKPOINT_INTERVAL == 0:
            checkpoint_rows: dict[str, Mapping[str, object]] = dict(durable_previous)
            checkpoint_rows.update(
                (str(row["output_id"]), row) for row in rows
            )
            write_manifest(checkpoint_rows.values(), manifest_path)
        if progress is not None:
            progress(len(rows), len(jobs))

    workers = _resolved_worker_count(config.workers)
    if workers == 1:
        for job in jobs:
            try:
                result = _render_synthetic_spec(
                    config,
                    job.spec,
                    output,
                    durable_previous,
                    set(),
                    fonts,
                    job.identity,
                    job.rng_state,
                )
            except Exception as error:
                raise _render_failure(job, error) from error
            record_result(result)
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_synthetic_worker,
            initargs=(config, output, durable_previous, fonts),
        )
        pending: dict[Future[GenerationResult], GenerationJob] = {}
        job_iterator = iter(jobs)
        pending_limit = workers * 2

        def fill_pending() -> None:
            while len(pending) < pending_limit:
                try:
                    job = next(job_iterator)
                except StopIteration:
                    return
                pending[executor.submit(_render_synthetic_job_worker, job)] = job

        try:
            fill_pending()
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                ordered = sorted(
                    completed, key=lambda future: pending[future].spec.output_id
                )
                for future in ordered:
                    job = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        raise _render_failure(job, error) from error
                    record_result(result)
                fill_pending()
        except Exception:
            for future in pending:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

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
    forbidden_plate_texts: AbstractSet[str] = frozenset(),
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
        return _build_synthetic_only(
            config,
            records,
            output,
            frozenset(rejected_ids),
            progress if callable(progress) else None,
            forbidden_plate_texts,
        )

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
