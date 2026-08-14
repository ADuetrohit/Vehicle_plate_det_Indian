from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal, Mapping

from PIL import Image

from .config import BuildConfig
from .manifests import sha256_file
from .quotas import generation_quotas
from .split import split_target_counts


_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ValidationIssue:
    severity: Literal["error", "warning"]
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    image_count: int
    label_count: int
    error_count: int
    warning_count: int
    split_counts: Mapping[str, int]
    distributions: Mapping[str, Mapping[str, int]]
    issues: tuple[ValidationIssue, ...]


def _issue(
    issues: list[ValidationIssue],
    code: str,
    path: Path | str,
    message: str,
    severity: Literal["error", "warning"] = "error",
) -> None:
    issues.append(ValidationIssue(severity, code, str(path), message))


def _read_manifest(path: Path, issues: list[ValidationIssue]) -> list[dict[str, str]]:
    if not path.is_file():
        _issue(issues, "missing_manifest", path, "generation manifest is missing")
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        _issue(issues, "invalid_manifest", path, f"cannot read manifest: {error}")
        return []
    required = {
        "output_id",
        "split",
        "origin",
        "source_family",
        "image_path",
        "label_path",
        "image_sha256",
        "label_sha256",
        "negative",
    }
    if rows and not required.issubset(rows[0]):
        _issue(issues, "invalid_manifest", path, "manifest is missing required columns")
        return []
    return rows


def _validate_label(
    label: Path,
    image_size: tuple[int, int],
    negative: bool,
    config: BuildConfig,
    issues: list[ValidationIssue],
) -> bool:
    try:
        text = label.read_text(encoding="utf-8")
    except OSError as error:
        _issue(issues, "unreadable_label", label, str(error))
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        if not negative:
            _issue(issues, "unexpected_empty_label", label, "positive image has no plate box")
        return False
    if negative:
        _issue(issues, "negative_has_box", label, "hard-negative label must be empty")
    width, height = image_size
    scale = config.training_imgsz / max(width, height)
    for line_number, line in enumerate(lines, start=1):
        values = line.split()
        if len(values) != 5:
            _issue(issues, "invalid_yolo_row", label, f"line {line_number} must contain five values")
            continue
        try:
            class_value, x, y, box_width, box_height = map(float, values)
        except ValueError:
            _issue(issues, "invalid_yolo_row", label, f"line {line_number} is not numeric")
            continue
        numbers = (class_value, x, y, box_width, box_height)
        if not all(math.isfinite(value) for value in numbers):
            _issue(issues, "non_finite_coordinate", label, f"line {line_number} contains NaN or infinity")
            continue
        if class_value != 0:
            _issue(issues, "invalid_class", label, f"line {line_number} uses class {class_value:g}, expected 0")
        if not all(0.0 <= value <= 1.0 for value in (x, y, box_width, box_height)):
            _issue(issues, "coordinate_out_of_range", label, f"line {line_number} contains a coordinate outside [0, 1]")
            continue
        if box_width <= 0 or box_height <= 0:
            _issue(issues, "non_positive_box", label, f"line {line_number} has non-positive size")
            continue
        if x - box_width / 2 < 0 or x + box_width / 2 > 1 or y - box_height / 2 < 0 or y + box_height / 2 > 1:
            _issue(issues, "box_out_of_bounds", label, f"line {line_number} extends beyond its image")
        training_width = box_width * width * scale
        training_height = box_height * height * scale
        minimum_width, minimum_height = config.min_box_at_training_size
        if training_width < minimum_width or training_height < minimum_height:
            _issue(
                issues,
                "box_below_training_minimum",
                label,
                f"line {line_number} becomes {training_width:.2f}x{training_height:.2f} pixels at training size",
            )
    return True


def _validate_synthetic_quotas(
    rows: list[dict[str, str]],
    observed_negatives: Counter[str],
    config: BuildConfig,
    manifest_path: Path,
    issues: list[ValidationIssue],
) -> None:
    quotas = generation_quotas(config)
    for row in rows:
        if row.get("origin") != "synthetic":
            _issue(
                issues,
                "invalid_synthetic_origin",
                row.get("output_id", manifest_path),
                "synthetic-only datasets may contain only synthetic manifest rows",
            )
    for split, quota in quotas.items():
        selected = [row for row in rows if row.get("split") == split]
        if len(selected) != quota.total:
            _issue(
                issues,
                "synthetic_split_count_mismatch",
                split,
                f"manifest has {len(selected)}, expected {quota.total}",
            )
        if observed_negatives[split] != quota.negatives:
            _issue(
                issues,
                "negative_count_mismatch",
                split,
                f"found {observed_negatives[split]} empty labels, expected {quota.negatives}",
            )
        positives = [row for row in selected if row.get("negative", "").lower() != "true"]
        mh_positives = sum(row.get("state") == "MH" for row in positives)
        if mh_positives != quota.mh_positives:
            _issue(
                issues,
                "mh_positive_count_mismatch",
                split,
                f"found {mh_positives}, expected {quota.mh_positives}",
            )
        double_rows = sum(row.get("plate_layout") == "double" for row in positives)
        if double_rows != quota.double_row_positives:
            _issue(
                issues,
                "double_row_count_mismatch",
                split,
                f"found {double_rows}, expected {quota.double_row_positives}",
            )
        low_light = sum(row.get("effect") == "night" for row in selected)
        if low_light != quota.low_light:
            _issue(
                issues,
                "low_light_count_mismatch",
                split,
                f"found {low_light}, expected {quota.low_light}",
            )
        adverse = sum(row.get("effect") == "rain" for row in selected)
        if adverse != quota.adverse:
            _issue(
                issues,
                "adverse_condition_count_mismatch",
                split,
                f"found {adverse}, expected {quota.adverse}",
            )


def _validate_synthetic_manifest_rows(
    root: Path,
    rows: list[dict[str, str]],
    issues: list[ValidationIssue],
) -> set[str]:
    output_ids = Counter(row.get("output_id", "") for row in rows)
    pair_keys = Counter(
        (row.get("split", ""), row.get("output_id", "")) for row in rows
    )
    for output_id, count in output_ids.items():
        if output_id and count > 1:
            _issue(
                issues,
                "duplicate_manifest_output_id",
                output_id,
                f"output_id appears in {count} manifest rows",
            )
    for pair_key, count in pair_keys.items():
        if pair_key[1] and count > 1:
            _issue(
                issues,
                "duplicate_manifest_pair",
                ":".join(pair_key),
                f"manifest pair appears in {count} rows",
            )

    claimed_crops: dict[str, list[str]] = defaultdict(list)
    valid_crops: set[str] = set()
    for row in rows:
        output_id = row.get("output_id", "")
        split = row.get("split", "")
        if split not in _SPLITS:
            _issue(
                issues,
                "invalid_manifest_split",
                output_id or "manifest",
                f"split {split!r} is not one of {_SPLITS}",
            )
            continue
        expected_image = f"detection/images/{split}/{output_id}.jpg"
        expected_label = f"detection/labels/{split}/{output_id}.txt"
        paths_match = (
            row.get("image_path") == expected_image
            and row.get("label_path") == expected_label
        )
        if not paths_match:
            _issue(
                issues,
                "invalid_manifest_pair_path",
                output_id or split,
                "image_path and label_path must match the split and output_id",
            )
        pair_exists = (root / expected_image).is_file() and (root / expected_label).is_file()
        if not pair_exists:
            _issue(
                issues,
                "missing_manifest_pair",
                output_id or split,
                "manifest row does not reference an existing image-label pair",
            )
        if row.get("negative", "").lower() == "true":
            continue
        ocr_path = row.get("ocr_path", "")
        if ocr_path:
            claimed_crops[ocr_path].append(output_id)
        expected_crop = f"ocr/images/{split}/{output_id}.jpg"
        if ocr_path != expected_crop:
            _issue(
                issues,
                "invalid_ocr_path",
                output_id or split,
                "positive OCR path must match its split and output_id",
            )
        crop = root / expected_crop
        if not crop.is_file():
            _issue(issues, "missing_ocr_crop", crop, "positive image has no linked OCR crop")
            continue
        if not row.get("ocr_sha256") or sha256_file(crop) != row.get("ocr_sha256"):
            _issue(issues, "ocr_checksum_mismatch", crop, "ocr_sha256 does not match manifest")
            continue
        if paths_match and pair_exists and ocr_path == expected_crop:
            valid_crops.add(expected_crop)
    for crop_path, output_ids in claimed_crops.items():
        if len(output_ids) > 1:
            _issue(
                issues,
                "duplicate_ocr_crop",
                crop_path,
                f"OCR crop is linked by {sorted(output_ids)}",
            )
    return valid_crops


def validate_dataset(root: Path, config: BuildConfig) -> ValidationReport:
    root = Path(root)
    issues: list[ValidationIssue] = []
    image_root = root / "detection" / "images"
    label_root = root / "detection" / "labels"
    images = sorted(image_root.glob("*/*.jpg"))
    labels = sorted(label_root.glob("*/*.txt"))
    image_by_key = {(path.parent.name, path.stem): path for path in images}
    label_by_key = {(path.parent.name, path.stem): path for path in labels}

    for key in sorted(image_by_key.keys() - label_by_key.keys()):
        _issue(issues, "missing_label", image_by_key[key], "image has no matching label")
    for key in sorted(label_by_key.keys() - image_by_key.keys()):
        _issue(issues, "missing_image", label_by_key[key], "label has no matching image")

    manifest_path = root / "metadata" / "generation_manifest.csv"
    rows = _read_manifest(manifest_path, issues)
    valid_ocr_crops = (
        _validate_synthetic_manifest_rows(root, rows, issues)
        if config.synthetic_only
        else set()
    )
    row_by_key = {
        (row.get("split", ""), Path(row.get("image_path", "")).stem): row
        for row in rows
    }
    observed_negatives: Counter[str] = Counter()

    for key in sorted(image_by_key.keys() & label_by_key.keys()):
        image_path = image_by_key[key]
        label_path = label_by_key[key]
        row = row_by_key.get(key)
        if row is None:
            _issue(issues, "missing_manifest_row", image_path, "pair has no generation manifest row")
        try:
            with Image.open(image_path) as image:
                image.load()
                image_size = image.size
        except (OSError, ValueError) as error:
            _issue(issues, "unreadable_image", image_path, str(error))
            continue
        negative = bool(row and row.get("negative", "").lower() == "true")
        has_box = _validate_label(label_path, image_size, negative, config, issues)
        if not has_box:
            observed_negatives[key[0]] += 1
        if row:
            for path, field in ((image_path, "image_sha256"), (label_path, "label_sha256")):
                expected = row.get(field, "")
                if not expected or sha256_file(path) != expected:
                    _issue(issues, "checksum_mismatch", path, f"{field} does not match manifest")
            ocr_eligible = row.get("ocr_eligible", "").lower() == "true"
            has_text = bool(row.get("plate_text", ""))
            if ocr_eligible and (negative or not has_text):
                _issue(issues, "invalid_ocr_eligibility", image_path, "OCR eligibility requires a visible plate with known text")

    split_counts = Counter(path.parent.name for path in images)
    targets = split_target_counts(config.target_images, config.split)
    for split in ("train", "val", "test"):
        if split_counts[split] != targets[split]:
            _issue(
                issues,
                "split_count_mismatch",
                image_root / split,
                f"found {split_counts[split]}, expected {targets[split]}",
            )

    families: dict[str, set[str]] = defaultdict(set)
    exact_hashes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family = row.get("source_family", "")
        split = row.get("split", "")
        if family:
            families[family].add(split)
        checksum = row.get("image_sha256", "")
        if checksum:
            exact_hashes[checksum].add(split)
    for family, splits in families.items():
        if len(splits) > 1:
            _issue(issues, "cross_split_family", family, f"family appears in {sorted(splits)}")
    for checksum, splits in exact_hashes.items():
        if len(splits) > 1:
            _issue(issues, "cross_split_exact_duplicate", checksum[:12], f"identical image appears in {sorted(splits)}")

    if config.synthetic_only:
        _validate_synthetic_quotas(rows, observed_negatives, config, manifest_path, issues)
        if config.target_images == 50_000 and len(valid_ocr_crops) < 46_250:
            _issue(
                issues,
                "ocr_crop_count_mismatch",
                root / "ocr" / "images",
                f"found {len(valid_ocr_crops)} valid crops, expected at least 46250",
            )
    else:
        for split in ("val", "test"):
            selected = [row for row in rows if row.get("split") == split]
            if selected:
                real_share = sum(row.get("origin") == "real" for row in selected) / len(selected)
                if real_share < 0.80:
                    _issue(issues, "real_holdout_share", split, f"real share is {real_share:.3f}, below 0.80")

    known_positive = [
        row
        for row in rows
        if row.get("state") and row.get("negative", "").lower() != "true"
    ]
    if known_positive:
        mh_share = sum(row.get("state") == "MH" for row in known_positive) / len(known_positive)
        if not config.mh_share[0] <= mh_share <= config.mh_share[1]:
            _issue(issues, "mh_share", manifest_path, f"Maharashtra share {mh_share:.3f} is outside {config.mh_share}")
    if rows:
        negative_share = sum(row.get("negative", "").lower() == "true" for row in rows) / len(rows)
        if not config.negative_share[0] <= negative_share <= config.negative_share[1]:
            _issue(issues, "negative_share", manifest_path, f"negative share {negative_share:.3f} is outside {config.negative_share}")

    distributions: dict[str, Mapping[str, int]] = {
        field: dict(sorted(Counter(row.get(field, "unknown") or "unknown" for row in rows).items()))
        for field in (
            "split",
            "origin",
            "state",
            "vehicle_type",
            "viewpoint",
            "plate_style",
            "plate_layout",
            "effect",
            "negative",
        )
    }
    for field in ("vehicle_type", "viewpoint", "plate_layout", "effect"):
        if distributions[field].get("unknown", 0):
            _issue(issues, "unknown_category", manifest_path, f"{field} has unknown values", "warning")

    issues.sort(key=lambda item: (item.severity, item.code, item.path, item.message))
    return ValidationReport(
        image_count=len(images),
        label_count=len(labels),
        error_count=sum(issue.severity == "error" for issue in issues),
        warning_count=sum(issue.severity == "warning" for issue in issues),
        split_counts={split: split_counts[split] for split in ("train", "val", "test")},
        distributions=distributions,
        issues=tuple(issues),
    )
