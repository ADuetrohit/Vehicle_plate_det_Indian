from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

from PIL import Image

from .config import BuildConfig


class InsufficientStorage(RuntimeError):
    """Raised when a projected build would violate the configured disk reserve."""


@dataclass(frozen=True)
class StorageEstimate:
    projected_bytes: int
    reserve_bytes: int

    @property
    def required_bytes(self) -> int:
        return self.projected_bytes + self.reserve_bytes


def estimate_storage(
    config: BuildConfig, source_samples: Iterable[Path]
) -> StorageEstimate:
    """Estimate final and temporary build storage from measured JPEG source density."""
    samples = [Path(path) for path in source_samples]
    if not samples:
        raise ValueError("at least one source sample is required for storage estimation")

    measurements: list[tuple[int, int, int]] = []
    for path in samples:
        with Image.open(path) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"source image has invalid dimensions: {path}")
        measurements.append((path.stat().st_size, width, height))

    bytes_per_pixel = sum(
        size / (width * height) for size, width, height in measurements
    ) / len(measurements)
    average_scaled_pixels = sum(
        width
        * height
        * min(1.0, (config.max_scene_edge / max(width, height)) ** 2)
        for _, width, height in measurements
    ) / len(measurements)
    scene_bytes = math.ceil(
        bytes_per_pixel * average_scaled_pixels * config.target_images
    )
    positive_share = 1.0 - sum(config.negative_share) / 2.0
    crop_bytes = math.ceil(
        bytes_per_pixel
        * config.ocr_canvas[0]
        * config.ocr_canvas[1]
        * config.target_images
        * positive_share
    )
    raw_bytes = sum(size for size, _, _ in measurements)
    base_bytes = raw_bytes + scene_bytes + crop_bytes
    projected_bytes = base_bytes + math.ceil(base_bytes * 0.15)
    reserve_bytes = math.ceil(config.min_free_gb * 1024**3)
    return StorageEstimate(projected_bytes=projected_bytes, reserve_bytes=reserve_bytes)


def require_storage(estimate: StorageEstimate, free_bytes: int) -> None:
    """Reject a build if it cannot keep the required post-build reserve free."""
    if free_bytes < estimate.required_bytes:
        raise InsufficientStorage(
            "insufficient storage: "
            f"required_bytes={estimate.required_bytes} free_bytes={free_bytes}"
        )
