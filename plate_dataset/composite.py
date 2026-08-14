from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from .records import Box
from .render import RenderedPlate


@dataclass(frozen=True)
class CompositeResult:
    image: np.ndarray
    box: Box
    tags: Mapping[str, str]
    keep_detection: bool
    ocr_eligible: bool


def composite_plate(
    scene: np.ndarray,
    anchor: Box,
    plate: RenderedPlate,
    rng: np.random.Generator,
) -> CompositeResult:
    if scene.ndim != 3 or scene.shape[2] != 3 or scene.dtype != np.uint8:
        raise ValueError("scene must be an RGB uint8 array")
    height, width = scene.shape[:2]
    clipped = anchor.clip(width, height)
    anchor_width = clipped.x_max - clipped.x_min
    anchor_height = clipped.y_max - clipped.y_min
    if anchor_width < 8 or anchor_height < 4:
        raise ValueError("anchor is too small for plate compositing")

    plate_array = np.asarray(plate.image.convert("RGB"), dtype=np.uint8)
    plate_height, plate_width = plate_array.shape[:2]
    source = np.float32(
        [[0, 0], [plate_width - 1, 0], [plate_width - 1, plate_height - 1], [0, plate_height - 1]]
    )
    x1, y1, x2, y2 = clipped.x_min, clipped.y_min, clipped.x_max, clipped.y_max
    destination = np.float32(
        [
            [x1 + anchor_width * rng.uniform(0.01, 0.05), y1 + anchor_height * rng.uniform(0.02, 0.14)],
            [x2 - anchor_width * rng.uniform(0.01, 0.05), y1 + anchor_height * rng.uniform(0.02, 0.14)],
            [x2 - anchor_width * rng.uniform(0.01, 0.05), y2 - anchor_height * rng.uniform(0.02, 0.14)],
            [x1 + anchor_width * rng.uniform(0.01, 0.05), y2 - anchor_height * rng.uniform(0.02, 0.14)],
        ]
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    warped_plate = cv2.warpPerspective(
        plate_array, transform, (width, height), flags=cv2.INTER_LINEAR
    )
    source_mask = np.full((plate_height, plate_width), 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(
        source_mask, transform, (width, height), flags=cv2.INTER_LINEAR
    )
    alpha = warped_mask.astype(np.float32)[:, :, None] / 255.0
    composite = np.clip(
        scene.astype(np.float32) * (1.0 - alpha)
        + warped_plate.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    result_box = Box(
        0,
        float(destination[:, 0].min()),
        float(destination[:, 1].min()),
        float(destination[:, 0].max()),
        float(destination[:, 1].max()),
    ).clip(width, height)
    return CompositeResult(
        image=composite,
        box=result_box,
        tags={
            "state": plate.identity.state,
            "plate_style": plate.style.category,
            "plate_layout": plate.style.layout,
        },
        keep_detection=True,
        ocr_eligible=plate.ocr_eligible,
    )


def erase_plate(
    scene: np.ndarray, anchor: Box, rng: np.random.Generator
) -> np.ndarray:
    if scene.ndim != 3 or scene.shape[2] != 3 or scene.dtype != np.uint8:
        raise ValueError("scene must be an RGB uint8 array")
    height, width = scene.shape[:2]
    clipped = anchor.clip(width, height)
    x1, y1 = int(clipped.x_min), int(clipped.y_min)
    x2, y2 = int(np.ceil(clipped.x_max)), int(np.ceil(clipped.y_max))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[max(0, y1 - 2) : min(height, y2 + 2), max(0, x1 - 2) : min(width, x2 + 2)] = 255
    bgr = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
    result = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    region = result[y1:y2, x1:x2].astype(np.int16)
    if region.size:
        noise = rng.integers(-3, 4, size=region.shape, dtype=np.int16)
        result[y1:y2, x1:x2] = np.clip(region + noise, 0, 255).astype(np.uint8)
    return result

