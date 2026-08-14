from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .composite import CompositeResult


@dataclass(frozen=True)
class EffectProfile:
    name: str = "day"
    brightness: float = 1.0
    contrast: float = 1.0
    gaussian_noise: float = 0.0
    motion_blur: int = 0
    jpeg_quality: int = 95
    rain: float = 0.0
    fog: float = 0.0
    glare: float = 0.0

    def __post_init__(self) -> None:
        if self.brightness <= 0 or self.contrast <= 0:
            raise ValueError("brightness and contrast must be positive")
        if self.gaussian_noise < 0:
            raise ValueError("gaussian_noise must be non-negative")
        if self.motion_blur < 0 or (self.motion_blur > 0 and self.motion_blur % 2 == 0):
            raise ValueError("motion_blur must be zero or a positive odd integer")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if not 0.0 <= self.rain <= 1.0 or not 0.0 <= self.fog <= 1.0 or not 0.0 <= self.glare <= 1.0:
            raise ValueError("rain, fog, and glare must be between zero and one")


def named_effect_profile(name: str, rng: np.random.Generator) -> EffectProfile:
    if name == "day":
        return EffectProfile(name, brightness=float(rng.uniform(0.90, 1.15)), contrast=float(rng.uniform(0.9, 1.1)), jpeg_quality=int(rng.integers(82, 99)))
    if name == "night":
        return EffectProfile(name, brightness=float(rng.uniform(0.35, 0.65)), contrast=float(rng.uniform(0.75, 1.05)), gaussian_noise=float(rng.uniform(5, 16)), jpeg_quality=int(rng.integers(55, 86)), glare=float(rng.uniform(0.05, 0.35)))
    if name == "rain":
        return EffectProfile(name, brightness=float(rng.uniform(0.60, 0.90)), contrast=float(rng.uniform(0.75, 1.0)), gaussian_noise=float(rng.uniform(2, 9)), jpeg_quality=int(rng.integers(50, 86)), rain=float(rng.uniform(0.30, 0.80)))
    if name == "fog":
        return EffectProfile(name, brightness=float(rng.uniform(0.75, 1.05)), contrast=float(rng.uniform(0.55, 0.85)), jpeg_quality=int(rng.integers(60, 91)), fog=float(rng.uniform(0.25, 0.70)))
    if name == "glare":
        return EffectProfile(name, brightness=float(rng.uniform(0.85, 1.15)), contrast=float(rng.uniform(0.8, 1.1)), jpeg_quality=int(rng.integers(60, 91)), glare=float(rng.uniform(0.35, 0.85)))
    if name == "shadow":
        return EffectProfile(name, brightness=float(rng.uniform(0.45, 0.75)), contrast=float(rng.uniform(0.7, 1.0)), jpeg_quality=int(rng.integers(65, 91)))
    if name == "motion":
        return EffectProfile(name, brightness=float(rng.uniform(0.7, 1.0)), motion_blur=int(rng.choice((7, 11, 15, 21))), jpeg_quality=int(rng.integers(50, 86)))
    if name == "compression":
        return EffectProfile(name, brightness=float(rng.uniform(0.8, 1.05)), gaussian_noise=float(rng.uniform(0, 8)), jpeg_quality=int(rng.integers(20, 50)))
    raise ValueError(f"unknown camera effect profile: {name}")


def apply_camera_effects(
    result: CompositeResult,
    profile: EffectProfile,
    rng: np.random.Generator,
) -> CompositeResult:
    image = result.image.astype(np.float32)
    image = (image - 127.5) * profile.contrast + 127.5
    image *= profile.brightness
    if profile.fog > 0:
        image = image * (1.0 - profile.fog * 0.65) + 235.0 * profile.fog * 0.65
    image = np.clip(image, 0, 255).astype(np.uint8)
    if profile.glare > 0:
        image = _apply_glare(image, profile.glare, rng)
    if profile.rain > 0:
        image = _apply_rain(image, profile.rain, rng)
    if profile.gaussian_noise > 0:
        noise = rng.normal(0.0, profile.gaussian_noise, size=image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if profile.motion_blur > 0:
        kernel = np.zeros((profile.motion_blur, profile.motion_blur), dtype=np.float32)
        kernel[profile.motion_blur // 2, :] = 1.0 / profile.motion_blur
        image = cv2.filter2D(image, -1, kernel)
    if profile.jpeg_quality < 100:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, profile.jpeg_quality]
        )
        if not success:
            raise RuntimeError("JPEG camera simulation failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    ocr_eligible = result.ocr_eligible and not (
        profile.motion_blur >= 17
        or profile.gaussian_noise >= 28
        or profile.jpeg_quality < 35
        or profile.fog >= 0.65
        or profile.glare >= 0.75
    )
    return CompositeResult(
        image=image,
        box=result.box,
        tags={**result.tags, "effect": profile.name},
        keep_detection=result.keep_detection,
        ocr_eligible=ocr_eligible,
    )


def _apply_rain(
    image: np.ndarray, strength: float, rng: np.random.Generator
) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    count = int(30 + strength * 220)
    for _ in range(count):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        length = int(rng.integers(5, max(6, int(8 + 22 * strength))))
        cv2.line(output, (x, y), (min(width - 1, x + 2), min(height - 1, y + length)), (205, 215, 225), 1)
    return output


def _apply_glare(
    image: np.ndarray, strength: float, rng: np.random.Generator
) -> np.ndarray:
    height, width = image.shape[:2]
    center_x = int(rng.integers(0, width))
    center_y = int(rng.integers(0, height))
    radius = max(10, int(min(width, height) * (0.08 + 0.25 * strength)))
    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    alpha = np.clip(1.0 - distance / radius, 0.0, 1.0) * strength * 0.75
    return np.clip(
        image.astype(np.float32) * (1.0 - alpha[:, :, None])
        + 255.0 * alpha[:, :, None],
        0,
        255,
    ).astype(np.uint8)
