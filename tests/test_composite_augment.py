from __future__ import annotations

import numpy as np
import pytest

from plate_dataset.augment import EffectProfile, apply_camera_effects, named_effect_profile
from plate_dataset.composite import CompositeResult, composite_plate, erase_plate
from plate_dataset.records import Box
from plate_dataset.registration import PlateIdentity
from plate_dataset.render import PlateStyle, discover_font_paths, render_plate


def _scene() -> np.ndarray:
    x = np.linspace(40, 210, 400, dtype=np.uint8)
    y = np.linspace(20, 130, 300, dtype=np.uint8)[:, None]
    scene = np.empty((300, 400, 3), dtype=np.uint8)
    scene[:, :, 0] = x
    scene[:, :, 1] = y
    scene[:, :, 2] = 90
    return scene


def _plate():
    identity = PlateIdentity(
        state="MH",
        district="12",
        series="AB",
        number="1234",
        compact_text="MH12AB1234",
        display_lines=("MH 12 AB 1234",),
    )
    return render_plate(
        identity,
        PlateStyle("private", "single", wear=0.0),
        discover_font_paths(),
        np.random.default_rng(1),
    )


def test_composite_is_seeded_and_box_stays_inside_anchor() -> None:
    """Catches nondeterministic transforms or labels that escape the plate region."""
    scene = _scene()
    anchor = Box(0, 100, 80, 300, 150)

    first = composite_plate(scene, anchor, _plate(), np.random.default_rng(9))
    second = composite_plate(scene, anchor, _plate(), np.random.default_rng(9))

    assert np.array_equal(first.image, second.image)
    assert first.box == second.box
    assert 100 <= first.box.x_min < first.box.x_max <= 300
    assert 80 <= first.box.y_min < first.box.y_max <= 150
    assert not np.array_equal(first.image, scene)
    assert first.tags["state"] == "MH"


def test_composite_rejects_tiny_anchor() -> None:
    """Catches synthetic labels too small to survive detector resizing."""
    with pytest.raises(ValueError, match="anchor is too small"):
        composite_plate(
            _scene(),
            Box(0, 10, 10, 14, 12),
            _plate(),
            np.random.default_rng(2),
        )


def test_severe_blur_disables_ocr_but_keeps_detection() -> None:
    """Catches unreadable crops being kept as OCR labels."""
    result = composite_plate(
        _scene(), Box(0, 20, 20, 180, 70), _plate(), np.random.default_rng(2)
    )

    degraded = apply_camera_effects(
        result, EffectProfile(name="motion", motion_blur=21), np.random.default_rng(3)
    )

    assert degraded.keep_detection
    assert not degraded.ocr_eligible
    assert degraded.tags["effect"] == "motion"


def test_camera_effects_are_deterministic() -> None:
    """Catches irreproducible synthetic scenes for the same manifest seed."""
    result = composite_plate(
        _scene(), Box(0, 20, 20, 180, 70), _plate(), np.random.default_rng(2)
    )
    profile = EffectProfile(
        name="rain-night",
        brightness=0.55,
        gaussian_noise=12.0,
        rain=0.5,
        jpeg_quality=55,
    )

    first = apply_camera_effects(result, profile, np.random.default_rng(44))
    second = apply_camera_effects(result, profile, np.random.default_rng(44))

    assert np.array_equal(first.image, second.image)
    assert first.tags == second.tags


def test_distance_profile_reduces_detail_without_moving_detector_box() -> None:
    """Catches a distance metadata label that applies no real image degradation."""
    result = composite_plate(
        _scene(), Box(0, 20, 20, 180, 70), _plate(), np.random.default_rng(2)
    )
    try:
        profile = named_effect_profile("distance", np.random.default_rng(11))
    except ValueError:
        profile = EffectProfile(name="distance")

    assert getattr(profile, "distance_scale", 1.0) < 1.0
    degraded = apply_camera_effects(result, profile, np.random.default_rng(44))
    original_detail = sum(
        np.abs(np.diff(result.image.astype(np.float32), axis=axis)).mean()
        for axis in (0, 1)
    )
    degraded_detail = sum(
        np.abs(np.diff(degraded.image.astype(np.float32), axis=axis)).mean()
        for axis in (0, 1)
    )

    assert degraded.image.shape == result.image.shape
    assert degraded.box == result.box
    assert degraded.keep_detection == result.keep_detection
    assert degraded_detail < original_detail
    assert degraded.tags["effect"] == "distance"


def test_occlusion_profile_changes_only_part_of_visible_plate_interior() -> None:
    """Catches fake occlusion metadata or an overlay that hides the detector boundary."""
    image = np.full((120, 240, 3), 30, dtype=np.uint8)
    box = Box(0, 60, 40, 180, 80)
    image[40:80, 60:180] = 220
    result = CompositeResult(
        image=image,
        box=box,
        tags={},
        keep_detection=True,
        ocr_eligible=True,
    )
    try:
        profile = named_effect_profile("occlusion", np.random.default_rng(11))
    except ValueError:
        profile = EffectProfile(name="occlusion")

    assert 0.0 < getattr(profile, "occlusion", 0.0) < 0.5
    first = apply_camera_effects(result, profile, np.random.default_rng(44))
    second = apply_camera_effects(result, profile, np.random.default_rng(44))
    changed = np.any(first.image != image, axis=2)
    changed_y, changed_x = np.nonzero(changed)

    assert np.array_equal(first.image, second.image)
    assert first.box == box
    assert changed_x.size > 0
    assert changed_x.min() > box.x_min and changed_x.max() < box.x_max - 1
    assert changed_y.min() > box.y_min and changed_y.max() < box.y_max - 1
    assert changed.sum() < (box.x_max - box.x_min) * (box.y_max - box.y_min) * 0.5
    assert np.array_equal(first.image[40, 60:180], image[40, 60:180])
    assert np.array_equal(first.image[79, 60:180], image[79, 60:180])
    assert first.tags["effect"] == "occlusion"


@pytest.mark.parametrize("name", ["day", "night", "rain", "fog", "glare", "shadow", "motion", "compression"])
def test_named_profiles_are_valid(name: str) -> None:
    """Catches missing approved camera-condition profiles."""
    profile = named_effect_profile(name, np.random.default_rng(11))

    assert profile.name == name
    assert 1 <= profile.jpeg_quality <= 100
    assert profile.motion_blur == 0 or profile.motion_blur % 2 == 1


def test_erase_plate_changes_only_local_region() -> None:
    """Catches hard-negative generation erasing an entire vehicle scene."""
    scene = _scene()
    anchor = Box(0, 100, 80, 300, 150)

    erased = erase_plate(scene, anchor, np.random.default_rng(4))

    assert erased.shape == scene.shape
    assert np.array_equal(erased[:60], scene[:60])
    assert not np.array_equal(erased[85:145, 105:295], scene[85:145, 105:295])
