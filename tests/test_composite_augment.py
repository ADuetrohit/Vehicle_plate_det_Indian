from __future__ import annotations

import numpy as np
import pytest

from plate_dataset.augment import EffectProfile, apply_camera_effects, named_effect_profile
from plate_dataset.composite import composite_plate, erase_plate
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
