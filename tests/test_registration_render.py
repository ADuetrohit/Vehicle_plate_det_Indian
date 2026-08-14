from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from PIL import Image
import pytest

from plate_dataset.registration import PlateIdentity, generate_identity
from plate_dataset.render import PlateStyle, discover_font_paths, render_plate


def _sample_identity() -> PlateIdentity:
    return PlateIdentity(
        state="MH",
        district="12",
        series="AB",
        number="1234",
        compact_text="MH12AB1234",
        display_lines=("MH 12 AB 1234",),
    )


def test_generated_mh_registration_is_valid_and_not_forbidden() -> None:
    """Catches malformed or copied registration identities."""
    identity = generate_identity(
        np.random.default_rng(7),
        mh_probability=1.0,
        forbidden={"MH12AB1234"},
    )

    assert re.fullmatch(r"MH\d{2}[A-HJ-NP-Z]{1,3}\d{4}", identity.compact_text)
    assert identity.compact_text != "MH12AB1234"
    assert identity.state == "MH"


def test_seeded_batch_hits_maharashtra_target() -> None:
    """Catches a generator that drifts outside the approved 60–70% MH range."""
    rng = np.random.default_rng(20260814)
    identities = [generate_identity(rng, 0.65, set()) for _ in range(2_000)]

    share = sum(item.state == "MH" for item in identities) / len(identities)
    assert 0.60 <= share <= 0.70


def test_series_never_uses_ambiguous_i_or_o() -> None:
    """Catches ambiguous glyphs entering synthetic OCR ground truth."""
    rng = np.random.default_rng(23)
    identities = [generate_identity(rng, 0.65, set()) for _ in range(1_000)]

    assert all("I" not in item.series and "O" not in item.series for item in identities)


def test_font_discovery_finds_a_readable_bold_font() -> None:
    """Catches a build starting without a usable Latin plate font."""
    paths = discover_font_paths()

    assert paths
    assert all(path.is_file() for path in paths)


@pytest.mark.parametrize(
    ("layout", "expected_size"),
    [("single", (520, 110)), ("double", (280, 200))],
)
def test_rendered_plate_has_expected_geometry(
    layout: str, expected_size: tuple[int, int]
) -> None:
    """Catches plate templates using inconsistent detection/OCR geometry."""
    rendered = render_plate(
        _sample_identity(),
        PlateStyle("private", layout),
        discover_font_paths(),
        np.random.default_rng(1),
    )

    assert rendered.image.size == expected_size
    assert rendered.image.mode == "RGB"
    assert rendered.ocr_eligible
    assert rendered.identity.compact_text == "MH12AB1234"


@pytest.mark.parametrize(
    ("category", "expected_pixel"),
    [
        ("private", (245, 245, 245)),
        ("commercial", (255, 204, 0)),
        ("electric_private", (0, 140, 70)),
        ("temporary", (255, 220, 0)),
    ],
)
def test_plate_category_uses_expected_background(
    category: str, expected_pixel: tuple[int, int, int]
) -> None:
    """Catches legal plate color categories being rendered interchangeably."""
    rendered = render_plate(
        _sample_identity(),
        PlateStyle(category, "single", wear=0.0),
        discover_font_paths(),
        np.random.default_rng(5),
    )

    assert rendered.image.getpixel((5, 5)) == expected_pixel
    assert rendered.image.getbbox() == (0, 0, 520, 110)


def test_excessive_wear_marks_crop_ineligible_for_ocr() -> None:
    """Catches unreadable synthetic text being emitted as OCR ground truth."""
    rendered = render_plate(
        _sample_identity(),
        PlateStyle("commercial", "single", wear=0.9),
        discover_font_paths(),
        np.random.default_rng(8),
    )

    assert isinstance(rendered.image, Image.Image)
    assert not rendered.ocr_eligible
