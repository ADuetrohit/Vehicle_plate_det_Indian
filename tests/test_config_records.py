from pathlib import Path

import pytest

from plate_dataset.config import load_config
from plate_dataset.records import Box


def test_box_normalizes_to_yolo() -> None:
    """Catches swapped axes or incorrect center/size normalization."""
    box = Box(class_id=0, x_min=20, y_min=10, x_max=60, y_max=30)

    assert box.to_yolo(100, 50) == pytest.approx((0, 0.4, 0.4, 0.4, 0.4))


def test_box_rejects_inverted_coordinates() -> None:
    """Catches acceptance of zero-area or inverted annotations."""
    with pytest.raises(ValueError, match="positive area"):
        Box(class_id=0, x_min=20, y_min=10, x_max=20, y_max=30)


def test_box_rejects_non_plate_class() -> None:
    """Catches accidental introduction of a second detection class."""
    with pytest.raises(ValueError, match="class ID 0"):
        Box(class_id=1, x_min=1, y_min=1, x_max=2, y_max=2)


def test_default_config_exposes_approved_build_contract() -> None:
    """Catches a build silently using unapproved counts or split ratios."""
    cfg = load_config(Path("config/default.yaml"))

    assert cfg.target_images == 12_000
    assert cfg.max_images == 15_000
    assert cfg.mh_share == (0.60, 0.70)
    assert cfg.split == (0.80, 0.10, 0.10)


def test_config_rejects_split_that_does_not_sum_to_one(tmp_path: Path) -> None:
    """Catches split configurations that can drop or duplicate records."""
    config = tmp_path / "bad.yaml"
    config.write_text(
        """workspace: .
seed: 1
target_images: 100
max_images: 100
mh_share: [0.60, 0.70]
split: [0.80, 0.10, 0.20]
negative_share: [0.05, 0.10]
training_imgsz: 512
min_box_at_training_size: [8, 4]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sum to 1"):
        load_config(config)


def test_box_clips_coordinates_before_normalizing() -> None:
    """Catches labels escaping the image after source conversion."""
    box = Box(class_id=0, x_min=-10, y_min=5, x_max=110, y_max=55)

    assert box.to_yolo(100, 50) == pytest.approx((0, 0.5, 0.55, 1.0, 0.9))


def test_box_rejects_non_finite_coordinates() -> None:
    """Catches NaN coordinates that would poison detector training."""
    with pytest.raises(ValueError, match="finite"):
        Box(class_id=0, x_min=1, y_min=1, x_max=float("nan"), y_max=2)


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    """Catches misspelled settings that would otherwise be silently ignored."""
    config = tmp_path / "unknown.yaml"
    config.write_text(
        """workspace: .
seed: 1
target_images: 100
max_images: 120
mh_share: [0.60, 0.70]
split: [0.80, 0.10, 0.10]
negative_share: [0.05, 0.10]
training_imgsz: 512
min_box_at_training_size: [8, 4]
typo_target: 999
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="config keys"):
        load_config(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_images", 0),
        ("max_images", 50),
        ("mh_share", [0.75, 0.60]),
        ("negative_share", [-0.1, 0.1]),
        ("training_imgsz", 0),
        ("min_box_at_training_size", [8, 0]),
    ],
)
def test_config_rejects_invalid_ranges(
    tmp_path: Path, field: str, value: object
) -> None:
    """Catches non-positive counts and malformed probability ranges."""
    raw = {
        "workspace": ".",
        "seed": 1,
        "target_images": 100,
        "max_images": 120,
        "mh_share": [0.60, 0.70],
        "split": [0.80, 0.10, 0.10],
        "negative_share": [0.05, 0.10],
        "training_imgsz": 512,
        "min_box_at_training_size": [8, 4],
    }
    raw[field] = value
    config = tmp_path / f"invalid-{field}.yaml"
    import yaml

    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid config"):
        load_config(config)
