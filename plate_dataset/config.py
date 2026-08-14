from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BuildConfig:
    workspace: Path
    seed: int
    target_images: int
    max_images: int
    mh_share: tuple[float, float]
    split: tuple[float, float, float]
    negative_share: tuple[float, float]
    training_imgsz: int
    min_box_at_training_size: tuple[int, int]


def load_config(path: Path) -> BuildConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "workspace",
        "seed",
        "target_images",
        "max_images",
        "mh_share",
        "split",
        "negative_share",
        "training_imgsz",
        "min_box_at_training_size",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError(f"config keys must be exactly {sorted(required)}")
    workspace = Path(raw["workspace"])
    if not workspace.is_absolute():
        workspace = (path.parent / workspace).resolve()

    config = BuildConfig(
        workspace=workspace,
        seed=int(raw["seed"]),
        target_images=int(raw["target_images"]),
        max_images=int(raw["max_images"]),
        mh_share=tuple(map(float, raw["mh_share"])),
        split=tuple(map(float, raw["split"])),
        negative_share=tuple(map(float, raw["negative_share"])),
        training_imgsz=int(raw["training_imgsz"]),
        min_box_at_training_size=tuple(
            map(int, raw["min_box_at_training_size"])
        ),
    )
    if abs(sum(config.split) - 1.0) > 1e-9:
        raise ValueError("split must sum to 1")
    probability_ranges = (config.mh_share, config.negative_share)
    ranges_valid = all(
        len(values) == 2 and 0.0 <= values[0] <= values[1] <= 1.0
        for values in probability_ranges
    )
    split_valid = len(config.split) == 3 and all(
        0.0 <= value <= 1.0 for value in config.split
    )
    positive_sizes = (
        config.target_images > 0
        and config.max_images >= config.target_images
        and config.training_imgsz > 0
        and len(config.min_box_at_training_size) == 2
        and all(value > 0 for value in config.min_box_at_training_size)
    )
    if not (ranges_valid and split_valid and positive_sizes):
        raise ValueError("invalid config value or range")
    return config
