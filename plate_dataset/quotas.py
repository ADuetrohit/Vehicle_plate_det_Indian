from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from .config import BuildConfig
from .split import SplitName, split_target_counts


@dataclass(frozen=True)
class GenerationQuota:
    split: Literal["train", "val", "test"]
    total: int
    positives: int
    negatives: int
    mh_positives: int
    double_row_positives: int
    low_light: int
    adverse: int


def _largest_remainder(total: int, split_counts: dict[SplitName, int]) -> dict[SplitName, int]:
    grand_total = sum(split_counts.values())
    exact = {name: total * count / grand_total for name, count in split_counts.items()}
    allocated = {name: math.floor(value) for name, value in exact.items()}
    remainder = total - sum(allocated.values())
    for name in sorted(split_counts, key=lambda item: (-(exact[item] % 1), item))[:remainder]:
        allocated[name] += 1
    return allocated


def _nearest_count(total: int, share: float) -> int:
    return math.floor(total * share + 0.5)


def generation_quotas(config: BuildConfig) -> dict[str, GenerationQuota]:
    totals = split_target_counts(config.target_images, config.split)
    negatives = _largest_remainder(
        _nearest_count(config.target_images, config.negative_share[0]), totals
    )
    positives = {name: totals[name] - negatives[name] for name in totals}
    positive_total = sum(positives.values())
    mh_positives = _largest_remainder(_nearest_count(positive_total, 0.65), positives)
    double_row_positives = _largest_remainder(
        _nearest_count(positive_total, 0.20), positives
    )
    low_light = _largest_remainder(_nearest_count(config.target_images, 0.25), totals)
    adverse = _largest_remainder(_nearest_count(config.target_images, 0.15), totals)
    return {
        name: GenerationQuota(
            split=name,
            total=totals[name],
            positives=positives[name],
            negatives=negatives[name],
            mh_positives=mh_positives[name],
            double_row_positives=double_row_positives[name],
            low_light=low_light[name],
            adverse=adverse[name],
        )
        for name in ("train", "val", "test")
    }
