from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Literal

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
    distance: int
    occlusion: int


@dataclass(frozen=True)
class GenerationSlotPlan:
    negative_slots: frozenset[int]
    mh_positive_slots: frozenset[int]
    double_row_positive_slots: frozenset[int]
    conditions: tuple[str, ...]


_BASE_EFFECTS = ("day", "fog", "glare", "shadow", "motion", "compression")


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
    distance = _largest_remainder(_nearest_count(config.target_images, 0.075), totals)
    occlusion = _largest_remainder(_nearest_count(config.target_images, 0.075), totals)
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
            distance=distance[name],
            occlusion=occlusion[name],
        )
        for name in ("train", "val", "test")
    }


def _select_salted_slots(
    candidates: Iterable[int],
    count: int,
    *,
    seed: int,
    split: SplitName,
    salt: str,
) -> frozenset[int]:
    available = tuple(sorted(candidates))
    if not 0 <= count <= len(available):
        raise ValueError(
            f"cannot select {count} {salt} slots from {len(available)} candidates"
        )

    def score(slot: int) -> tuple[bytes, int]:
        payload = f"{seed}:{split}:{salt}:{slot}".encode("utf-8")
        return hashlib.sha256(payload).digest(), slot

    return frozenset(sorted(available, key=score)[:count])


def generation_slot_plan(config: BuildConfig, split: SplitName) -> GenerationSlotPlan:
    """Assign exact split quotas without coupling any dimension to slot prefixes."""
    quota = generation_quotas(config)[split]
    all_slots = frozenset(range(quota.total))
    negative_slots = _select_salted_slots(
        all_slots,
        quota.negatives,
        seed=config.seed,
        split=split,
        salt="negative",
    )
    positive_slots = all_slots - negative_slots
    mh_positive_slots = _select_salted_slots(
        positive_slots,
        quota.mh_positives,
        seed=config.seed,
        split=split,
        salt="state:mh",
    )
    double_row_positive_slots = _select_salted_slots(
        positive_slots,
        quota.double_row_positives,
        seed=config.seed,
        split=split,
        salt="layout:double",
    )

    effect_slots: dict[str, frozenset[int]] = {}
    remaining = set(all_slots)
    effect_targets = (
        ("occlusion", quota.occlusion, positive_slots),
        ("night", quota.low_light, all_slots),
        ("rain", quota.adverse, all_slots),
        ("distance", quota.distance, all_slots),
    )
    for effect, count, eligible in effect_targets:
        selected = _select_salted_slots(
            remaining.intersection(eligible),
            count,
            seed=config.seed,
            split=split,
            salt=f"effect:{effect}",
        )
        effect_slots[effect] = selected
        remaining.difference_update(selected)

    conditions = [""] * quota.total
    for effect, selected in effect_slots.items():
        for slot in selected:
            conditions[slot] = effect
    base_order = sorted(
        remaining,
        key=lambda slot: (
            hashlib.sha256(
                f"{config.seed}:{split}:effect:base:{slot}".encode("utf-8")
            ).digest(),
            slot,
        ),
    )
    for index, slot in enumerate(base_order):
        conditions[slot] = _BASE_EFFECTS[index % len(_BASE_EFFECTS)]

    return GenerationSlotPlan(
        negative_slots=negative_slots,
        mh_positive_slots=mh_positive_slots,
        double_row_positive_slots=double_row_positive_slots,
        conditions=tuple(conditions),
    )
