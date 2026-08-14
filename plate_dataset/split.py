from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
from typing import Literal, Sequence, TYPE_CHECKING

from .config import BuildConfig
from .records import ImageRecord

if TYPE_CHECKING:
    from .builder import InsufficientSourceData


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SplitAssignment:
    split: SplitName
    source_family: str


def split_target_counts(total: int, ratios: tuple[float, float, float]) -> dict[SplitName, int]:
    val = int(round(total * ratios[1]))
    test = int(round(total * ratios[2]))
    return {"train": total - val - test, "val": val, "test": test}


def assign_splits(
    records: Sequence[ImageRecord], config: BuildConfig
) -> dict[str, SplitAssignment]:
    # Import here to avoid a module cycle while keeping the public exception in builder.
    from .builder import InsufficientSourceData

    if len({record.record_id for record in records}) != len(records):
        raise ValueError("record IDs must be unique")
    targets = split_target_counts(config.target_images, config.split)
    if not config.synthetic_only:
        minimum_real = {
            "val": int(math.ceil(targets["val"] * 0.80)),
            "test": int(math.ceil(targets["test"] * 0.80)),
        }
        available_real = sum(record.is_real for record in records)
        required_real = minimum_real["val"] + minimum_real["test"]
        if available_real < required_real:
            raise InsufficientSourceData(
                f"real holdout requirement cannot be met: required_real={required_real} "
                f"available_real={available_real}"
            )

    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.source_family].append(record)
    if config.synthetic_only:
        pool_targets = split_target_counts(len(records), config.split)
        families = sorted(
            grouped,
            key=lambda family: hashlib.sha256(
                f"{config.seed}:{family}".encode("utf-8")
            ).hexdigest(),
        )
        chosen: dict[SplitName, list[str]] = {"train": [], "val": [], "test": []}
        counts: dict[SplitName, int] = {"train": 0, "val": 0, "test": 0}
        for index, family in enumerate(families):
            unfilled = [split for split in chosen if not chosen[split]]
            if len(unfilled) == len(families) - index:
                split = unfilled[0]
            else:
                split = max(
                    chosen,
                    key=lambda candidate: (
                        pool_targets[candidate] - counts[candidate],
                        candidate == "train",
                    ),
                )
            chosen[split].append(family)
            counts[split] += len(grouped[family])
        return {
            record.record_id: SplitAssignment(split, family)
            for split, family_names in chosen.items()
            for family in family_names
            for record in grouped[family]
        }
    remaining = set(grouped)
    chosen: dict[SplitName, list[str]] = {"train": [], "val": [], "test": []}
    counts: dict[SplitName, int] = {"train": 0, "val": 0, "test": 0}

    def family_key(family: str, split: SplitName) -> tuple[int, str]:
        digest = hashlib.sha256(
            f"{config.seed}:{split}:{family}".encode("utf-8")
        ).hexdigest()
        return len(grouped[family]), digest

    def add_family(family: str, split: SplitName) -> None:
        chosen[split].append(family)
        counts[split] += len(grouped[family])
        remaining.remove(family)

    # Reserve the required real examples in each holdout before using any extras.
    for split in ("val", "test"):
        real_count = 0
        candidates = sorted(
            (
                family
                for family in remaining
                if any(record.is_real for record in grouped[family])
            ),
            key=lambda family: family_key(family, split),
        )
        for family in candidates:
            family_size = len(grouped[family])
            family_real = sum(record.is_real for record in grouped[family])
            if counts[split] + family_size > targets[split]:
                continue
            add_family(family, split)
            real_count += family_real
            if real_count >= minimum_real[split]:
                break
        if real_count < minimum_real[split]:
            raise InsufficientSourceData(
                f"family-safe real holdout cannot be formed for {split}: "
                f"required_real={minimum_real[split]} available_real={available_real}"
            )

    # Only enough existing records to overflow the final training capacity need to
    # be assigned as holdout extras. Synthetic families are preferred for diversity.
    overflow = max(
        0,
        sum(len(grouped[family]) for family in remaining) - targets["train"],
    )
    for split in ("val", "test"):
        deficit = min(targets[split] - counts[split], overflow)
        while deficit > 0:
            candidates = sorted(
                (
                    family
                    for family in remaining
                    if len(grouped[family]) <= deficit
                    and counts[split] + len(grouped[family]) <= targets[split]
                ),
                key=lambda family: (
                    any(record.is_real for record in grouped[family]),
                    *family_key(family, split),
                ),
            )
            if not candidates:
                raise InsufficientSourceData(
                    "source-family sizes cannot satisfy the requested split counts"
                )
            family = candidates[0]
            size = len(grouped[family])
            add_family(family, split)
            deficit -= size
            overflow -= size

    for family in sorted(remaining, key=lambda item: family_key(item, "train")):
        if counts["train"] + len(grouped[family]) > targets["train"]:
            raise InsufficientSourceData(
                "existing source records exceed the family-safe training capacity"
            )
        add_family(family, "train")

    assignments: dict[str, SplitAssignment] = {}
    for split, families in chosen.items():
        for family in families:
            for record in grouped[family]:
                assignments[record.record_id] = SplitAssignment(split, family)
    return assignments
