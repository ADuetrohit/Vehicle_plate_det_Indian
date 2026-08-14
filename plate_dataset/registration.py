from __future__ import annotations

from dataclasses import dataclass
import re
from typing import AbstractSet

import numpy as np


SERIES_LETTERS = tuple("ABCDEFGHJKLMNPQRSTUVWXYZ")
OTHER_STATE_PREFIXES = (
    "AP",
    "BR",
    "CG",
    "DL",
    "GA",
    "GJ",
    "HR",
    "KA",
    "KL",
    "MP",
    "OD",
    "PB",
    "RJ",
    "TN",
    "TS",
    "UP",
    "UK",
    "WB",
)


@dataclass(frozen=True)
class PlateIdentity:
    state: str
    district: str
    series: str
    number: str
    compact_text: str
    display_lines: tuple[str, ...]


def generate_identity(
    rng: np.random.Generator,
    mh_probability: float,
    forbidden: AbstractSet[str],
) -> PlateIdentity:
    if not 0.0 <= mh_probability <= 1.0:
        raise ValueError("mh_probability must be between zero and one")
    forbidden_compact = {
        re.sub(r"[^A-Za-z0-9]", "", value).upper() for value in forbidden
    }
    for _ in range(10_000):
        state = "MH" if rng.random() < mh_probability else str(rng.choice(OTHER_STATE_PREFIXES))
        district_limit = 58 if state == "MH" else 99
        district = f"{int(rng.integers(1, district_limit + 1)):02d}"
        series_length = int(rng.choice((1, 2, 3), p=(0.10, 0.75, 0.15)))
        series = "".join(str(rng.choice(SERIES_LETTERS)) for _ in range(series_length))
        number = f"{int(rng.integers(1, 10_000)):04d}"
        compact = f"{state}{district}{series}{number}"
        if compact in forbidden_compact:
            continue
        return PlateIdentity(
            state=state,
            district=district,
            series=series,
            number=number,
            compact_text=compact,
            display_lines=(f"{state} {district} {series} {number}",),
        )
    raise RuntimeError("could not generate a fictitious registration")

