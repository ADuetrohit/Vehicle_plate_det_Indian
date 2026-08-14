from __future__ import annotations

import hashlib
from pathlib import Path
import re


PLATE_ALIASES = {"number_plate", "license_plate", "licence_plate", "plate"}
INDIAN_REGISTRATION = re.compile(
    r"^[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,3}[ -]?\d{1,4}$"
)


def normalize_label(value: str) -> str:
    return re.sub(r"[-\s]+", "_", value.strip().lower())


def plate_text_from_label(value: str) -> str | None:
    compact = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    if normalize_label(value) in PLATE_ALIASES:
        return None
    return compact if INDIAN_REGISTRATION.fullmatch(compact) else None


def is_plate_label(value: str) -> bool:
    return normalize_label(value) in PLATE_ALIASES or plate_text_from_label(value) is not None


def record_id(source_id: str, image_path: Path) -> str:
    payload = f"{source_id}:{image_path.as_posix()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]

