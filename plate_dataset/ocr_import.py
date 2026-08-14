from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import re
from typing import Literal, Sequence
import zipfile

from PIL import Image


@dataclass(frozen=True)
class OCRRecord:
    image_name: str
    image_path: Path
    plate_text: str
    split: Literal["train", "val", "test"]
    source_id: str
    synthetic: bool
    reconciliation: Literal["preserved", "assigned_orphan"]


def import_existing_ocr(archive: Path, output_root: Path) -> list[OCRRecord]:
    with zipfile.ZipFile(archive) as handle:
        annotation_member = _member_ending(handle, "/annotations.csv")
        prefix = annotation_member[: -len("annotations.csv")]
        all_rows = _read_csv(handle, annotation_member)
        train_rows = _read_csv(handle, prefix + "train_annotations.csv")
        val_rows = _read_csv(handle, prefix + "val_annotations.csv")
        train_names = {_safe_relative(row["image_path"]) for row in train_rows}
        val_names = {_safe_relative(row["image_path"]) for row in val_rows}

        accepted: list[tuple[PurePosixPath, str]] = []
        for row in all_rows:
            if row.get("reviewed", "").strip().lower() != "true":
                continue
            if row.get("rejected", "").strip().lower() == "true":
                continue
            text = re.sub(r"[^A-Za-z0-9]", "", row.get("plate_text", "")).upper()
            if not text:
                continue
            accepted.append((_safe_relative(row["image_path"]), text))

        records: list[OCRRecord] = []
        split_counts = {"train": 0, "val": 0}
        for relative, text in sorted(accepted, key=lambda item: item[0].as_posix()):
            if relative in train_names:
                split: Literal["train", "val", "test"] = "train"
                reconciliation: Literal["preserved", "assigned_orphan"] = "preserved"
            elif relative in val_names:
                split = "val"
                reconciliation = "preserved"
            else:
                split = min(("train", "val"), key=lambda name: (split_counts[name], name))
                reconciliation = "assigned_orphan"
            split_counts[split] += 1
            member = prefix + relative.as_posix()
            try:
                payload = handle.read(member)
            except KeyError as exc:
                raise ValueError(f"OCR crop missing from archive: {member}") from exc
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
            output_dir = output_root / "images" / split
            output_dir.mkdir(parents=True, exist_ok=True)
            output_name = _collision_safe_name(relative, output_dir, payload)
            output_path = output_dir / output_name
            _write_bytes_atomic(output_path, payload)
            records.append(
                OCRRecord(
                    image_name=output_name,
                    image_path=output_path,
                    plate_text=text,
                    split=split,
                    source_id="existing_archive",
                    synthetic=False,
                    reconciliation=reconciliation,
                )
            )
    return records


def write_ocr_labels(records: Sequence[OCRRecord], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "image_path",
        "plate_text",
        "split",
        "source_id",
        "synthetic",
        "reconciliation",
    ]
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for record in sorted(records, key=lambda item: (item.split, item.image_name)):
                try:
                    image_path = record.image_path.relative_to(csv_path.parent).as_posix()
                except ValueError:
                    image_path = record.image_path.as_posix()
                writer.writerow(
                    {
                        "image_name": record.image_name,
                        "image_path": image_path,
                        "plate_text": record.plate_text,
                        "split": record.split,
                        "source_id": record.source_id,
                        "synthetic": str(record.synthetic).lower(),
                        "reconciliation": record.reconciliation,
                    }
                )
        os.replace(temporary, csv_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_csv(handle: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    try:
        text = handle.read(member).decode("utf-8-sig")
    except KeyError as exc:
        raise ValueError(f"required OCR annotation missing: {member}") from exc
    return list(csv.DictReader(io.StringIO(text)))


def _member_ending(handle: zipfile.ZipFile, suffix: str) -> str:
    matches = sorted(name for name in handle.namelist() if name.endswith(suffix))
    if len(matches) != 1:
        raise ValueError(f"expected one archive member ending with {suffix}")
    return matches[0]


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError(f"unsafe OCR image path: {value}")
    return path


def _collision_safe_name(
    relative: PurePosixPath, output_dir: Path, payload: bytes
) -> str:
    candidate = relative.name
    candidate_path = output_dir / candidate
    if not candidate_path.exists() or candidate_path.read_bytes() == payload:
        return candidate
    digest = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{Path(candidate).stem}_{digest}{Path(candidate).suffix.lower()}"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
