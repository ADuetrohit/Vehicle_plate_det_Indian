"""Rebuild the OCR split so held-out accuracy is measured on real photographs.

``ocr/labels.csv`` as generated puts every real crop in train/val and leaves the
test split 100% synthetic, so a clean test score would only prove the model can
read its own renderer. This script re-partitions the corpus:

* the reviewed real crops become the **test** split, after label repair;
* the synthetic crops become **train** and **val**;
* real labels that can not be parsed into a valid registration are quarantined
  to a separate CSV instead of being guessed at.

Nothing is moved on disk and ``ocr/labels.csv`` is not modified. The script only
writes new manifests that reference the existing image paths.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from charset import in_vocab, normalise_registration  # noqa: E402

FIELDS = ["image_path", "plate_text", "split", "origin", "layout", "source_status"]


def _stable_bucket(key: str) -> float:
    """Deterministic 0-1 hash so the train/val cut never shifts between runs."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _edit_distance_one(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` differ by exactly one substituted character."""
    if len(a) != len(b):
        return False
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    return diffs == 1


def _apply_corpus_consensus(test: list[dict], min_support: int = 3) -> list[tuple[str, str]]:
    """Re-point weakly-supported repairs at a well-attested near-identical label.

    The corpus contains repeated frames of the same vehicle, so a plate a human
    mistyped once often appears correctly many times over. ``MH040W9020`` is a
    single annotation of a plate written ``MH04DW9020`` eleven times elsewhere;
    glyph-confusion rules can not bridge ``0``/``D``, but frequency can.

    Only ``repaired`` rows are eligible, and only when a label that parsed
    untouched sits one substitution away with at least ``min_support`` support.
    """
    support = collections.Counter(
        row["plate_text"] for row in test if row["source_status"] == "exact"
    )
    well_attested = [text for text, n in support.items() if n >= min_support]
    fixes: list[tuple[str, str]] = []
    for row in test:
        if row["source_status"] != "repaired":
            continue
        current = row["plate_text"]
        if support.get(current, 0) >= min_support:
            continue
        matches = [t for t in well_attested if _edit_distance_one(current, t)]
        if len(matches) == 1:
            fixes.append((current, matches[0]))
            row["plate_text"] = matches[0]
            row["source_status"] = "repaired_consensus"
    return fixes


def build(root: Path, val_fraction: float) -> dict:
    labels_csv = root / "ocr" / "labels.csv"
    manifest_csv = root / "metadata" / "generation_manifest.csv"
    if not labels_csv.exists():
        raise SystemExit(f"missing {labels_csv}")

    layout_by_output = {}
    if manifest_csv.exists():
        with manifest_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("output_id"):
                    layout_by_output[row["output_id"]] = row.get("plate_layout", "unknown")

    with labels_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    train, val, test, rejected = [], [], [], []
    repair_counts = collections.Counter()

    for row in rows:
        image_path = (root / "ocr" / row["image_path"]).resolve()
        rel = image_path.relative_to(root).as_posix()
        is_synthetic = row["synthetic"] == "true"
        layout = layout_by_output.get(row.get("output_id", ""), "unknown")

        if is_synthetic:
            # Renderer output is trusted: the text was chosen before drawing.
            text = row["plate_text"]
            if not in_vocab(text):
                rejected.append({
                    "image_path": rel, "plate_text": text, "split": "",
                    "origin": "synthetic", "layout": layout,
                    "source_status": "out_of_vocab",
                })
                repair_counts["synthetic_out_of_vocab"] += 1
                continue
            record = {
                "image_path": rel, "plate_text": text, "split": "",
                "origin": "synthetic", "layout": layout, "source_status": "generated",
            }
            bucket = val if _stable_bucket(rel) < val_fraction else train
            record["split"] = "val" if bucket is val else "train"
            bucket.append(record)
            continue

        # Real crop: repair the human annotation before trusting it as truth.
        cleaned, status = normalise_registration(row["plate_text"])
        repair_counts[status] += 1
        if cleaned is None or not in_vocab(cleaned):
            rejected.append({
                "image_path": rel, "plate_text": row["plate_text"], "split": "",
                "origin": "real", "layout": "unknown", "source_status": "unparseable",
            })
            continue
        test.append({
            "image_path": rel, "plate_text": cleaned, "split": "test",
            "origin": "real", "layout": "unknown", "source_status": status,
        })

    consensus_fixes = _apply_corpus_consensus(test)
    repair_counts["repaired_by_consensus"] = len(consensus_fixes)

    out_dir = root / "ocr_training" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)

    with (out_dir / "rejected.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rejected)

    stats = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "rejected": len(rejected),
        "test_is_real_only": all(r["origin"] == "real" for r in test),
        "test_unique_registrations": len({r["plate_text"] for r in test}),
        "label_repair": dict(repair_counts),
        "train_layout": dict(collections.Counter(r["layout"] for r in train)),
        "val_layout": dict(collections.Counter(r["layout"] for r in val)),
    }
    (out_dir / "split_statistics.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    stats = build(args.root, args.val_fraction)
    print(json.dumps(stats, indent=2))
    if not stats["test_is_real_only"]:
        print("ERROR: test split is not purely real imagery", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
