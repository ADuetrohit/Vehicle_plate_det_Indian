from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plate_dataset.config import load_config
from plate_dataset.download import download_source
from plate_dataset.sources import source_registry


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Download license-approved plate datasets from Kaggle."
    )
    result.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    result.add_argument("--kaggle-config-dir", type=Path)
    result.add_argument(
        "--dry-run", action="store_true", help="Show source policy without writing files."
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    credential_dir = args.kaggle_config_dir or Path(
        os.environ.get("KAGGLE_CONFIG_DIR", config.workspace.parent)
    )
    specs = sorted(source_registry(), key=lambda item: item.priority)
    if args.dry_run:
        print(f"workspace={config.workspace}")
        print(f"kaggle_credentials_detected={(credential_dir / 'kaggle.json').is_file()}")
        print("credential_values_never_printed=true")
        for spec in specs:
            print(
                f"source={spec.slug} policy={spec.license_status} "
                f"license={spec.expected_license}"
            )
        return 0

    raw_dir = config.workspace / "raw"
    license_dir = config.workspace / "metadata" / "licenses"
    license_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in specs:
        result = download_source(spec, raw_dir, credential_dir, license_dir)
        results.append(result)
        print(f"{spec.slug}: {result.status} ({result.reason})")
        if spec.license_status == "allowed":
            decision = license_dir / f"{spec.slug.replace('/', '__')}.yaml"
            if not decision.exists():
                decision.write_text(
                    f"source: {spec.slug}\nlicense: {spec.expected_license}\n"
                    "decision: allowed\n",
                    encoding="utf-8",
                )

    manifest = config.workspace / "metadata" / "source_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        fields = ["source_id", "status", "archive_path", "reason", "sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["archive_path"] = (
                result.archive_path.relative_to(config.workspace).as_posix()
                if result.archive_path
                else ""
            )
            writer.writerow(row)
    return 1 if any(result.status == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
