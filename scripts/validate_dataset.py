from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plate_dataset.config import load_config
from plate_dataset.reporting import (
    make_contact_sheets,
    write_statistics,
    write_validation_report,
)
from plate_dataset.validate import validate_dataset


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Validate the built dataset and write QA reports."
    )
    result.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--samples-per-sheet", type=int, default=16)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        print(f"dataset_root={config.workspace}")
        print("checks=pairs,boxes,checksums,families,splits,distributions")
        return 0
    report = validate_dataset(config.workspace, config)
    write_validation_report(
        report, config.workspace / "reports" / "validation_report.json"
    )
    write_statistics(
        report, config.workspace / "metadata" / "dataset_statistics.json"
    )
    make_contact_sheets(config.workspace, report, args.samples_per_sheet)
    print(
        f"images={report.image_count} labels={report.label_count} "
        f"errors={report.error_count} warnings={report.warning_count}"
    )
    return 0 if report.error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
