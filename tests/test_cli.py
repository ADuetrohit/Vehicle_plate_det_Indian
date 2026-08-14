from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml


@pytest.mark.parametrize(
    "script",
    [
        "download_sources.py",
        "convert_annotations.py",
        "generate_synthetic.py",
        "validate_dataset.py",
    ],
)
def test_cli_help_exits_zero(script: str) -> None:
    result = subprocess.run(
        [sys.executable, f"scripts/{script}", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_data_yaml_is_portable() -> None:
    data = yaml.safe_load(Path("detection/data.yaml").read_text(encoding="utf-8"))
    assert data == {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "number_plate"},
    }


def test_download_dry_run_does_not_create_workspace(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    workspace = tmp_path / "never-created"
    config.write_text(
        "\n".join(
            (
                f"workspace: '{workspace.as_posix()}'",
                "seed: 7",
                "target_images: 10",
                "max_images: 10",
                "mh_share: [0.60, 0.70]",
                "split: [0.80, 0.10, 0.10]",
                "negative_share: [0.05, 0.10]",
                "training_imgsz: 512",
                "min_box_at_training_size: [8, 4]",
            )
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/download_sources.py", "--config", str(config), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not workspace.exists()
    assert "credential_values_never_printed" in result.stdout
