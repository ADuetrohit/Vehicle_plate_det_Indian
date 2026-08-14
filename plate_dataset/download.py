from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Callable, Literal

import yaml

from .sources import SourceSpec


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class DownloadResult:
    source_id: str
    status: Literal["downloaded", "cached", "skipped", "failed"]
    archive_path: Path | None
    reason: str
    sha256: str | None


def download_source(
    spec: SourceSpec,
    output_dir: Path,
    kaggle_config_dir: Path,
    license_dir: Path,
    *,
    runner: Runner = subprocess.run,
) -> DownloadResult:
    if not _license_is_allowed(spec, license_dir):
        return DownloadResult(
            spec.slug, "skipped", None, "license_not_allowed", None
        )

    source_dir = output_dir / spec.slug.replace("/", "__")
    source_dir.mkdir(parents=True, exist_ok=True)
    cached = _find_valid_cache(source_dir)
    if cached is not None:
        archive, digest = cached
        return DownloadResult(spec.slug, "cached", archive, "checksum_match", digest)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        spec.slug,
        "-p",
        str(source_dir),
    ]
    environment = {**os.environ, "KAGGLE_CONFIG_DIR": str(kaggle_config_dir)}
    completed = runner(
        command,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return DownloadResult(
            spec.slug, "failed", None, "kaggle_download_failed", None
        )

    archives = sorted(source_dir.glob("*.zip"))
    if not archives:
        return DownloadResult(spec.slug, "failed", None, "archive_missing", None)
    archive = archives[0]
    digest = _sha256(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        digest, encoding="ascii"
    )
    return DownloadResult(spec.slug, "downloaded", archive, "ok", digest)


def _license_is_allowed(spec: SourceSpec, license_dir: Path) -> bool:
    if spec.license_status == "allowed":
        return True
    if spec.license_status != "verify":
        return False
    decision_path = license_dir / f"{spec.slug.replace('/', '__')}.yaml"
    if not decision_path.is_file():
        return False
    decision = yaml.safe_load(decision_path.read_text(encoding="utf-8"))
    return isinstance(decision, dict) and decision.get("decision") == "allowed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_valid_cache(source_dir: Path) -> tuple[Path, str] | None:
    for archive in sorted(source_dir.glob("*.zip")):
        checksum_path = archive.with_suffix(archive.suffix + ".sha256")
        if not checksum_path.is_file():
            continue
        expected = checksum_path.read_text(encoding="ascii").strip().lower()
        actual = _sha256(archive)
        if expected == actual:
            return archive, actual
    return None
