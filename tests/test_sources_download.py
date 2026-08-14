from __future__ import annotations

import json
from pathlib import Path
import subprocess
import zipfile

import pytest
import yaml

from plate_dataset.archives import safe_extract_zip
from plate_dataset.download import DownloadResult, download_source
from plate_dataset.sources import source_registry


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return path


def test_registry_enforces_known_license_policy() -> None:
    """Catches an unclear-license dataset becoming downloadable by default."""
    specs = {item.slug: item for item in source_registry()}

    assert specs["kedarsai/indian-license-plates-with-labels"].license_status == "allowed"
    assert specs["deepakat002/indian-vehicle-number-plate-yolo-annotation"].license_status == "allowed"
    assert specs["saisirishan/indian-vehicle-dataset"].license_status == "verify"
    assert specs["gauravsanwal/indian-licence-plate"].license_status == "verify"


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    """Catches a downloaded archive escaping its source directory."""
    archive = _write_zip(tmp_path / "bad.zip", {"../escape.txt": b"bad"})

    with pytest.raises(ValueError, match="unsafe archive member"):
        safe_extract_zip(archive, tmp_path / "out")


def test_safe_extract_returns_only_files_under_destination(tmp_path: Path) -> None:
    """Catches incorrect extraction paths or missing extracted files."""
    archive = _write_zip(
        tmp_path / "good.zip",
        {"dataset/images/a.jpg": b"image", "dataset/labels/a.txt": b"0 0.5 0.5 0.1 0.1"},
    )

    extracted = safe_extract_zip(archive, tmp_path / "out")

    assert {path.relative_to(tmp_path / "out").as_posix() for path in extracted} == {
        "dataset/images/a.jpg",
        "dataset/labels/a.txt",
    }


def test_safe_extract_rejects_symbolic_links(tmp_path: Path) -> None:
    """Catches an archive symlink redirecting a later write outside the destination."""
    archive = tmp_path / "link.zip"
    info = zipfile.ZipInfo("dataset/link")
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(info, "../../escape")

    with pytest.raises(ValueError, match="symbolic link"):
        safe_extract_zip(archive, tmp_path / "out")


def test_verify_source_is_skipped_without_local_approval(tmp_path: Path) -> None:
    """Catches accidental download of a source with unclear terms."""
    spec = next(x for x in source_registry() if x.license_status == "verify")

    def forbidden_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("Kaggle must not run for an unapproved source")

    result = download_source(
        spec,
        tmp_path / "downloads",
        tmp_path / "credentials",
        tmp_path / "licenses",
        runner=forbidden_runner,
    )

    assert result == DownloadResult(spec.slug, "skipped", None, "license_not_allowed", None)


def test_verify_source_downloads_after_explicit_local_approval(tmp_path: Path) -> None:
    """Catches a valid local license decision being ignored."""
    spec = next(x for x in source_registry() if x.license_status == "verify")
    license_dir = tmp_path / "licenses"
    license_dir.mkdir()
    decision = license_dir / f"{spec.slug.replace('/', '__')}.yaml"
    decision.write_text(yaml.safe_dump({"decision": "allowed"}), encoding="utf-8")

    def successful_runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        output = Path(command[command.index("-p") + 1])
        output.mkdir(parents=True, exist_ok=True)
        _write_zip(output / "download.zip", {"labels/a.txt": b"0 0.5 0.5 0.1 0.1"})
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    result = download_source(
        spec,
        tmp_path / "downloads",
        tmp_path / "credentials",
        license_dir,
        runner=successful_runner,
    )

    assert result.status == "downloaded"
    assert result.archive_path is not None and result.archive_path.exists()
    assert result.sha256 is not None and len(result.sha256) == 64


def test_cached_download_does_not_call_kaggle_again(tmp_path: Path) -> None:
    """Catches needless network calls that break resumability."""
    spec = next(x for x in source_registry() if x.license_status == "allowed")
    source_dir = tmp_path / "downloads" / spec.slug.replace("/", "__")
    source_dir.mkdir(parents=True)
    archive = _write_zip(source_dir / "download.zip", {"a.txt": b"data"})
    import hashlib

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(digest, encoding="ascii")

    def forbidden_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("cached source must not call Kaggle")

    result = download_source(
        spec,
        tmp_path / "downloads",
        tmp_path / "credentials",
        tmp_path / "licenses",
        runner=forbidden_runner,
    )

    assert result.status == "cached"
    assert result.sha256 == digest


def test_failed_download_result_does_not_expose_credential_path(tmp_path: Path) -> None:
    """Catches credential locations leaking into manifests or logs."""
    spec = next(x for x in source_registry() if x.license_status == "allowed")
    credential_dir = tmp_path / "credentials"

    def failed_runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr=f"could not read {credential_dir / 'kaggle.json'}",
        )

    result = download_source(
        spec,
        tmp_path / "downloads",
        credential_dir,
        tmp_path / "licenses",
        runner=failed_runner,
    )
    serialized = json.dumps(result.__dict__)

    assert result.status == "failed"
    assert str(credential_dir) not in serialized
    assert "kaggle.json" not in serialized
