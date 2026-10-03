from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from src.datasets.funsd import FunsdAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the public FUNSD dataset")
    parser.add_argument("--output", type=Path, default=Path("data/funsd"))
    parser.add_argument("--url", default=FunsdAdapter.source_url)
    parser.add_argument("--sha256", help="Optional expected SHA-256 digest for the archive")
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory")
    args = parser.parse_args()
    if (
        args.output.exists()
        and (not args.output.is_dir() or any(args.output.iterdir()))
        and not args.force
    ):
        raise SystemExit(f"Output directory is not empty: {args.output}; use --force to replace it")

    with tempfile.TemporaryDirectory(prefix="funsd-") as temporary:
        archive = Path(temporary) / "funsd.zip"
        try:
            urllib.request.urlretrieve(args.url, archive)
            _verify_archive(archive, args.sha256)
            extracted = Path(temporary) / "extracted"
            with zipfile.ZipFile(archive) as archive_file:
                archive_file.extractall(extracted)
            source = _find_dataset_root(extracted)
            if args.output.exists():
                shutil.rmtree(args.output)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, args.output)
            manifest = FunsdAdapter(args.output).manifest()
            (args.output / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        except (OSError, ValueError, zipfile.BadZipFile, urllib.error.URLError) as error:
            raise SystemExit(f"FUNSD setup failed: {error}") from error
    print(f"FUNSD dataset installed at {args.output}")


def _verify_archive(archive: Path, expected_sha256: str | None) -> None:
    if not zipfile.is_zipfile(archive):
        raise ValueError("downloaded file is not a valid ZIP archive")
    with zipfile.ZipFile(archive) as archive_file:
        corrupt_file = archive_file.testzip()
        if corrupt_file is not None:
            raise ValueError(f"ZIP integrity check failed for {corrupt_file}")
    if expected_sha256:
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest.casefold() != expected_sha256.casefold():
            raise ValueError("downloaded archive SHA-256 does not match --sha256")


def _find_dataset_root(extracted: Path) -> Path:
    for candidate in (extracted, extracted / "dataset"):
        if (candidate / "training_data").is_dir() or (candidate / "testing_data").is_dir():
            return candidate
    raise ValueError("archive does not contain FUNSD training_data or testing_data directories")


if __name__ == "__main__":
    main()
