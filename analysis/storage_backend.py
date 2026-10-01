"""Where the scripts read and write data: a local folder or the original Google Cloud Storage bucket.

Every analysis script originally talked to gs://podcast-dissertation-audio directly. This module keeps that code
unchanged but lets it run against a local copy of the bucket (for example the external drive):

    export DISSERTATION_DATA_DIR="/Volumes/Caro Drive/podcast-dissertation-audio"
    python analysis/recalculate_influence_scores.py

In local mode:
- reads come from DISSERTATION_DATA_DIR, laid out like the bucket (enriched_transcripts/, tenet_enhanced_analysis/…);
- anything a script would upload is written under DISSERTATION_OUTPUT_DIR (default ./outputs), never into the data
  folder, so the original data is never modified.

With DISSERTATION_DATA_DIR unset, the real google-cloud-storage client is used, as before.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DATA_DIR_ENV = "DISSERTATION_DATA_DIR"
OUTPUT_DIR_ENV = "DISSERTATION_OUTPUT_DIR"


def data_dir() -> Path | None:
    d = os.environ.get(DATA_DIR_ENV)
    return Path(d).expanduser() if d else None


def output_dir() -> Path:
    return Path(os.environ.get(OUTPUT_DIR_ENV, "outputs")).expanduser()


def is_local() -> bool:
    return data_dir() is not None


class LocalBlob:
    def __init__(self, root: Path, name: str):
        self.name = name
        self._root = root

    @property
    def _path(self) -> Path:
        return self._root / self.name

    def download_as_bytes(self) -> bytes:
        return self._path.read_bytes()

    download_as_string = download_as_bytes

    def download_as_text(self, encoding: str = "utf-8") -> str:
        return self._path.read_text(encoding=encoding)

    def download_to_filename(self, filename) -> None:
        shutil.copyfile(self._path, filename)

    def exists(self) -> bool:
        return self._path.exists()

    def upload_from_filename(self, filename) -> None:
        # Never write into the data folder: uploads land in the output folder under the same relative path.
        dest = output_dir() / self.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(filename, dest)

    def upload_from_string(self, data, content_type=None) -> None:
        dest = output_dir() / self.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)


class LocalBucket:
    def __init__(self, root: Path, name: str = "local"):
        if not root.is_dir():
            raise FileNotFoundError(f"{DATA_DIR_ENV} is set to {root}, which is not a folder")
        self.root = root
        self.name = name

    def blob(self, name: str) -> LocalBlob:
        return LocalBlob(self.root, name)

    def list_blobs(self, prefix: str = ""):
        # GCS prefixes are plain string prefixes, not folders: "enriched_transcripts/Right wing__show__" is valid.
        folder, _, stem = prefix.rpartition("/")
        base = self.root / folder if folder else self.root
        if not base.is_dir():
            return
        for path in sorted(base.rglob("*")):
            if path.is_file():
                rel = path.relative_to(self.root).as_posix()
                if rel.startswith(prefix):
                    yield LocalBlob(self.root, rel)


class LocalClient:
    def __init__(self, project=None, **_):
        self.project = project

    def bucket(self, name: str) -> LocalBucket:
        return LocalBucket(data_dir(), name)

    def list_blobs(self, bucket_name: str, prefix: str = ""):
        return self.bucket(bucket_name).list_blobs(prefix=prefix)


class _Storage:
    """Stands in for ``google.cloud.storage``: ``storage.Client(project=...)`` works in both modes."""

    @staticmethod
    def Client(*args, **kwargs):  # noqa: N802 (mirrors the google-cloud-storage API)
        if is_local():
            return LocalClient(*args, **kwargs)
        try:
            from google.cloud import storage as gcs
        except ImportError as e:
            raise ImportError(
                f"google-cloud-storage is not installed. Either `pip install google-cloud-storage`, or set "
                f"{DATA_DIR_ENV} to a local copy of the bucket."
            ) from e
        return gcs.Client(*args, **kwargs)


storage = _Storage()


def destination(bucket_name: str, path: str) -> str:
    """Where an upload of ``path`` actually went, for log messages."""
    return str(output_dir() / path) if is_local() else f"gs://{bucket_name}/{path}"
