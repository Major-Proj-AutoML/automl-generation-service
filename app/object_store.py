# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 The VESIT AutoML Project contributors

"""Thin MinIO wrapper for shipping generated scripts and model artifacts.

Best-effort. If MinIO is unreachable or unconfigured, uploads become no-ops so
DB rows stay authoritative and jobs never fail on storage hiccups.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from minio import Minio
from minio.error import S3Error

from app.config import settings

logger = logging.getLogger(__name__)

_client: Optional[Minio] = None
_ensured_buckets: set[str] = set()


def _sanitize(token: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in token)


def get_client() -> Optional[Minio]:
    global _client
    if _client is not None:
        return _client
    if not settings.minio_endpoint:
        return None
    _client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    return _client


def _ensure_bucket(client: Minio, bucket: str) -> None:
    if bucket in _ensured_buckets:
        return
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    _ensured_buckets.add(bucket)


def upload_file(bucket: str, key: str, path: Path) -> Optional[str]:
    client = get_client()
    if client is None or not path.exists():
        return None
    try:
        _ensure_bucket(client, bucket)
        client.fput_object(bucket, key, str(path))
        return f"s3://{bucket}/{key}"
    except S3Error:
        logger.exception("MinIO upload failed: %s -> %s/%s", path, bucket, key)
        return None


def upload_dir(bucket: str, key_prefix: str, dir_path: Path) -> list[str]:
    if get_client() is None or not dir_path.exists():
        return []
    urls: list[str] = []
    for f in sorted(dir_path.iterdir()):
        if f.is_file():
            url = upload_file(bucket, f"{key_prefix}/{f.name}", f)
            if url:
                urls.append(url)
    return urls


def build_key_prefix(
    dataset_id: int,
    condition: str,
    llm_backend: str,
    seed: int,
    iteration: int,
) -> str:
    return "/".join([
        f"dataset{dataset_id}",
        _sanitize(condition),
        _sanitize(llm_backend),
        f"seed{seed}",
        f"iter{iteration}",
    ])


def get_object(bucket: str, key: str) -> Optional[bytes]:
    client = get_client()
    if client is None:
        return None
    try:
        resp = client.get_object(bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()
    except S3Error as exc:
        if exc.code in ("NoSuchKey", "NoSuchBucket"):
            return None
        logger.exception("MinIO get failed: %s/%s", bucket, key)
        return None


def demo() -> None:
    """Round-trip self-check against a stub. Run: python -m app.object_store."""

    class _FakeClient:
        def __init__(self) -> None:
            self.store: dict[tuple[str, str], bytes] = {}
            self.buckets: set[str] = set()

        def bucket_exists(self, bucket: str) -> bool:
            return bucket in self.buckets

        def make_bucket(self, bucket: str) -> None:
            self.buckets.add(bucket)

        def fput_object(self, bucket: str, key: str, path: str) -> None:
            self.store[(bucket, key)] = Path(path).read_bytes()

        def get_object(self, bucket: str, key: str):
            data = self.store.get((bucket, key))
            if data is None:
                raise S3Error("NoSuchKey", "missing", "", "", "", None)

            class _R:
                def read(self_inner) -> bytes:
                    return data
                def close(self_inner) -> None: ...
                def release_conn(self_inner) -> None: ...
            return _R()

    global _client, _ensured_buckets
    _client = _FakeClient()  # type: ignore[assignment]
    _ensured_buckets = set()
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as f:
        f.write(b"print('hi')\n")
        p = Path(f.name)
    try:
        url = upload_file("scripts", "x/y.py", p)
        assert url == "s3://scripts/x/y.py", url
        assert get_object("scripts", "x/y.py") == b"print('hi')\n"
        assert get_object("scripts", "missing") is None
        assert build_key_prefix(7, "B2", "llama3:70b", 42, 1) == "dataset7/B2/llama3_70b/seed42/iter1"
        print("object_store demo OK")
    finally:
        p.unlink(missing_ok=True)
        _client = None
        _ensured_buckets = set()


if __name__ == "__main__":
    demo()
