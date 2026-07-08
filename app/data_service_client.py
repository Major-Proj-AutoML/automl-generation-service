from __future__ import annotations

import httpx
from fastapi import HTTPException

from app.config import settings


class DataServiceClient:
    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        self.base_url = (base_url or settings.data_service_url).rstrip("/")
        self.timeout = timeout or settings.http_timeout_seconds

    def get_dataset(self, dataset_id: int) -> dict:
        try:
            r = httpx.get(
                f"{self.base_url}/datasets/{dataset_id}", timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"data-service unreachable at {self.base_url}: {exc}",
            )
        if r.status_code == 404:
            raise HTTPException(
                status_code=404, detail=f"Dataset {dataset_id} not found"
            )
        if r.status_code >= 400:
            raise HTTPException(
                status_code=502, detail=f"data-service error {r.status_code}: {r.text}"
            )
        return r.json()

    def ping(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=self.timeout)
            return r.status_code == 200
        except httpx.HTTPError:
            return False
