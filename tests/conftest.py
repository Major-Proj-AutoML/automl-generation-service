from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://stub:6379/0")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.data_service_client import DataServiceClient  # noqa: E402
from app.db import Base, get_session  # noqa: E402
from app.main import app, get_data_client, get_job_queue  # noqa: E402
from app import models  # noqa: F401, E402


class _StubJob:
    def __init__(self, job_id: str = "test-job-id"):
        self.id = job_id


class _StubQueue:
    def __init__(self):
        self.enqueued: list[tuple] = []

    def enqueue(self, func, *args, **kwargs):
        kwargs.pop("job_timeout", None)
        self.enqueued.append((func.__name__, args, kwargs))
        return _StubJob(f"job-{len(self.enqueued)}")


class _StubDataClient(DataServiceClient):
    def __init__(self):
        self.base_url = "stub"
        self.timeout = 1.0

    def get_dataset(self, dataset_id: int):
        return {
            "id": dataset_id, "name": f"ds_{dataset_id}", "source": "custom",
            "openml_id": None, "target_col": "label",
            "task_type": "binary_classification",
            "train_path": "/tmp/nope.csv", "test_path": "/tmp/nope.csv",
            "n_rows": 0, "n_cols": 0,
        }

    def ping(self) -> bool:
        return True


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def stub_queue():
    return _StubQueue()


@pytest.fixture
def client(db_session, stub_queue):
    def _override_session():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_data_client] = lambda: _StubDataClient()
    app.dependency_overrides[get_job_queue] = lambda: stub_queue
    with TestClient(app) as c:
        c._queue = stub_queue  # type: ignore[attr-defined]
        yield c
    app.dependency_overrides.clear()
