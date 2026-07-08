"""FastAPI entry point for the automl-generation-service."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, FastAPI, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.data_service_client import DataServiceClient
from app.db import get_session
from app.queue import get_queue, ping_redis
from app.schemas import (
    EnqueueResponse,
    HealthResponse,
    RunRequest,
    RunResponse,
    SweepRequest,
    SweepResponse,
)
from app.service import (
    create_sweep,
    enqueue_run,
    get_run,
    get_sweep,
    list_runs,
    list_sweeps,
)

logging.basicConfig(level=settings.log_level)

app = FastAPI(
    title="automl-generation-service",
    version="0.1.0",
    description="Async LLM generation. Enqueues (dataset, condition, model, seed) cells "
    "via Redis/RQ. Workers call Ollama, execute code, retry on failure, write results to Postgres.",
)


def get_data_client() -> DataServiceClient:
    return DataServiceClient()


def get_job_queue():
    return get_queue()


@app.get("/health", response_model=HealthResponse)
def health(
    session: Session = Depends(get_session),
    client: DataServiceClient = Depends(get_data_client),
) -> HealthResponse:
    db_ok = False
    try:
        session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    redis_ok = ping_redis()
    ds_ok = client.ping()
    status = "ok" if (db_ok and redis_ok and ds_ok) else "degraded"
    return HealthResponse(
        status=status,
        database="connected" if db_ok else "unreachable",
        redis="connected" if redis_ok else "unreachable",
        data_service="reachable" if ds_ok else "unreachable",
    )


@app.post("/runs", response_model=EnqueueResponse, status_code=202)
def create_run(
    body: RunRequest,
    session: Session = Depends(get_session),
    queue=Depends(get_job_queue),
) -> EnqueueResponse:
    rq_job_id, _ = enqueue_run(
        session, queue,
        dataset_id=body.dataset_id,
        condition=body.condition,
        llm_backend=body.llm_backend,
        seed=body.seed,
        max_iter=body.max_iter,
        timeout_seconds=body.timeout_seconds,
    )
    return EnqueueResponse(
        rq_job_id=rq_job_id,
        status_url=f"/runs/by-rq-job/{rq_job_id}",
    )


@app.get("/runs", response_model=list[RunResponse])
def list_runs_endpoint(
    dataset_id: Optional[int] = Query(None),
    condition: Optional[str] = Query(None),
    llm_backend: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    session: Session = Depends(get_session),
) -> list[RunResponse]:
    rows = list_runs(session, dataset_id, condition, llm_backend, limit)
    return [RunResponse.model_validate(r) for r in rows]


@app.get("/runs/{run_id}", response_model=RunResponse)
def get_run_endpoint(
    run_id: int, session: Session = Depends(get_session)
) -> RunResponse:
    return RunResponse.model_validate(get_run(session, run_id))


@app.post("/sweeps", response_model=EnqueueResponse, status_code=202)
def create_sweep_endpoint(
    body: SweepRequest,
    session: Session = Depends(get_session),
    queue=Depends(get_job_queue),
) -> EnqueueResponse:
    sweep = create_sweep(
        session, queue,
        dataset_ids=body.dataset_ids,
        conditions=body.conditions,
        llm_backends=body.llm_backends,
        seeds=body.seeds,
        max_iter=body.max_iter,
        timeout_seconds=body.timeout_seconds,
    )
    return EnqueueResponse(
        sweep_id=sweep.id,
        rq_job_id=sweep.rq_job_id or "",
        status_url=f"/sweeps/{sweep.id}",
    )


@app.get("/sweeps", response_model=list[SweepResponse])
def list_sweeps_endpoint(
    limit: int = Query(50, ge=1, le=500),
    session: Session = Depends(get_session),
) -> list[SweepResponse]:
    return [SweepResponse.model_validate(s) for s in list_sweeps(session, limit)]


@app.get("/sweeps/{sweep_id}", response_model=SweepResponse)
def get_sweep_endpoint(
    sweep_id: int, session: Session = Depends(get_session)
) -> SweepResponse:
    return SweepResponse.model_validate(get_sweep(session, sweep_id))
