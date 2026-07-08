"""Orchestration logic. Enqueues jobs and reads state back from Postgres."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app import jobs
from app.models import RunResultRecord, SweepJobRecord


def enqueue_run(
    session: Session,
    queue,  # rq.Queue or a stub with .enqueue()
    dataset_id: int,
    condition: str,
    llm_backend: str,
    seed: int,
    max_iter: int,
    timeout_seconds: int,
) -> tuple[str, dict]:
    """Enqueue a single-cell run. Returns (rq_job_id, params_snapshot)."""
    job = queue.enqueue(
        jobs.run_cell,
        dataset_id=dataset_id,
        condition_key=condition,
        llm_backend=llm_backend,
        seed=seed,
        max_iter=max_iter,
        timeout_seconds=timeout_seconds,
        job_timeout=timeout_seconds * (max_iter + 1) + 60,
    )
    return job.id, {
        "dataset_id": dataset_id,
        "condition": condition,
        "llm_backend": llm_backend,
        "seed": seed,
        "max_iter": max_iter,
        "timeout_seconds": timeout_seconds,
    }


def create_sweep(
    session: Session,
    queue,
    dataset_ids: list[int],
    conditions: list[str],
    llm_backends: list[str],
    seeds: list[int],
    max_iter: int,
    timeout_seconds: int,
) -> SweepJobRecord:
    params = {
        "dataset_ids": dataset_ids,
        "conditions": conditions,
        "llm_backends": llm_backends,
        "seeds": seeds,
        "max_iter": max_iter,
        "timeout_seconds": timeout_seconds,
    }
    total = len(dataset_ids) * len(conditions) * len(llm_backends) * len(seeds)

    sweep = SweepJobRecord(
        status="queued",
        params=params,
        total_cells=total,
        completed_cells=0,
        failed_cells=0,
    )
    session.add(sweep)
    session.commit()
    session.refresh(sweep)

    job = queue.enqueue(
        jobs.run_sweep,
        sweep_id=sweep.id,
        job_timeout=max(3600, total * timeout_seconds),
    )
    sweep.rq_job_id = job.id
    session.commit()
    session.refresh(sweep)
    return sweep


def get_run(session: Session, run_id: int) -> RunResultRecord:
    rec = session.get(RunResultRecord, run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return rec


def list_runs(
    session: Session,
    dataset_id: int | None = None,
    condition: str | None = None,
    llm_backend: str | None = None,
    limit: int = 100,
) -> list[RunResultRecord]:
    q = select(RunResultRecord).order_by(desc(RunResultRecord.created_at))
    if dataset_id is not None:
        q = q.where(RunResultRecord.dataset_id == dataset_id)
    if condition is not None:
        q = q.where(RunResultRecord.condition == condition)
    if llm_backend is not None:
        q = q.where(RunResultRecord.llm_backend == llm_backend)
    q = q.limit(limit)
    return list(session.scalars(q))


def get_sweep(session: Session, sweep_id: int) -> SweepJobRecord:
    rec = session.get(SweepJobRecord, sweep_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Sweep {sweep_id} not found")
    return rec


def list_sweeps(session: Session, limit: int = 50) -> list[SweepJobRecord]:
    q = select(SweepJobRecord).order_by(desc(SweepJobRecord.created_at)).limit(limit)
    return list(session.scalars(q))
