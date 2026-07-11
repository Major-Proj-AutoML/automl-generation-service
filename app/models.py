from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, JSON, String, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class RunResultRecord(Base):
    __tablename__ = "run_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    condition: Mapped[str] = mapped_column(String, nullable=False, index=True)
    llm_backend: Mapped[str] = mapped_column(String, nullable=False, index=True)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    test_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_category: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    iterations_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_iterations: Mapped[int | None] = mapped_column(Integer, nullable=True)
    runtime_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    generated_code_path: Mapped[str | None] = mapped_column(String, nullable=True)
    # B2 structured reasoning + mechanical audit. Both nullable — B0/B1 never
    # produce a trace, and B2 runs whose trace failed extraction/verification
    # leave these NULL and are flagged via error_category.
    reasoning_trace: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    verification_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SweepJobRecord(Base):
    __tablename__ = "sweep_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rq_job_id: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    params: Mapped[dict] = mapped_column(JSON, nullable=False)
    total_cells: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completed_cells: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_cells: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
