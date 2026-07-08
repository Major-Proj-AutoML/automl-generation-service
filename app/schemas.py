from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


ConditionKey = Literal["b0_naive", "b1_schema", "b2_metafeature"]


class RunRequest(BaseModel):
    dataset_id: int
    condition: ConditionKey
    llm_backend: str = Field(..., description="Ollama model tag, e.g. 'llama3.3:70b'")
    seed: int = 42
    max_iter: int = 3
    timeout_seconds: int = 300


class RunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dataset_id: int
    condition: str
    llm_backend: str
    seed: int
    iteration: int
    success: bool
    test_score: Optional[float] = None
    error_category: Optional[str] = None
    error_message: Optional[str] = None
    iterations_used: Optional[int] = None
    max_iterations: Optional[int] = None
    runtime_seconds: Optional[float] = None
    generated_code_path: Optional[str] = None
    created_at: datetime


class SweepRequest(BaseModel):
    dataset_ids: list[int]
    conditions: list[ConditionKey]
    llm_backends: list[str]
    seeds: list[int]
    max_iter: int = 3
    timeout_seconds: int = 300


class SweepResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    rq_job_id: Optional[str]
    status: str
    params: dict
    total_cells: Optional[int]
    completed_cells: int
    failed_cells: int
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]


class EnqueueResponse(BaseModel):
    run_id: Optional[int] = None
    sweep_id: Optional[int] = None
    rq_job_id: str
    status_url: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["connected", "unreachable"]
    redis: Literal["connected", "unreachable"]
    data_service: Literal["reachable", "unreachable"]
