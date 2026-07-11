"""Job functions executed by the RQ worker.

Must be importable by full module path so RQ can pickle references. Each job
opens its own DB session — the web-process session is not shared.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.conditions import B0Naive, B1Schema, B2MetaFeatureGuided
from src.contracts import ErrorCategory, GeneratedPipeline, MetaFeatures, RunResult, TaskType
from src.execution.metrics import extract_reasoning
from src.execution.runner import execute_pipeline
from src.execution.verification import verify_reasoning
from src.experiments.datasets import build_task_description
from src.experiments.runner import build_error_feedback, call_llm
from src.meta_features.extractor import extract as extract_meta

from app.config import settings
from app.data_service_client import DataServiceClient
from app.db import SessionLocal
from app.models import RunResultRecord, SweepJobRecord

logger = logging.getLogger(__name__)

_CONDITION_MAP = {
    "b0_naive": B0Naive,
    "b1_schema": B1Schema,
    "b2_metafeature": B2MetaFeatureGuided,
}


def _load_train_df(train_path: str) -> pd.DataFrame:
    p = Path(train_path)
    if not p.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    return pd.read_csv(p)


def run_cell(
    dataset_id: int,
    condition_key: str,
    llm_backend: str,
    seed: int,
    max_iter: int = 3,
    timeout_seconds: int = 300,
) -> int:
    """Execute one (dataset, condition, model, seed) cell with iterative refinement.

    Writes the outcome to run_results and returns the row id. Runs in the RQ
    worker process — must open its own DB session and its own HTTP client.
    """
    session = SessionLocal()
    try:
        client = DataServiceClient()
        ds = client.get_dataset(dataset_id)
        df_train = _load_train_df(ds["train_path"])

        task_type = TaskType(ds["task_type"])
        target_col = ds["target_col"]
        dataset_name = ds["name"]

        meta = extract_meta(
            df_train,
            target_col=target_col,
            dataset_id=dataset_name,
            task_type=task_type,
        )
        task_description = build_task_description({
            "dataset_id": ds["id"],
            "dataset_name": dataset_name,
            "task_type": task_type,
            "target_col": target_col,
            "train_path": ds["train_path"],
            "test_path": ds["test_path"],
            "df_train": df_train,
            "df_test": None,
        })
        condition = _CONDITION_MAP[condition_key]()
        base_prompt = condition.build_prompt(
            task_description, meta, df_train.head(5), target_col
        )

        current_prompt = base_prompt
        best_result = None
        final_result = None
        iterations_used = 0

        for i in range(max_iter):
            iterations_used = i + 1
            try:
                code = call_llm(current_prompt, backend=llm_backend, seed=seed + i)
            except ConnectionError as exc:
                final_result = _make_infra_failure(
                    dataset_id, condition, llm_backend, seed, i, iterations_used,
                    max_iter, f"ollama unreachable: {exc}",
                )
                break

            pipeline = GeneratedPipeline(
                code=code,
                condition=condition.condition_name,
                llm_backend=llm_backend,
                dataset_id=dataset_name,
                seed=seed,
                iteration=i,
            )
            result = execute_pipeline(
                pipeline, task_type=task_type, timeout_seconds=timeout_seconds
            )

            # B2 must additionally emit a REASONING trace that we can
            # mechanically verify against the code and meta-features.
            # Skip for B0/B1 (they are not prompted to produce a trace).
            if result.success and condition_key == "b2_metafeature":
                result = _verify_b2_reasoning(result, code, meta)

            if result.success:
                best_result = result
                final_result = result
                break

            final_result = result
            current_prompt = base_prompt + build_error_feedback(
                error_message=result.error_message or "",
                error_category=result.error_category,
                prev_code=code,
            )

        record = _persist(
            session, dataset_id, condition, llm_backend, seed,
            best_result or final_result, iterations_used, max_iter,
        )
        return record.id
    finally:
        session.close()


def _make_infra_failure(dataset_id, condition, llm_backend, seed, iteration,
                       iterations_used, max_iter, message):
    from src.contracts import RunResult
    return RunResult(
        dataset_id=str(dataset_id),
        condition=condition.condition_name,
        llm_backend=llm_backend,
        seed=seed,
        iteration=iteration,
        success=False,
        error_category="infrastructure",
        error_message=message,
        test_score=None,
        runtime_seconds=0.0,
        generated_code_path="",
    )


def _persist(session, dataset_id, condition, llm_backend, seed, result,
             iterations_used, max_iter):
    rec = RunResultRecord(
        dataset_id=dataset_id,
        condition=condition.condition_name,
        llm_backend=llm_backend,
        seed=seed,
        iteration=result.iteration if hasattr(result, "iteration") else 0,
        success=result.success,
        test_score=result.test_score,
        error_category=result.error_category,
        error_message=(result.error_message or None) if result.error_message else None,
        iterations_used=iterations_used,
        max_iterations=max_iter,
        runtime_seconds=result.runtime_seconds,
        generated_code_path=result.generated_code_path or None,
    )
    session.add(rec)
    session.commit()
    session.refresh(rec)
    return rec


def run_sweep(sweep_id: int) -> int:
    """Executes every (dataset, condition, model, seed) cell for a sweep.

    Reads sweep params from Postgres, iterates cells, delegates each to
    ``run_cell``, updates progress on the sweep row. Returns count of succeeded
    cells.
    """
    session = SessionLocal()
    try:
        sweep = session.get(SweepJobRecord, sweep_id)
        if sweep is None:
            logger.error("Sweep %d not found", sweep_id)
            return 0

        sweep.status = "running"
        session.commit()

        params = sweep.params
        succeeded = 0
        try:
            cells = _expand_cells(params)
            sweep.total_cells = len(cells)
            session.commit()

            for dataset_id, condition_key, llm_backend, seed in cells:
                try:
                    run_cell(
                        dataset_id=dataset_id,
                        condition_key=condition_key,
                        llm_backend=llm_backend,
                        seed=seed,
                        max_iter=params.get("max_iter", 3),
                        timeout_seconds=params.get("timeout_seconds", 300),
                    )
                    sweep.completed_cells += 1
                    succeeded += 1
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Cell failed: %s", exc)
                    sweep.failed_cells += 1
                    sweep.completed_cells += 1
                session.commit()

            sweep.status = "completed"
            sweep.completed_at = datetime.now(timezone.utc)
        except Exception as exc:  # noqa: BLE001
            sweep.status = "failed"
            sweep.error_message = f"{type(exc).__name__}: {exc}"
        session.commit()
        return succeeded
    finally:
        session.close()


def _expand_cells(params: dict) -> list[tuple[int, str, str, int]]:
    out = []
    for ds_id in params["dataset_ids"]:
        for cond in params["conditions"]:
            for model in params["llm_backends"]:
                for seed in params["seeds"]:
                    out.append((int(ds_id), str(cond), str(model), int(seed)))
    return out


def _verify_b2_reasoning(
    result: RunResult, code: str, meta: MetaFeatures | None
) -> RunResult:
    """For a successful B2 run, extract the LLM's REASONING trace, verify it
    mechanically against the generated code and meta-features, save trace and
    verification report as sidecar files next to the generated code, and flip
    the result to failed if the trace is missing or unfaithful.

    Never raises — any error becomes a REASONING_UNFAITHFUL failure with a
    descriptive message.
    """
    code_path = Path(result.generated_code_path)
    stdout_path = Path(str(code_path) + ".stdout.txt")
    trace_path = Path(str(code_path).replace(".py", "") + ".trace.json")
    verification_path = Path(str(code_path).replace(".py", "") + ".verification.json")

    try:
        stdout = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
    except OSError as exc:
        return _mark_unfaithful(result, f"Could not read stdout sidecar: {exc}")

    trace = extract_reasoning(stdout)
    if trace is None:
        return _mark_unfaithful(
            result,
            "No REASONING: {...} line found in stdout, or the JSON was malformed.",
        )

    try:
        report = verify_reasoning(trace, code, meta)
    except Exception as exc:  # noqa: BLE001
        return _mark_unfaithful(result, f"Verification raised: {type(exc).__name__}: {exc}")

    # Persist the trace + verification report even on failure — the artifacts
    # are the point of the exercise.
    try:
        trace_path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
        verification_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to write reasoning sidecars: %s", exc)

    if not report.faithful:
        note = report.overall_notes or f"{report.n_faithful}/{report.n_decisions} decisions verified."
        return _mark_unfaithful(result, f"Reasoning trace unfaithful: {note}")

    return result


def _mark_unfaithful(result: RunResult, message: str) -> RunResult:
    """Return a copy of ``result`` re-classified as reasoning_unfaithful."""
    return result.model_copy(update=dict(
        success=False,
        error_category=ErrorCategory.REASONING_UNFAITHFUL.value,
        error_message=message,
    ))
