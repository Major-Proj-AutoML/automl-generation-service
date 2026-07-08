"""Tests for the RQ job functions themselves — called synchronously.

Uses monkeypatching so ``call_llm`` doesn't need a real Ollama server and
``execute_pipeline`` doesn't spawn a real subprocess.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _write_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "train.csv"
    rows = ["a,b,label"] + [f"{i%3},{i*0.1},{i%2}" for i in range(80)]
    csv.write_text("\n".join(rows), encoding="utf-8")
    return csv


def test_run_cell_success_path(tmp_path, db_session, monkeypatch):
    from app import jobs

    csv = _write_csv(tmp_path)

    class _FakeDataClient:
        def get_dataset(self, dataset_id):
            return {
                "id": dataset_id, "name": "fake", "source": "custom",
                "target_col": "label", "task_type": "binary_classification",
                "train_path": str(csv), "test_path": str(csv),
            }

    monkeypatch.setattr(jobs, "DataServiceClient", lambda: _FakeDataClient())
    monkeypatch.setattr(jobs, "call_llm", lambda prompt, backend, seed: "print('SCORE: 0.842')")

    def _fake_execute(pipeline, task_type, timeout_seconds):
        from src.contracts import RunResult
        return RunResult(
            dataset_id=pipeline.dataset_id, condition=pipeline.condition,
            llm_backend=pipeline.llm_backend, seed=pipeline.seed,
            iteration=pipeline.iteration,
            success=True, error_category=None, error_message=None,
            test_score=0.842, runtime_seconds=1.0,
            generated_code_path=str(tmp_path / "gen.py"),
        )

    monkeypatch.setattr(jobs, "execute_pipeline", _fake_execute)
    monkeypatch.setattr(jobs, "SessionLocal", lambda: db_session)

    run_id = jobs.run_cell(
        dataset_id=1, condition_key="b2_metafeature",
        llm_backend="llama3:70b", seed=42, max_iter=3, timeout_seconds=60,
    )

    from app.models import RunResultRecord
    rec = db_session.get(RunResultRecord, run_id)
    assert rec is not None
    assert rec.success is True
    assert rec.test_score == pytest.approx(0.842)
    assert rec.condition == "B2"
    assert rec.iterations_used == 1


def test_run_cell_retries_on_failure_then_succeeds(tmp_path, db_session, monkeypatch):
    from app import jobs

    csv = _write_csv(tmp_path)

    class _FakeDataClient:
        def get_dataset(self, dataset_id):
            return {
                "id": dataset_id, "name": "fake", "source": "custom",
                "target_col": "label", "task_type": "binary_classification",
                "train_path": str(csv), "test_path": str(csv),
            }

    monkeypatch.setattr(jobs, "DataServiceClient", lambda: _FakeDataClient())
    calls = {"n": 0}

    def _fake_llm(prompt, backend, seed):
        calls["n"] += 1
        return "print('SCORE: 0.7')"

    monkeypatch.setattr(jobs, "call_llm", _fake_llm)

    def _fake_execute(pipeline, task_type, timeout_seconds):
        from src.contracts import RunResult
        # first attempt fails, second succeeds
        if pipeline.iteration == 0:
            return RunResult(
                dataset_id=pipeline.dataset_id, condition=pipeline.condition,
                llm_backend=pipeline.llm_backend, seed=pipeline.seed,
                iteration=0,
                success=False, error_category="missing_name",
                error_message="NameError: 'RobustScaler' is not defined",
                test_score=None, runtime_seconds=0.5, generated_code_path="",
            )
        return RunResult(
            dataset_id=pipeline.dataset_id, condition=pipeline.condition,
            llm_backend=pipeline.llm_backend, seed=pipeline.seed,
            iteration=pipeline.iteration,
            success=True, error_category=None, error_message=None,
            test_score=0.75, runtime_seconds=1.0, generated_code_path="",
        )

    monkeypatch.setattr(jobs, "execute_pipeline", _fake_execute)
    monkeypatch.setattr(jobs, "SessionLocal", lambda: db_session)

    run_id = jobs.run_cell(
        dataset_id=1, condition_key="b0_naive", llm_backend="m",
        seed=42, max_iter=3, timeout_seconds=60,
    )
    from app.models import RunResultRecord
    rec = db_session.get(RunResultRecord, run_id)
    assert rec.success is True
    assert rec.iterations_used == 2
    assert calls["n"] == 2
