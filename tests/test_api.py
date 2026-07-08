from __future__ import annotations


def test_enqueue_run_returns_job_id(client):
    body = {
        "dataset_id": 1,
        "condition": "b2_metafeature",
        "llm_backend": "llama3:70b",
        "seed": 42,
        "max_iter": 3,
        "timeout_seconds": 60,
    }
    r = client.post("/runs", json=body)
    assert r.status_code == 202, r.text
    resp = r.json()
    assert resp["rq_job_id"] == "job-1"
    assert resp["run_id"] is None
    assert client._queue.enqueued[0][0] == "run_cell"


def test_enqueue_run_rejects_bad_condition(client):
    body = {
        "dataset_id": 1,
        "condition": "not_a_condition",
        "llm_backend": "x",
        "seed": 1,
    }
    r = client.post("/runs", json=body)
    assert r.status_code == 422


def test_create_sweep(client):
    body = {
        "dataset_ids": [1, 2],
        "conditions": ["b0_naive", "b2_metafeature"],
        "llm_backends": ["m1"],
        "seeds": [42, 43],
        "max_iter": 3,
        "timeout_seconds": 60,
    }
    r = client.post("/sweeps", json=body)
    assert r.status_code == 202, r.text
    resp = r.json()
    assert resp["sweep_id"] is not None
    sweep_id = resp["sweep_id"]

    r = client.get(f"/sweeps/{sweep_id}")
    assert r.status_code == 200
    sweep = r.json()
    # 2 datasets * 2 conditions * 1 model * 2 seeds
    assert sweep["total_cells"] == 8
    assert sweep["status"] == "queued"


def test_list_sweeps(client):
    body = {
        "dataset_ids": [1], "conditions": ["b0_naive"],
        "llm_backends": ["m1"], "seeds": [42],
    }
    client.post("/sweeps", json=body)
    client.post("/sweeps", json=body)
    r = client.get("/sweeps")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_get_missing_run_returns_404(client):
    r = client.get("/runs/999")
    assert r.status_code == 404


def test_list_runs_with_filters(client, db_session):
    from app.models import RunResultRecord

    for i in range(3):
        db_session.add(RunResultRecord(
            dataset_id=1, condition="B0", llm_backend="m1", seed=42+i,
            iteration=0, success=True, test_score=0.8, iterations_used=1,
            max_iterations=3, runtime_seconds=1.0,
        ))
    db_session.add(RunResultRecord(
        dataset_id=1, condition="B2", llm_backend="m1", seed=42,
        iteration=0, success=True, test_score=0.85, iterations_used=1,
        max_iterations=3, runtime_seconds=1.0,
    ))
    db_session.commit()

    r = client.get("/runs?condition=B0")
    assert r.status_code == 200
    assert len(r.json()) == 3

    r = client.get("/runs?dataset_id=1&condition=B2")
    assert len(r.json()) == 1
