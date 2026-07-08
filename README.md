# automl-generation-service

Async LLM generation microservice. Web on port **8003**, worker as a separate process.

## Responsibilities

- Enqueue single-cell runs and full sweeps as Redis/RQ jobs.
- Worker pulls jobs, fetches dataset from `automl-data-service`, calls Ollama, executes the generated code in a subprocess, retries with error feedback up to `max_iter` times.
- Persists each cell's outcome to the `run_results` table in Postgres.
- Tracks sweep progress in `sweep_jobs`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + DB + Redis + data-service reachability |
| POST | `/runs` | Enqueue a single-cell run |
| GET | `/runs` | List runs (filters: `dataset_id`, `condition`, `llm_backend`, `limit`) |
| GET | `/runs/{id}` | Fetch one run's result |
| POST | `/sweeps` | Enqueue a batch sweep (cartesian product) |
| GET | `/sweeps` | List recent sweeps |
| GET | `/sweeps/{id}` | Poll sweep progress |

## Running

**Web:**

```bash
uvicorn app.main:app --port 8003
```

**Worker (separate process/container):**

```bash
python -m app.worker
```

Both use the same `.env`, both need Postgres + Redis + data-service reachable.

## Depends on

- `automl-reusables`
- `automl-data-service` (HTTP)
- `automl-infra` (Postgres + Redis)
- Ollama running at `OLLAMA_URL` for actual generation

## Tests

Uses SQLite in-memory + stubbed queue/data-client/LLM. No infra needed:

```bash
pytest -v
```
