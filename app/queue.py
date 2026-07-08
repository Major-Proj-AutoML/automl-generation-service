"""Redis + RQ queue setup, wrapped so tests can substitute a fake."""

from __future__ import annotations

from typing import Any, Callable

import redis
from rq import Queue

from app.config import settings

_conn: redis.Redis | None = None
_queue: Queue | None = None


def get_redis() -> redis.Redis:
    global _conn
    if _conn is None:
        _conn = redis.Redis.from_url(settings.redis_url)
    return _conn


def get_queue(name: str = "automl-generation") -> Queue:
    global _queue
    if _queue is None:
        _queue = Queue(name, connection=get_redis())
    return _queue


def ping_redis() -> bool:
    try:
        return bool(get_redis().ping())
    except Exception:
        return False


def enqueue(func: Callable, *args: Any, **kwargs: Any):
    q = get_queue()
    return q.enqueue(func, *args, **kwargs)
