"""RQ worker entry point.

Run as: `python -m app.worker` inside the worker container. Consumes the same
queue name the web process enqueues to.
"""

from __future__ import annotations

import logging

from rq import Worker

from app.config import settings
from app.queue import get_queue, get_redis

logging.basicConfig(
    level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s"
)


def main() -> None:
    queue = get_queue()
    Worker([queue], connection=get_redis()).work()


if __name__ == "__main__":
    main()
