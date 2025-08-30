import asyncio
import datetime
import logging
import os
import random
import sys
import time
from datetime import timedelta

import redis.exceptions

from shadows import Shadow

from .tasks import hello

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("chaos.producer")


def now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


async def main(tasks_to_produce: int):
    shadows = Shadow(
        name=os.environ["SHADOW_NAME"],
        url=os.environ["SHADOW_URL"],
    )
    tasks_sent = 0
    while tasks_sent < tasks_to_produce:
        try:
            async with shadows:
                async with shadows.redis() as r:
                    for _ in range(tasks_sent, tasks_to_produce):
                        jitter = 5 * ((random.random() * 2) - 1)
                        when = now() + timedelta(seconds=jitter)
                        execution = await shadows.add(hello, when=when)()
                        await r.zadd("hello:sent", {execution.key: time.time()})
                        logger.info("Added task %s", execution.key)
                        tasks_sent += 1
        except redis.exceptions.ConnectionError:
            logger.warning(
                "producer: Redis connection error, retrying in 5s... "
                f"({tasks_sent}/{tasks_to_produce} tasks sent)"
            )
            await asyncio.sleep(5)


if __name__ == "__main__":
    tasks = int(sys.argv[1])
    asyncio.run(main(tasks))
