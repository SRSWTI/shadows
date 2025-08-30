import asyncio
import random
from datetime import timedelta
from logging import Logger, LoggerAdapter
from typing import Annotated

from shadows import Shadow
from shadows.annotations import Logged
from shadows.dependencies import CurrentShadow, Perpetual, TaskLogger

from .common import run_example_workers


async def find(
    shadows: Shadow = CurrentShadow(),
    logger: LoggerAdapter[Logger] = TaskLogger(),
    perpetual: Perpetual = Perpetual(every=timedelta(seconds=3), automatic=True),
) -> None:
    logger.info("Starting find task")
    for i in range(1, 10 + 1):
        logger.debug("Scheduling flood task for item %d", i)
        await shadows.add(flood)(i)
    logger.info("Completed scheduling %d flood tasks", 10)


async def flood(
    item: Annotated[int, Logged],
    logger: LoggerAdapter[Logger] = TaskLogger(),
) -> None:
    logger.info("Starting flood task for item %s", item)
    sleep_time = random.uniform(0.5, 2)
    logger.debug("Sleeping for %.2f seconds", sleep_time)
    await asyncio.sleep(sleep_time)
    logger.info("Completed processing item %s", item)


    # async def monitor(
    #     shadows: Shadow = CurrentShadow(),
    #     logger: LoggerAdapter[Logger] = TaskLogger(),
    #     perpetual: Perpetual = Perpetual(every=timedelta(seconds=5), automatic=True),
    # ) -> None:
    #     logger.info("Running monitoring check")
    #     pending_tasks = await shadows.pending_count()
    #     logger.info("Current pending tasks: %d", pending_tasks)


# tasks = [find, flood, monitor]
tasks = [find, flood]


if __name__ == "__main__":
    logger = TaskLogger().new()
    logger.info("Starting example workers")
    asyncio.run(
        run_example_workers(
            workers=3,
            concurrency=8,
            tasks="examples.find_and_flood:tasks",
        )
    )
