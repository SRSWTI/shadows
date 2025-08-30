from datetime import datetime, timedelta, timezone
from typing import cast
from unittest.mock import AsyncMock

import pytest
import redis.exceptions

from shadows.shadows import Shadow


async def test_shadows_aenter_propagates_connection_errors():
    """The shadows should propagate Redis connection errors"""

    shadows = Shadow(name="test-shadows", url="redis://nonexistent-host:12345/0")
    with pytest.raises(redis.exceptions.RedisError):
        await shadows.__aenter__()

    await shadows.__aexit__(None, None, None)


async def test_clear_empty_shadows(shadows: Shadow):
    """Clearing an empty shadows should succeed without error"""
    result = await shadows.clear()
    assert result == 0


async def test_clear_with_immediate_tasks(shadows: Shadow, the_task: AsyncMock):
    """Should clear immediate tasks from the stream"""
    shadows.register(the_task)

    await shadows.add(the_task)("arg1", kwarg1="value1")
    await shadows.add(the_task)("arg2", kwarg1="value2")
    await shadows.add(the_task)("arg3", kwarg1="value3")

    snapshot_before = await shadows.snapshot()
    assert len(snapshot_before.future) == 3

    result = await shadows.clear()
    assert result == 3

    snapshot_after = await shadows.snapshot()
    assert len(snapshot_after.future) == 0
    assert len(snapshot_after.running) == 0


async def test_clear_with_scheduled_tasks(shadows: Shadow, the_task: AsyncMock):
    """Should clear scheduled future tasks from the queue"""
    shadows.register(the_task)

    future = datetime.now(timezone.utc) + timedelta(seconds=60)
    await shadows.add(the_task, when=future)("arg1")
    await shadows.add(the_task, when=future + timedelta(seconds=1))("arg2")

    snapshot_before = await shadows.snapshot()
    assert len(snapshot_before.future) == 2

    result = await shadows.clear()
    assert result == 2

    snapshot_after = await shadows.snapshot()
    assert len(snapshot_after.future) == 0
    assert len(snapshot_after.running) == 0


async def test_clear_with_mixed_tasks(
    shadows: Shadow, the_task: AsyncMock, another_task: AsyncMock
):
    """Should clear both immediate and scheduled tasks"""
    shadows.register(the_task)
    shadows.register(another_task)

    future = datetime.now(timezone.utc) + timedelta(seconds=60)

    await shadows.add(the_task)("immediate1")
    await shadows.add(another_task)("immediate2")
    await shadows.add(the_task, when=future)("scheduled1")
    await shadows.add(another_task, when=future + timedelta(seconds=1))("scheduled2")

    snapshot_before = await shadows.snapshot()
    assert len(snapshot_before.future) == 4

    result = await shadows.clear()
    assert result == 4

    snapshot_after = await shadows.snapshot()
    assert len(snapshot_after.future) == 0
    assert len(snapshot_after.running) == 0


async def test_clear_with_parked_tasks(shadows: Shadow, the_task: AsyncMock):
    """Should clear parked tasks (tasks with specific keys)"""
    shadows.register(the_task)

    await shadows.add(the_task, key="task1")("arg1")
    await shadows.add(the_task, key="task2")("arg2")

    snapshot_before = await shadows.snapshot()
    assert len(snapshot_before.future) == 2

    result = await shadows.clear()
    assert result == 2

    snapshot_after = await shadows.snapshot()
    assert len(snapshot_after.future) == 0


async def test_clear_preserves_strikes(shadows: Shadow, the_task: AsyncMock):
    """Should not affect strikes when clearing"""
    shadows.register(the_task)

    await shadows.strike("the_task")
    await shadows.add(the_task)("arg1")

    # Check that the task wasn't scheduled due to the strike
    snapshot_before = await shadows.snapshot()
    assert len(snapshot_before.future) == 0  # Task was stricken, so not scheduled

    result = await shadows.clear()
    assert result == 0  # Nothing to clear since task was stricken

    # Strikes should still be in effect - clear doesn't affect strikes
    snapshot_after = await shadows.snapshot()
    assert len(snapshot_after.future) == 0


async def test_clear_returns_total_count(shadows: Shadow, the_task: AsyncMock):
    """Should return the total number of tasks cleared"""
    shadows.register(the_task)

    future = datetime.now(timezone.utc) + timedelta(seconds=60)

    await shadows.add(the_task)("immediate1")
    await shadows.add(the_task)("immediate2")
    await shadows.add(the_task, when=future)("scheduled1")
    await shadows.add(the_task, key="keyed1")("keyed1")

    result = await shadows.clear()
    assert result == 4


async def test_clear_no_redis_key_leaks(shadows: Shadow, the_task: AsyncMock):
    """Should not leak Redis keys when clearing tasks"""
    shadows.register(the_task)

    await shadows.add(the_task)("immediate1")
    await shadows.add(the_task)("immediate2")
    await shadows.add(the_task, key="keyed1")("keyed_task")

    future = datetime.now(timezone.utc) + timedelta(seconds=60)
    await shadows.add(the_task, when=future)("scheduled1")
    await shadows.add(the_task, when=future + timedelta(seconds=1))("scheduled2")

    async with shadows.redis() as r:
        keys_before = cast(list[str], await r.keys("*"))  # type: ignore
        keys_before_count = len(keys_before)

    result = await shadows.clear()
    assert result == 5

    async with shadows.redis() as r:
        keys_after = cast(list[str], await r.keys("*"))  # type: ignore
        keys_after_count = len(keys_after)

    assert keys_after_count <= keys_before_count

    snapshot = await shadows.snapshot()
    assert len(snapshot.future) == 0
    assert len(snapshot.running) == 0
