"""Tests that illustrate the core behavior of shadows.

These tests should serve as documentation highlighting the core behavior of shadows and
don't need to cover detailed edge cases.  Keep these tests as straightforward and clean
as possible to aid with understanding shadows.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from logging import LoggerAdapter
from typing import Annotated, AsyncGenerator, Callable
from unittest.mock import AsyncMock, call
from uuid import uuid4

import pytest

from shadows import (
    CurrentShadow,
    CurrentExecution,
    CurrentWorker,
    Depends,
    Shadow,
    Execution,
    ExponentialRetry,
    Logged,
    Perpetual,
    Retry,
    TaskArgument,
    TaskKey,
    TaskLogger,
    Timeout,
    Worker,
    tasks,
)


@pytest.fixture
def the_task() -> AsyncMock:
    task = AsyncMock()
    task.__name__ = "the_task"
    return task


async def test_immediate_task_execution(
    shadows: Shadow, worker: Worker, the_task: AsyncMock
):
    """shadows should execute a task immediately."""

    await shadows.add(the_task)("a", "b", c="c")

    await worker.run_until_finished()

    the_task.assert_awaited_once_with("a", "b", c="c")


async def test_immedate_task_execution_by_name(
    shadows: Shadow, worker: Worker, the_task: AsyncMock
):
    """shadows should execute a task immediately by name."""

    shadows.register(the_task)

    await shadows.add("the_task")("a", "b", c="c")

    await worker.run_until_finished()

    the_task.assert_awaited_once_with("a", "b", c="c")


async def test_scheduled_execution(
    shadows: Shadow, worker: Worker, the_task: AsyncMock, now: Callable[[], datetime]
):
    """shadows should execute a task at a specific time."""

    when = now() + timedelta(milliseconds=100)
    await shadows.add(the_task, when)("a", "b", c="c")

    await worker.run_until_finished()

    the_task.assert_awaited_once_with("a", "b", c="c")

    assert when <= now()


async def test_adding_is_idempotent(
    shadows: Shadow, worker: Worker, the_task: AsyncMock, now: Callable[[], datetime]
):
    """shadows should allow for rescheduling a task for later"""

    key = f"my-cool-task:{uuid4()}"

    soon = now() + timedelta(milliseconds=10)
    await shadows.add(the_task, soon, key=key)("a", "b", c="c")

    later = now() + timedelta(milliseconds=500)
    await shadows.add(the_task, later, key=key)("b", "c", c="d")

    await worker.run_until_finished()

    the_task.assert_awaited_once_with("a", "b", c="c")

    assert soon <= now() < later


async def test_rescheduling_later(
    shadows: Shadow, worker: Worker, the_task: AsyncMock, now: Callable[[], datetime]
):
    """shadows should allow for rescheduling a task for later"""

    key = f"my-cool-task:{uuid4()}"

    soon = now() + timedelta(milliseconds=10)
    await shadows.add(the_task, soon, key=key)("a", "b", c="c")

    later = now() + timedelta(milliseconds=100)
    await shadows.replace(the_task, later, key=key)("b", "c", c="d")

    await worker.run_until_finished()

    the_task.assert_awaited_once_with("b", "c", c="d")

    assert later <= now()


async def test_rescheduling_earlier(
    shadows: Shadow, worker: Worker, the_task: AsyncMock, now: Callable[[], datetime]
):
    """shadows should allow for rescheduling a task for earlier"""

    key = f"my-cool-task:{uuid4()}"

    soon = now() + timedelta(milliseconds=100)
    await shadows.add(the_task, soon, key)("a", "b", c="c")

    earlier = now() + timedelta(milliseconds=10)
    await shadows.replace(the_task, earlier, key)("b", "c", c="d")

    await worker.run_until_finished()

    the_task.assert_awaited_once_with("b", "c", c="d")

    assert earlier <= now()


async def test_rescheduling_by_name(
    shadows: Shadow, worker: Worker, the_task: AsyncMock, now: Callable[[], datetime]
):
    """shadows should allow for rescheduling a task for later"""

    key = f"my-cool-task:{uuid4()}"

    soon = now() + timedelta(milliseconds=100)
    await shadows.add(the_task, soon, key=key)("a", "b", c="c")

    later = now() + timedelta(milliseconds=200)
    await shadows.replace("the_task", later, key=key)("b", "c", c="d")

    await worker.run_until_finished()

    the_task.assert_awaited_once_with("b", "c", c="d")

    assert later <= now()


async def test_task_keys_are_idempotent_in_the_future(
    shadows: Shadow, worker: Worker, the_task: AsyncMock, now: Callable[[], datetime]
):
    """shadows should only allow one task with the same key to be scheduled or due"""

    key = f"my-cool-task:{uuid4()}"

    soon = now() + timedelta(milliseconds=10)
    await shadows.add(the_task, when=soon, key=key)("a", "b", c="c")
    await shadows.add(the_task, when=now(), key=key)("d", "e", c="f")

    await worker.run_until_finished()

    the_task.assert_awaited_once_with("a", "b", c="c")
    the_task.reset_mock()

    # It should be fine to run it afterward
    await shadows.add(the_task, key=key)("d", "e", c="f")

    await worker.run_until_finished()

    the_task.assert_awaited_once_with("d", "e", c="f")


async def test_task_keys_are_idempotent_between_the_future_and_present(
    shadows: Shadow, worker: Worker, the_task: AsyncMock, now: Callable[[], datetime]
):
    """shadows should only allow one task with the same key to be scheduled or due"""

    key = f"my-cool-task:{uuid4()}"

    soon = now() + timedelta(milliseconds=10)
    await shadows.add(the_task, when=now(), key=key)("a", "b", c="c")
    await shadows.add(the_task, when=soon, key=key)("d", "e", c="f")

    await worker.run_until_finished()

    the_task.assert_awaited_once_with("a", "b", c="c")
    the_task.reset_mock()

    # It should be fine to run it afterward
    await shadows.add(the_task, key=key)("d", "e", c="f")

    await worker.run_until_finished()

    the_task.assert_awaited_once_with("d", "e", c="f")


async def test_task_keys_are_idempotent_in_the_present(
    shadows: Shadow, worker: Worker, the_task: AsyncMock, now: Callable[[], datetime]
):
    """shadows should only allow one task with the same key to be scheduled or due"""

    key = f"my-cool-task:{uuid4()}"

    await shadows.add(the_task, when=now(), key=key)("a", "b", c="c")
    await shadows.add(the_task, when=now(), key=key)("d", "e", c="f")

    await worker.run_until_finished()

    the_task.assert_awaited_once_with("a", "b", c="c")
    the_task.reset_mock()

    # It should be fine to run it afterward
    await shadows.add(the_task, key=key)("d", "e", c="f")

    await worker.run_until_finished()

    the_task.assert_awaited_once_with("d", "e", c="f")


async def test_cancelling_future_task(
    shadows: Shadow, worker: Worker, the_task: AsyncMock, now: Callable[[], datetime]
):
    """shadows should allow for cancelling a task"""

    soon = now() + timedelta(milliseconds=100)
    execution = await shadows.add(the_task, soon)("a", "b", c="c")

    await shadows.cancel(execution.key)

    await worker.run_until_finished()

    the_task.assert_not_called()


async def test_cancelling_immediate_task(
    shadows: Shadow, worker: Worker, the_task: AsyncMock, now: Callable[[], datetime]
):
    """shadows can cancel a task that is scheduled immediately"""

    execution = await shadows.add(the_task, now())("a", "b", c="c")

    await shadows.cancel(execution.key)

    await worker.run_until_finished()

    the_task.assert_not_called()


async def test_cancellation_is_idempotent(
    shadows: Shadow, worker: Worker, the_task: AsyncMock, now: Callable[[], datetime]
):
    """Test that canceling the same task twice doesn't error."""
    key = f"test-task:{uuid4()}"

    # Schedule a task
    later = now() + timedelta(seconds=1)
    await shadows.add(the_task, later, key=key)("test")

    # Cancel it twice - both should succeed without error
    await shadows.cancel(key)
    await shadows.cancel(key)  # Should be idempotent

    # Run worker to ensure the task was actually cancelled
    await worker.run_until_finished()

    # Task should not have been executed since it was cancelled
    the_task.assert_not_called()


async def test_errors_are_logged(
    shadows: Shadow,
    worker: Worker,
    the_task: AsyncMock,
    now: Callable[[], datetime],
    caplog: pytest.LogCaptureFixture,
):
    """shadows should log errors when a task fails"""

    the_task.side_effect = Exception("Faily McFailerson")
    await shadows.add(the_task, now())("a", "b", c="c")

    await worker.run_until_finished()

    the_task.assert_awaited_once_with("a", "b", c="c")

    assert "Faily McFailerson" in caplog.text


async def test_supports_simple_linear_retries(
    shadows: Shadow, worker: Worker, now: Callable[[], datetime]
):
    """shadows should support simple linear retries"""

    calls = 0

    async def the_task(
        a: str,
        b: str = "b",
        retry: Retry = Retry(attempts=3),
    ) -> None:
        assert a == "a"
        assert b == "c"

        assert retry is not None

        nonlocal calls
        calls += 1

        assert retry.attempts == 3
        assert retry.attempt == calls

        raise Exception("Failed")

    await shadows.add(the_task)("a", b="c")

    await worker.run_until_finished()

    assert calls == 3


async def test_supports_simple_linear_retries_with_delay(
    shadows: Shadow, worker: Worker, now: Callable[[], datetime]
):
    """shadows should support simple linear retries with a delay"""

    calls = 0

    async def the_task(
        a: str,
        b: str = "b",
        retry: Retry = Retry(attempts=3, delay=timedelta(milliseconds=100)),
    ) -> None:
        assert a == "a"
        assert b == "c"

        assert retry is not None

        nonlocal calls
        calls += 1

        assert retry.attempts == 3
        assert retry.attempt == calls

        raise Exception("Failed")

    await shadows.add(the_task)("a", b="c")

    start = now()

    await worker.run_until_finished()

    total_delay = now() - start
    assert total_delay >= timedelta(milliseconds=200)

    assert calls == 3


async def test_supports_infinite_retries(
    shadows: Shadow, worker: Worker, now: Callable[[], datetime]
):
    """shadows should support infinite retries (None for attempts)"""

    calls = 0

    async def the_task(
        a: str,
        b: str = "b",
        retry: Retry = Retry(attempts=None),
    ) -> None:
        assert a == "a"
        assert b == "c"

        assert retry is not None
        assert retry.attempts is None

        nonlocal calls
        calls += 1

        assert retry.attempt == calls

        if calls < 3:
            raise Exception("Failed")

    await shadows.add(the_task)("a", b="c")

    await worker.run_until_finished()

    assert calls == 3


async def test_supports_exponential_backoff_retries(
    shadows: Shadow, worker: Worker, now: Callable[[], datetime]
):
    """shadows should support exponential backoff retries"""

    calls = 0

    async def the_task(
        a: str,
        b: str = "b",
        retry: Retry = ExponentialRetry(
            attempts=5,
            minimum_delay=timedelta(milliseconds=25),
            maximum_delay=timedelta(milliseconds=1000),
        ),
    ) -> None:
        assert a == "a"
        assert b == "c"

        assert isinstance(retry, ExponentialRetry)

        nonlocal calls
        calls += 1

        assert retry.attempts == 5
        assert retry.attempt == calls

        raise Exception("Failed")

    await shadows.add(the_task)("a", b="c")

    start = now()

    await worker.run_until_finished()

    total_delay = now() - start
    assert total_delay >= timedelta(milliseconds=25 + 50 + 100 + 200)

    assert calls == 5


async def test_supports_exponential_backoff_retries_under_maximum_delay(
    shadows: Shadow, worker: Worker, now: Callable[[], datetime]
):
    """shadows should support exponential backoff retries"""

    calls = 0

    async def the_task(
        a: str,
        b: str = "b",
        retry: Retry = ExponentialRetry(
            attempts=5,
            minimum_delay=timedelta(milliseconds=25),
            maximum_delay=timedelta(milliseconds=100),
        ),
    ) -> None:
        assert a == "a"
        assert b == "c"

        assert isinstance(retry, ExponentialRetry)

        nonlocal calls
        calls += 1

        assert retry.attempts == 5
        assert retry.attempt == calls

        raise Exception("Failed")

    await shadows.add(the_task)("a", b="c")

    start = now()

    await worker.run_until_finished()

    total_delay = now() - start
    assert total_delay >= timedelta(milliseconds=25 + 50 + 100 + 100)

    assert calls == 5


async def test_supports_requesting_current_shadows(
    shadows: Shadow, worker: Worker, now: Callable[[], datetime]
):
    """shadows should support providing the current shadows to a task"""

    called = False

    async def the_task(a: str, b: str, this_shadows: Shadow = CurrentShadow()):
        assert a == "a"
        assert b == "c"
        assert this_shadows is shadows

        nonlocal called
        called = True

    await shadows.add(the_task)("a", b="c")

    await worker.run_until_finished()

    assert called


async def test_supports_requesting_current_worker(
    shadows: Shadow, worker: Worker, now: Callable[[], datetime]
):
    """shadows should support providing the current worker to a task"""

    called = False

    async def the_task(a: str, b: str, this_worker: Worker = CurrentWorker()):
        assert a == "a"
        assert b == "c"
        assert this_worker is worker

        nonlocal called
        called = True

    await shadows.add(the_task)("a", b="c")

    await worker.run_until_finished()

    assert called


async def test_supports_requesting_current_execution(
    shadows: Shadow, worker: Worker, now: Callable[[], datetime]
):
    """shadows should support providing the current execution to a task"""

    called = False

    async def the_task(a: str, b: str, this_execution: Execution = CurrentExecution()):
        assert a == "a"
        assert b == "c"

        assert isinstance(this_execution, Execution)
        assert this_execution.key == "my-cool-task:123"

        nonlocal called
        called = True

    await shadows.add(the_task, key="my-cool-task:123")("a", b="c")

    await worker.run_until_finished()

    assert called


async def test_supports_requesting_current_task_key(
    shadows: Shadow, worker: Worker, now: Callable[[], datetime]
):
    """shadows should support providing the current task key to a task"""

    called = False

    async def the_task(a: str, b: str, this_key: str = TaskKey()):
        assert a == "a"
        assert b == "c"
        assert this_key == "my-cool-task:123"

        nonlocal called
        called = True

    await shadows.add(the_task, key="my-cool-task:123")("a", b="c")

    await worker.run_until_finished()

    assert called


async def test_all_shadowss_have_a_trace_task(
    shadows: Shadow, worker: Worker, caplog: pytest.LogCaptureFixture
):
    """All shadowss should have a trace task"""

    await shadows.add(tasks.trace)("Hello, world!")

    with caplog.at_level(logging.INFO):
        await worker.run_until_finished()

        assert "Hello, world!" in caplog.text


async def test_all_shadowss_have_a_fail_task(
    shadows: Shadow, worker: Worker, caplog: pytest.LogCaptureFixture
):
    """All shadowss should have a fail task"""

    await shadows.add(tasks.fail)("Hello, world!")

    with caplog.at_level(logging.ERROR):
        await worker.run_until_finished()

        assert "Hello, world!" in caplog.text


async def test_tasks_can_opt_into_argument_logging(
    shadows: Shadow, worker: Worker, caplog: pytest.LogCaptureFixture
):
    """Tasks can opt into argument logging for specific arguments"""

    async def the_task(
        a: Annotated[str, Logged],
        b: str,
        c: Annotated[str, Logged()] = "c",
        d: Annotated[str, "nah chief"] = "d",
        shadows: Shadow = CurrentShadow(),
    ):
        pass

    await shadows.add(the_task)("value-a", b="value-b", c="value-c", d="value-d")

    with caplog.at_level(logging.INFO):
        await worker.run_until_finished()

        assert "the_task('value-a', b=..., c='value-c', d=...)" in caplog.text
        assert "value-b" not in caplog.text
        assert "value-d" not in caplog.text


async def test_tasks_can_opt_into_logging_collection_lengths(
    shadows: Shadow, worker: Worker, caplog: pytest.LogCaptureFixture
):
    """Tasks can opt into logging the length of collections"""

    async def the_task(
        a: Annotated[list[str], Logged(length_only=True)],
        b: Annotated[dict[str, str], Logged(length_only=True)],
        c: Annotated[tuple[str, ...], Logged(length_only=True)],
        d: Annotated[set[str], Logged(length_only=True)],
        e: Annotated[int, Logged(length_only=True)],
        shadows: Shadow = CurrentShadow(),
    ):
        pass

    await shadows.add(the_task)(
        ["a", "b"], b={"d": "e", "f": "g"}, c=("h", "i"), d={"a", "b", "c"}, e=123
    )

    with caplog.at_level(logging.INFO):
        await worker.run_until_finished()

        assert (
            "the_task([len 2], b={len 2}, c=(len 2), d={len 3}, e=123)" in caplog.text
        )


async def test_logging_inside_of_task(
    shadows: Shadow,
    worker: Worker,
    now: Callable[[], datetime],
    caplog: pytest.LogCaptureFixture,
):
    """shadows should support providing a logger with task context"""
    called = False

    async def the_task(
        a: str, b: str, logger: LoggerAdapter[logging.Logger] = TaskLogger()
    ):
        assert a == "a"
        assert b == "c"

        logger.info("Task is running")

        nonlocal called
        called = True

    await shadows.add(the_task, key="my-cool-task:123")("a", b="c")

    with caplog.at_level(logging.INFO):
        await worker.run_until_finished()

    assert called
    assert "Task is running" in caplog.text
    assert "shadows.task.the_task" in caplog.text


async def test_self_perpetuating_immediate_tasks(
    shadows: Shadow, worker: Worker, now: Callable[[], datetime]
):
    """shadows should support self-perpetuating tasks"""

    calls: dict[str, list[int]] = {
        "first": [],
        "second": [],
    }

    async def the_task(start: int, iteration: int, key: str = TaskKey()):
        calls[key].append(start + iteration)
        if iteration < 3:
            await shadows.add(the_task, key=key)(start, iteration + 1)

    await shadows.add(the_task, key="first")(10, 1)
    await shadows.add(the_task, key="second")(20, 1)

    await worker.run_until_finished()

    assert calls["first"] == [11, 12, 13]
    assert calls["second"] == [21, 22, 23]


async def test_self_perpetuating_scheduled_tasks(
    shadows: Shadow, worker: Worker, now: Callable[[], datetime]
):
    """shadows should support self-perpetuating tasks"""

    calls: dict[str, list[int]] = {
        "first": [],
        "second": [],
    }

    async def the_task(start: int, iteration: int, key: str = TaskKey()):
        calls[key].append(start + iteration)
        if iteration < 3:
            soon = now() + timedelta(milliseconds=100)
            await shadows.add(the_task, key=key, when=soon)(start, iteration + 1)

    await shadows.add(the_task, key="first")(10, 1)
    await shadows.add(the_task, key="second")(20, 1)

    await worker.run_until_finished()

    assert calls["first"] == [11, 12, 13]
    assert calls["second"] == [21, 22, 23]


async def test_infinitely_self_perpetuating_tasks(
    shadows: Shadow, worker: Worker, now: Callable[[], datetime]
):
    """shadows should support testing use cases for infinitely self-perpetuating tasks"""

    calls: dict[str, list[int]] = {
        "first": [],
        "second": [],
        "unaffected": [],
    }

    async def the_task(start: int, iteration: int, key: str = TaskKey()):
        calls[key].append(start + iteration)
        soon = now() + timedelta(milliseconds=100)
        await shadows.add(the_task, key=key, when=soon)(start, iteration + 1)

    async def unaffected_task(start: int, iteration: int, key: str = TaskKey()):
        calls[key].append(start + iteration)
        if iteration < 3:
            await shadows.add(unaffected_task, key=key)(start, iteration + 1)

    await shadows.add(the_task, key="first")(10, 1)
    await shadows.add(the_task, key="second")(20, 1)
    await shadows.add(unaffected_task, key="unaffected")(30, 1)

    # Using worker.run_until_finished() would hang here because the task is always
    # queueing up a future run of itself.  With worker.run_at_most(),
    # we can specify tasks keys that will only be allowed to run a limited number of
    # times, thus allowing the worker to exist cleanly.
    await worker.run_at_most({"first": 4, "second": 2})

    assert calls["first"] == [11, 12, 13, 14]
    assert calls["second"] == [21, 22]
    assert calls["unaffected"] == [31, 32, 33]


async def test_striking_entire_tasks(
    shadows: Shadow, worker: Worker, the_task: AsyncMock, another_task: AsyncMock
):
    """shadows should support striking and restoring entire tasks"""

    await shadows.add(the_task)("a", b="c")
    await shadows.add(another_task)("d", e="f")

    await shadows.strike(the_task)

    await worker.run_until_finished()

    the_task.assert_not_called()
    the_task.reset_mock()

    another_task.assert_awaited_once_with("d", e="f")
    another_task.reset_mock()

    await shadows.restore(the_task)

    await shadows.add(the_task)("g", h="i")
    await shadows.add(another_task)("j", k="l")

    await worker.run_until_finished()

    the_task.assert_awaited_once_with("g", h="i")
    another_task.assert_awaited_once_with("j", k="l")


async def test_striking_entire_parameters(
    shadows: Shadow, worker: Worker, the_task: AsyncMock, another_task: AsyncMock
):
    """shadows should support striking and restoring entire parameters"""

    await shadows.add(the_task)(customer_id="123", order_id="456")
    await shadows.add(the_task)(customer_id="456", order_id="789")
    await shadows.add(the_task)(customer_id="789", order_id="012")
    await shadows.add(another_task)(customer_id="456", order_id="012")
    await shadows.add(another_task)(customer_id="789", order_id="456")

    await shadows.strike(None, "customer_id", "==", "789")

    await worker.run_until_finished()

    assert the_task.call_count == 2
    the_task.assert_has_awaits(
        [
            call(customer_id="123", order_id="456"),
            call(customer_id="456", order_id="789"),
            # customer_id == 789 is stricken
        ],
        any_order=True,
    )
    the_task.reset_mock()

    assert another_task.call_count == 1
    another_task.assert_has_awaits(
        [
            call(customer_id="456", order_id="012"),
            # customer_id == 789 is stricken
        ],
        any_order=True,
    )
    another_task.reset_mock()

    await shadows.add(the_task)(customer_id="123", order_id="456")
    await shadows.add(the_task)(customer_id="456", order_id="789")
    await shadows.add(the_task)(customer_id="789", order_id="012")
    await shadows.add(another_task)(customer_id="456", order_id="012")
    await shadows.add(another_task)(customer_id="789", order_id="456")

    await shadows.strike(None, "customer_id", "==", "123")

    await worker.run_until_finished()

    assert the_task.call_count == 1
    the_task.assert_has_awaits(
        [
            # customer_id == 123 is stricken
            call(customer_id="456", order_id="789"),
            # customer_id == 789 is stricken
        ],
        any_order=True,
    )
    the_task.reset_mock()

    assert another_task.call_count == 1
    another_task.assert_has_awaits(
        [
            call(customer_id="456", order_id="012"),
            # customer_id == 789 is stricken
        ],
        any_order=True,
    )
    another_task.reset_mock()

    await shadows.restore(None, "customer_id", "==", "123")

    await shadows.add(the_task)(customer_id="123", order_id="456")
    await shadows.add(the_task)(customer_id="456", order_id="789")
    await shadows.add(the_task)(customer_id="789", order_id="012")
    await shadows.add(another_task)(customer_id="456", order_id="012")
    await shadows.add(another_task)(customer_id="789", order_id="456")

    await worker.run_until_finished()

    assert the_task.call_count == 2
    the_task.assert_has_awaits(
        [
            call(customer_id="123", order_id="456"),
            call(customer_id="456", order_id="789"),
            # customer_id == 789 is still stricken
        ],
        any_order=True,
    )

    assert another_task.call_count == 1
    another_task.assert_has_awaits(
        [
            call(customer_id="456", order_id="012"),
            # customer_id == 789 is still stricken
        ],
        any_order=True,
    )


async def test_striking_tasks_for_specific_parameters(
    shadows: Shadow, worker: Worker, the_task: AsyncMock, another_task: AsyncMock
):
    """shadows should support striking and restoring tasks for specific parameters"""
    await shadows.add(the_task)("a", b=1)
    await shadows.add(the_task)("a", b=2)
    await shadows.add(the_task)("a", b=3)
    await shadows.add(another_task)("d", b=1)
    await shadows.add(another_task)("d", b=2)
    await shadows.add(another_task)("d", b=3)

    await shadows.strike(the_task, "b", "<=", 2)

    await worker.run_until_finished()

    assert the_task.call_count == 1
    the_task.assert_has_awaits(
        [
            # b <= 2 is stricken, so b=1 is out
            # b <= 2 is stricken, so b=2 is out
            call("a", b=3),
        ],
        any_order=True,
    )
    the_task.reset_mock()

    assert another_task.call_count == 3
    another_task.assert_has_awaits(
        [
            call("d", b=1),
            call("d", b=2),
            call("d", b=3),
        ],
        any_order=True,
    )
    another_task.reset_mock()

    await shadows.restore(the_task, "b", "<=", 2)

    await shadows.add(the_task)("a", b=1)
    await shadows.add(the_task)("a", b=2)
    await shadows.add(the_task)("a", b=3)
    await shadows.add(another_task)("d", b=1)
    await shadows.add(another_task)("d", b=2)
    await shadows.add(another_task)("d", b=3)

    await worker.run_until_finished()

    assert the_task.call_count == 3
    the_task.assert_has_awaits(
        [
            call("a", b=1),
            call("a", b=2),
            call("a", b=3),
        ],
        any_order=True,
    )

    assert another_task.call_count == 3
    another_task.assert_has_awaits(
        [
            call("d", b=1),
            call("d", b=2),
            call("d", b=3),
        ],
        any_order=True,
    )


async def test_adding_task_by_name_when_not_registered(shadows: Shadow):
    """shadows should raise an error when attempting to add a task by name that isn't registered"""

    with pytest.raises(KeyError, match="unregistered_task"):
        await shadows.add("unregistered_task")()


async def test_adding_task_with_unbindable_arguments(
    shadows: Shadow,
    worker: Worker,
    caplog: pytest.LogCaptureFixture,
):
    """Should not raise an error when a task is scheduled or executed with
    incorrect arguments."""

    async def task_with_specific_args(a: str, b: int, c: bool = False) -> None:
        pass  # pragma: no cover

    await shadows.add(task_with_specific_args)("a", 2, d="unexpected")  # type: ignore[arg-type]

    with caplog.at_level(logging.ERROR):
        await worker.run_until_finished()

    assert "got an unexpected keyword argument 'd'" in caplog.text


async def test_perpetual_tasks(shadows: Shadow, worker: Worker):
    """Perpetual tasks should reschedule themselves forever"""

    calls = 0

    async def perpetual_task(
        a: str,
        b: int,
        perpetual: Perpetual = Perpetual(every=timedelta(milliseconds=50)),
    ):
        assert a == "a"
        assert b == 2

        assert isinstance(perpetual, Perpetual)

        assert perpetual.every == timedelta(milliseconds=50)

        nonlocal calls
        calls += 1

    execution = await shadows.add(perpetual_task)(a="a", b=2)

    await worker.run_at_most({execution.key: 3})

    assert calls == 3


async def test_perpetual_tasks_can_cancel_themselves(shadows: Shadow, worker: Worker):
    """A perpetual task can request its own cancellation"""
    calls = 0

    async def perpetual_task(
        a: str,
        b: int,
        perpetual: Perpetual = Perpetual(every=timedelta(milliseconds=50)),
    ):
        assert a == "a"
        assert b == 2

        assert isinstance(perpetual, Perpetual)

        assert perpetual.every == timedelta(milliseconds=50)

        nonlocal calls
        calls += 1

        if calls == 3:
            perpetual.cancel()

    await shadows.add(perpetual_task)(a="a", b=2)

    await worker.run_until_finished()

    assert calls == 3


async def test_perpetual_tasks_can_change_their_parameters(
    shadows: Shadow, worker: Worker
):
    """Perpetual tasks may change their parameters each time"""
    arguments: list[tuple[str, int]] = []

    async def perpetual_task(
        a: str,
        b: int,
        perpetual: Perpetual = Perpetual(every=timedelta(milliseconds=50)),
    ):
        arguments.append((a, b))
        perpetual.perpetuate(a + "a", b=b + 1)

    execution = await shadows.add(perpetual_task)(a="a", b=1)

    await worker.run_at_most({execution.key: 3})

    assert len(arguments) == 3
    assert arguments == [("a", 1), ("aa", 2), ("aaa", 3)]


async def test_perpetual_tasks_perpetuate_even_after_errors(
    shadows: Shadow, worker: Worker
):
    """Perpetual tasks may change their parameters each time"""
    calls = 0

    async def perpetual_task(
        a: str,
        b: int,
        perpetual: Perpetual = Perpetual(every=timedelta(milliseconds=50)),
    ):
        nonlocal calls
        calls += 1

        raise ValueError("woops!")

    execution = await shadows.add(perpetual_task)(a="a", b=1)

    await worker.run_at_most({execution.key: 3})

    assert calls == 3


async def test_perpetual_tasks_can_be_automatically_scheduled(
    shadows: Shadow, worker: Worker
):
    """Perpetual tasks can be automatically scheduled"""

    calls = 0

    async def my_automatic_task(
        perpetual: Perpetual = Perpetual(
            every=timedelta(milliseconds=50), automatic=True
        ),
    ):
        assert isinstance(perpetual, Perpetual)

        assert perpetual.every == timedelta(milliseconds=50)

        nonlocal calls
        calls += 1

    # Note we never add this task to the shadows, we just register it.
    shadows.register(my_automatic_task)

    # The automatic key will be the task function's name
    await worker.run_at_most({"my_automatic_task": 3})

    assert calls == 3


async def test_simple_timeout(shadows: Shadow, worker: Worker):
    """A task can be scheduled with a timeout"""

    called = False

    async def task_with_timeout(
        timeout: Timeout = Timeout(timedelta(milliseconds=100)),
    ):
        await asyncio.sleep(0.01)

        nonlocal called
        called = True

    await shadows.add(task_with_timeout)()

    start = datetime.now(timezone.utc)

    await worker.run_until_finished()

    elapsed = datetime.now(timezone.utc) - start

    assert called
    assert elapsed <= timedelta(milliseconds=150)


async def test_simple_timeout_cancels_tasks(shadows: Shadow, worker: Worker):
    """A task can be scheduled with a timeout and are cancelled"""

    called = False

    async def task_with_timeout(
        timeout: Timeout = Timeout(timedelta(milliseconds=100)),
    ):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            nonlocal called
            called = True

    await shadows.add(task_with_timeout)()

    start = datetime.now(timezone.utc)

    await worker.run_until_finished()

    elapsed = datetime.now(timezone.utc) - start

    assert called
    assert timedelta(milliseconds=100) <= elapsed <= timedelta(milliseconds=200)


async def test_timeout_can_be_extended(shadows: Shadow, worker: Worker):
    """A task can be scheduled with a timeout and extend themselves"""

    called = False

    async def task_with_timeout(
        timeout: Timeout = Timeout(timedelta(milliseconds=100)),
    ):
        await asyncio.sleep(0.05)

        timeout.extend(timedelta(milliseconds=200))

        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            nonlocal called
            called = True

    await shadows.add(task_with_timeout)()

    start = datetime.now(timezone.utc)

    await worker.run_until_finished()

    elapsed = datetime.now(timezone.utc) - start

    assert called
    assert timedelta(milliseconds=250) <= elapsed <= timedelta(milliseconds=400)


async def test_timeout_extends_by_base_by_default(shadows: Shadow, worker: Worker):
    """A task can be scheduled with a timeout and extend itself by the base timeout"""

    called = False

    async def task_with_timeout(
        timeout: Timeout = Timeout(timedelta(milliseconds=100)),
    ):
        await asyncio.sleep(0.05)

        timeout.extend()  # defaults to the base timeout

        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            nonlocal called
            called = True

    await shadows.add(task_with_timeout)()

    start = datetime.now(timezone.utc)

    await worker.run_until_finished()

    elapsed = datetime.now(timezone.utc) - start

    assert called
    assert timedelta(milliseconds=150) <= elapsed <= timedelta(milliseconds=300)


async def test_timeout_is_compatible_with_retry(shadows: Shadow, worker: Worker):
    """A task that times out can be retried"""

    successes: list[int] = []

    async def task_with_timeout(
        retry: Retry = Retry(attempts=3),
        timeout: Timeout = Timeout(timedelta(milliseconds=100)),
    ):
        if retry.attempt == 1:
            await asyncio.sleep(1)

        successes.append(retry.attempt)

    await shadows.add(task_with_timeout)()

    await worker.run_until_finished()

    assert successes == [2]


async def test_simple_function_dependencies(shadows: Shadow, worker: Worker):
    """A task can depend on the return value of simple functions"""

    async def dependency_one() -> str:
        return f"one-{uuid4()}"

    async def dependency_two() -> str:
        return f"two-{uuid4()}"

    called = 0

    async def dependent_task(
        one_a: str = Depends(dependency_one),
        one_b: str = Depends(dependency_one),
        two: str = Depends(dependency_two),
    ):
        assert one_a.startswith("one-")
        assert one_b == one_a

        assert two.startswith("two-")

        nonlocal called
        called += 1

    await shadows.add(dependent_task)()

    await worker.run_until_finished()

    assert called == 1


async def test_contextual_dependencies(shadows: Shadow, worker: Worker):
    """A task can depend on the return value of async context managers"""

    stages: list[str] = []

    @asynccontextmanager
    async def dependency_one() -> AsyncGenerator[str, None]:
        stages.append("one-before")
        yield f"one-{uuid4()}"
        stages.append("one-after")

    async def dependency_two() -> str:
        return f"two-{uuid4()}"

    called = 0

    async def dependent_task(
        one_a: str = Depends(dependency_one),
        one_b: str = Depends(dependency_one),
        two: str = Depends(dependency_two),
    ):
        assert one_a.startswith("one-")
        assert one_b == one_a

        assert two.startswith("two-")

        nonlocal called
        called += 1

    await shadows.add(dependent_task)()

    await worker.run_until_finished()

    assert called == 1
    assert stages == ["one-before", "one-after"]


async def test_dependencies_of_dependencies(shadows: Shadow, worker: Worker):
    """A task dependency can depend on other dependencies"""
    counter = 0

    async def dependency_one() -> list[str]:
        nonlocal counter
        counter += 1
        return [f"one-{counter}"]

    async def dependency_two(my_one: list[str] = Depends(dependency_one)) -> list[str]:
        nonlocal counter
        counter += 1
        return my_one + [f"two-{counter}"]

    async def dependency_three(
        my_one: list[str] = Depends(dependency_one),
        my_two: list[str] = Depends(dependency_two),
    ) -> list[str]:
        nonlocal counter
        counter += 1
        return my_one + my_two + [f"three-{counter}"]

    async def dependent_task(
        one_a: list[str] = Depends(dependency_one),
        one_b: list[str] = Depends(dependency_one),
        two: list[str] = Depends(dependency_two),
        three: list[str] = Depends(dependency_three),
    ):
        assert one_a is one_b

        assert one_a == ["one-1"]
        assert two == ["one-1", "two-2"]
        assert three == ["one-1", "two-2", "three-3"]

    await shadows.add(dependent_task)()

    await worker.run_until_finished()


async def test_dependencies_can_ask_for_shadows_dependencies(
    shadows: Shadow, worker: Worker
):
    """A task dependency can ask for a shadows dependency"""

    called = 0

    async def dependency_one(this_shadows: Shadow = CurrentShadow()) -> str:
        assert this_shadows is shadows

        nonlocal called
        called += 1

        return f"one-{called}"

    async def dependency_two(
        this_worker: Worker = CurrentWorker(),
        one: str = Depends(dependency_one),
    ) -> str:
        assert this_worker is worker

        assert one == "one-1"

        nonlocal called
        called += 1

        return f"two-{called}"

    async def dependent_task(
        one: str = Depends(dependency_one),
        two: str = Depends(dependency_two),
        this_shadows: Shadow = CurrentShadow(),
        this_worker: Worker = CurrentWorker(),
    ):
        assert one == "one-1"
        assert two == "two-2"

        assert this_shadows is shadows
        assert this_worker is worker

        nonlocal called
        called += 1

    await shadows.add(dependent_task)()

    await worker.run_until_finished()


async def test_dependency_failures_are_task_failures(
    shadows: Shadow, worker: Worker, caplog: pytest.LogCaptureFixture
):
    """A task dependency failure will cause the task to fail"""

    called: bool = False

    async def dependency_one() -> str:
        raise ValueError("this one is bad")

    async def dependency_two() -> str:
        raise ValueError("and so is this one")

    async def dependent_task(
        a: str = Depends(dependency_one),
        b: str = Depends(dependency_two),
    ) -> None:
        nonlocal called
        called = True  # pragma: no cover

    await shadows.add(dependent_task)()

    with caplog.at_level(logging.ERROR):
        await worker.run_until_finished()

    assert not called

    assert "Failed to resolve dependencies for parameter(s): a, b" in caplog.text
    assert "ValueError: this one is bad" in caplog.text
    assert "ValueError: and so is this one" in caplog.text


async def test_contextual_dependency_before_failures_are_task_failures(
    shadows: Shadow, worker: Worker, caplog: pytest.LogCaptureFixture
):
    """A contextual task dependency failure will cause the task to fail"""

    called: int = 0

    @asynccontextmanager
    async def dependency_before() -> AsyncGenerator[str, None]:
        raise ValueError("this one is bad")
        yield "this won't be used"  # pragma: no cover

    async def dependent_task(
        a: str = Depends(dependency_before),
    ) -> None:
        nonlocal called
        called += 1  # pragma: no cover

    await shadows.add(dependent_task)()

    with caplog.at_level(logging.ERROR):
        await worker.run_until_finished()

    assert not called

    assert "Failed to resolve dependencies for parameter(s): a" in caplog.text
    assert "ValueError: this one is bad" in caplog.text


async def test_contextual_dependency_after_failures_are_task_failures(
    shadows: Shadow, worker: Worker, caplog: pytest.LogCaptureFixture
):
    """A contextual task dependency failure will cause the task to fail"""

    called: int = 0

    @asynccontextmanager
    async def dependency_after() -> AsyncGenerator[str, None]:
        yield "this will be used"
        raise ValueError("this one is bad")

    async def dependent_task(
        a: str = Depends(dependency_after),
    ) -> None:
        assert a == "this will be used"

        nonlocal called
        called += 1

    await shadows.add(dependent_task)()

    with caplog.at_level(logging.ERROR):
        await worker.run_until_finished()

    assert called == 1

    assert "ValueError: this one is bad" in caplog.text


async def test_dependencies_can_ask_for_task_arguments(shadows: Shadow, worker: Worker):
    """A task dependency can ask for a task argument"""

    called = 0

    async def dependency_one(a: list[str] = TaskArgument()) -> list[str]:
        return a

    async def dependency_two(another_name: list[str] = TaskArgument("a")) -> list[str]:
        return another_name

    async def dependent_task(
        a: list[str],
        b: list[str] = TaskArgument("a"),
        c: list[str] = Depends(dependency_one),
        d: list[str] = Depends(dependency_two),
    ) -> None:
        assert a is b
        assert a is c
        assert a is d

        nonlocal called
        called += 1

    await shadows.add(dependent_task)(a=["hello", "world"])

    await worker.run_until_finished()

    assert called == 1


async def test_task_arguments_may_be_optional(shadows: Shadow, worker: Worker):
    """A task dependency can ask for a task argument optionally"""

    called = 0

    async def dependency_one(
        a: list[str] | None = TaskArgument(optional=True),
    ) -> list[str] | None:
        return a

    async def dependent_task(
        not_a: list[str],
        b: list[str] | None = Depends(dependency_one),
    ) -> None:
        assert not_a == ["hello", "world"]
        assert b is None

        nonlocal called
        called += 1

    await shadows.add(dependent_task)(not_a=["hello", "world"])

    await worker.run_until_finished()

    assert called == 1
