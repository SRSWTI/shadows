# testing with shadow

shadow includes the utilities you need to test all your background task systems in realistic ways. the ergonomic design supports testing complex workflows with minimal setup.

## testing tasks as simple functions

often you can test your tasks without running a worker at all! shadow tasks are just python functions, so you can call them directly and pass test values for dependency parameters:

```python
from shadows import Shadow, CurrentShadow, Retry
from unittest.mock import AsyncMock

async def process_order(
    order_id: int,
    shadows: Shadow = CurrentShadow(),
    retry: Retry = Retry(attempts=3)
) -> None:
    # Your task logic here
    order = await fetch_order(order_id)
    await charge_payment(order)
    await shadows.add(send_confirmation)(order_id)

async def test_process_order_logic() -> None:
    """Test the task logic without running a worker."""
    mock_shadows = AsyncMock()

    # Call the task directly with test parameters
    await process_order(
        order_id=123,
        shadows=mock_shadows,
        retry=Retry(attempts=1)
    )

    # Verify the task scheduled follow-up work
    mock_shadows.add.assert_called_once()
```

this approach is great for testing business logic quickly without the overhead of setting up shadowss and workers.

## testing with pytest fixtures

the most powerful way to test with shadow is using pytest fixtures to set up your shadows and worker. this approach, used throughout shadow's own test suite, provides clean isolation and reusable test infrastructure.

### basic fixture setup

create fixtures for your test shadows and worker:

```python
import pytest
from datetime import datetime, timedelta
from typing import AsyncGenerator, Callable
from uuid import uuid4
from unittest.mock import AsyncMock
from shadows import Shadow, Worker

@pytest.fixture
async def test_shadows() -> AsyncGenerator[Shadow, None]:
    """Create a test shadows with a unique name for each test."""
    async with Shadow(
        name=f"test-{uuid4()}",
        url="redis://localhost:6379/0"
    ) as shadows:
        yield shadows

@pytest.fixture
async def test_worker(test_shadows: Shadow) -> AsyncGenerator[Worker, None]:
    """Create a test worker with fast polling for quick tests."""
    async with Worker(
        test_shadows,
        minimum_check_interval=timedelta(milliseconds=5),
        scheduling_resolution=timedelta(milliseconds=5)
    ) as worker:
        yield worker
```

### using fixtures in tests

with these fixtures, your tests become much cleaner:

```python
async def send_notification(user_id: int, message: str) -> None:
    """Example task for testing."""
    print(f"Sending '{message}' to user {user_id}")

async def test_task_execution(test_shadows: Shadow, test_worker: Worker) -> None:
    """Test that tasks execute with correct arguments."""
    test_shadows.register(send_notification)
    await test_shadows.add(send_notification)(123, "Welcome!")

    await test_worker.run_until_finished()

    # Verify by checking side effects or using test doubles

async def test_idempotent_scheduling(test_shadows: Shadow, test_worker: Worker) -> None:
    """Test that tasks with same key don't duplicate."""
    test_shadows.register(send_notification)
    key = "unique-notification"

    # Schedule same task multiple times with same key
    await test_shadows.add(send_notification, key=key)(123, "message1")
    await test_shadows.add(send_notification, key=key)(123, "message2")  # Should replace
    await test_shadows.add(send_notification, key=key)(123, "message3")  # Should replace

    # Verify only one task is scheduled
    snapshot = await test_shadows.snapshot()
    assert len([t for t in snapshot.future if t.key == key]) == 1
```

## running until finished

for tests and batch processing, use [`run_until_finished()`](api-reference.md#shadows.Worker.run_until_finished) to process all pending tasks then stop:

```python
async def test_order_processing(test_shadows: Shadow, test_worker: Worker) -> None:
    """Test order processing workflow."""
    test_shadows.register(process_order)
    test_shadows.register(send_confirmation)
    test_shadows.register(update_inventory)

    # Schedule some work
    await test_shadows.add(process_order)(order_id=123)
    await test_shadows.add(send_confirmation)(order_id=123)
    await test_shadows.add(update_inventory)(product_id=456)

    # Process all pending tasks
    await test_worker.run_until_finished()

    # Now verify results
    assert order_is_processed(123)
    assert confirmation_was_sent(123)
    assert inventory_was_updated(456)
```

this works well for testing workflows where you need to ensure all tasks complete before making assertions.

### testing task registration

test that tasks are properly registered and can be called by name:

```python
async def test_task_registration_by_name(test_shadows: Shadow, test_worker: Worker) -> None:
    """Test executing tasks by string name."""
    async def example_task(data: str) -> None:
        print(f"Processing: {data}")

    test_shadows.register(example_task)

    # Execute by name instead of function reference
    await test_shadows.add("example_task")("test data")

    await test_worker.run_until_finished()

    # Verify by checking side effects or logs
```

## controlling perpetual tasks

use [`run_at_most()`](api-reference.md#shadows.Worker.run_at_most) to limit how many times specific tasks run, which is essential for testing perpetual tasks:

```python
async def test_perpetual_monitoring(test_shadows: Shadow, test_worker: Worker) -> None:
    """Test perpetual task monitoring."""
    test_shadows.register(health_check_service)
    test_shadows.register(process_data)
    test_shadows.register(send_reports)

    # This would normally run forever
    await test_shadows.add(health_check_service)("https://api.example.com")

    # Also schedule some regular tasks
    await test_shadows.add(process_data)(dataset="test")
    await test_shadows.add(send_reports)()

    # Let health check run 3 times, everything else runs to completion
    await test_worker.run_at_most({"health_check_service": 3})

    # Verify the health check ran the expected number of times
    assert health_check_call_count == 3
```

the [`run_at_most()`](api-reference.md#shadows.Worker.run_at_most) method takes a dictionary mapping task names to maximum execution counts. tasks not in the dictionary run to completion as normal.

## testing self-perpetuating chains

for tasks that create chains of future work, you can control the chain length:

```python
async def test_batch_processing_chain(test_shadows: Shadow, test_worker: Worker) -> None:
    """Test batch processing chains."""
    test_shadows.register(process_batch)

    # This creates a chain: batch 1 -> batch 2 -> batch 3
    await test_shadows.add(process_batch, key="batch-job")(batch_id=1, total_batches=3)

    # Let this specific key run 3 times (for 3 batches)
    await test_worker.run_at_most({"batch-job": 3})

    # Verify all batches were processed
    assert all_batches_processed([1, 2, 3])
```

you can use task keys in [`run_at_most()`](api-reference.md#shadows.Worker.run_at_most) to control specific task instances rather than all tasks of a given type.

## testing task scheduling

test that tasks are scheduled correctly without running them:

```python
from datetime import datetime, timedelta, timezone

async def test_scheduling_logic(test_shadows: Shadow) -> None:
    """Test task scheduling without execution."""
    test_shadows.register(send_reminder)

    # Schedule some tasks
    future_time = datetime.now(timezone.utc) + timedelta(hours=1)
    await test_shadows.add(send_reminder, when=future_time, key="reminder-123")(
        customer_id=123,
        message="Your subscription expires soon"
    )

    # Check that task was scheduled (but not executed)
    snapshot = await test_shadows.snapshot()

    assert len(snapshot.future) == 1
    assert len(snapshot.running) == 0
    assert snapshot.future[0].key == "reminder-123"
    assert snapshot.future[0].function.__name__ == "send_reminder"
```

## integration testing with real redis

for integration tests, use a real redis instance but with a test-specific shadows name:

```python
import pytest
from typing import AsyncGenerator
from uuid import uuid4
from shadows import Shadow, Worker
from redis.asyncio import Redis

@pytest.fixture
async def test_shadows() -> AsyncGenerator[Shadow, None]:
    # Use a unique shadows name for each test
    test_name = f"test-{uuid4()}"

    async with Shadow(name=test_name, url="redis://localhost:6379/1") as shadows:
        yield shadows

        # Clean up after test
        await shadows.clear()

async def test_full_workflow(test_shadows: Shadow) -> None:
    test_shadows.register(process_order)
    test_shadows.register(send_confirmation)

    await test_shadows.add(process_order)(order_id=123)

    async with Worker(test_shadows) as worker:
        await worker.run_until_finished()

    # Verify against real external systems
    assert order_exists_in_database(123)
    assert email_was_sent_to_customer(123)
```

## testing guidelines

### use descriptive task keys

use meaningful task keys in tests to make debugging easier:

```python
from uuid import uuid4

# Good: Clear what this task represents
await test_shadows.add(process_order, key=f"test-order-{order_id}")(order_id)

# Less clear: Generic key doesn't help with debugging
await test_shadows.add(process_order, key=f"task-{uuid4()}")(order_id)
```

### test error scenarios

always test what happens when tasks fail:

```python
from unittest import mock
async def test_order_processing_failure(test_shadows: Shadow, test_worker: Worker) -> None:
    """Test error handling in order processing."""
    test_shadows.register(process_order)

    # Simulate a failing external service
    with mock.patch('external_service.process_payment', side_effect=PaymentError):
        await test_shadows.add(process_order)(order_id=123)

        await test_worker.run_until_finished()

        # Verify error handling
        assert order_status(123) == "payment_failed"
        assert error_notification_sent()
```

### test idempotency

verify that tasks with the same key don't create duplicate work:

```python
async def test_idempotent_scheduling(test_shadows: Shadow) -> None:
    """Test idempotent task scheduling."""
    test_shadows.register(process_order)
    key = "process-order-123"

    # Schedule the same task multiple times
    await test_shadows.add(process_order, key=key)(order_id=123)
    await test_shadows.add(process_order, key=key)(order_id=123)
    await test_shadows.add(process_order, key=key)(order_id=123)

    snapshot = await test_shadows.snapshot()

    # Should only have one task scheduled
    assert len(snapshot.future) == 1
    assert snapshot.future[0].key == key
```

### test timing-sensitive logic

for tasks that depend on timing, use controlled time in tests:

```python
from datetime import datetime, timedelta, timezone
from unittest import mock

async def test_scheduled_task_timing(test_shadows: Shadow, test_worker: Worker) -> None:
    """Test timing-sensitive task scheduling."""
    test_shadows.register(send_reminder)
    now = datetime.now(timezone.utc)
    future_time = now + timedelta(seconds=10)

    await test_shadows.add(send_reminder, when=future_time)(customer_id=123)

    # Task should not run immediately
    await test_worker.run_until_finished()

    assert not reminder_was_sent(123)

    # Fast-forward time and test again
    with mock.patch('shadows.datetime') as mock_datetime:
        mock_datetime.now.return_value = future_time + timedelta(seconds=1)

        await test_worker.run_until_finished()

        assert reminder_was_sent(123)
```

shadow's testing utilities make it straightforward to write comprehensive tests for even complex distributed task workflows. the key is using [`run_until_finished()`](api-reference.md#shadows.Worker.run_until_finished) for deterministic execution and [`run_at_most()`](api-reference.md#shadows.Worker.run_at_most) for controlling perpetual or self-scheduling tasks.
