shadows represents a sick distributed background task processing framework designed specifically for python applications, emphasizing seamless scheduling capabilities for both immediate and future computational workloads with exceptional efficiency.

## installing shadows

shadow is [available on PyPI](https://pypi.org/project/shadow-task/) under the package name
`shadow-task`. it targets python 3.12 or above.

with [`uv`](https://docs.astral.sh/uv/):

```bash
uv pip install shadow-task

or

uv add shadow-task
```

with `pip`:

```bash
pip install shadow-task
```

shadow requires a [redis](http://redis.io/) server with streams support (introduced in redis 5.0.0). shadow is thoroughly tested with redis 6 and 7, ensuring compatibility and reliability.

## at a glance

```python
from datetime import datetime, timedelta, timezone

from shadows import Shadow


async def greet(name: str, greeting="Hello") -> None:
    print(f"{greeting}, {name} at {datetime.now()}!")


async with Shadow() as shadows:
    await shadows.add(greet)("rohit")

    now = datetime.now(timezone.utc)
    soon = now + timedelta(seconds=3)
    await shadows.add(greet, when=soon)("rahul", greeting="what'sgood")
```

```python
from shadows import Shadow, Worker

async with Shadow() as shadows:
    async with Worker(shadows) as worker:
        await worker.run_until_finished()
```

```
Hello, rohit at 2025-03-05 13:58:21.552644!
what'sgood, rahul at 2025-03-05 13:58:24.550773!
```

## how shadows works

shadows operates on a redis-based architecture that provides reliable distributed task processing with at-least-once delivery semantics. the system uses two primary redis data structures:

- **redis streams**: handle immediate task execution with consumer groups ensuring each task is processed by exactly one worker
- **redis sorted sets**: manage scheduled tasks with execution timestamps, automatically moved to streams when ready

workers continuously poll the redis stream for available tasks and move scheduled tasks from the sorted set to the stream when their execution time arrives. this movement is performed atomically using lua scripts to ensure consistency.

## advanced examples

### retry functionality with exponential backoff

```python
from shadows import Retry, ExponentialRetry

async def flaky_api_call(
    url: str,
    retry: ExponentialRetry = ExponentialRetry(
        attempts=5,
        minimum_delay=timedelta(seconds=1),
        maximum_delay=timedelta(minutes=5)
    )
) -> None:
    # retries with delays: 1s, 2s, 4s, 8s, 16s (capped at 5 minutes)
    response = await http_client.get(url)
    response.raise_for_status()
    print(f"api call succeeded on attempt {retry.attempt}")
```

### perpetual tasks with self-cancellation

```python
from shadows import Perpetual

async def monitor_deployment(
    deployment_id: str,
    perpetual: Perpetual = Perpetual(every=timedelta(seconds=30))
) -> None:
    status = await check_deployment_status(deployment_id)

    if status in ["completed", "failed"]:
        await notify_deployment_finished(deployment_id, status)
        perpetual.cancel()  # stop monitoring this deployment
        return

    print(f"deployment {deployment_id} status: {status}")
```

### task chaining with dependency injection

```python
from shadows import CurrentShadow

async def process_user_data(
    user_id: int,
    shadows: Shadow = CurrentShadow()
) -> None:
    print(f"processing user {user_id}")
    
    # simulate some work
    await asyncio.sleep(0.1)
    
    # schedule follow-up tasks
    await shadows.add(send_notification)(user_id, "processing completed")
    await shadows.add(update_analytics)(user_id)
```

### idempotent task scheduling with custom keys

```python
async def process_order(order_id: int) -> None:
    print(f"processing order {order_id}")

async with Shadow() as shadows:
    # only one task per order_id can be scheduled
    key = f"process-order-{order_id}"
    await shadows.add(process_order, key=key)(order_id)
    
    # duplicate scheduling is ignored
    await shadows.add(process_order, key=key)(order_id)  # ignored
```

## command line interface

shadows provides a comprehensive cli for managing tasks and workers:

### basic cli operations

```bash
# start a worker with custom tasks
shadows worker --tasks myapp.tasks:all_tasks --concurrency 5

# view current shadows state
shadows snapshot --shadows my-shadows

# clear all pending tasks
shadows clear --shadows my-shadows

# add built-in trace tasks for debugging
shadows tasks trace "system startup completed"

# list active workers
shadows workers ls --shadows my-shadows
```

### production worker configuration

```bash
shadows worker \
  --shadows orders \
  --url redis://redis.prod.com:6379/0 \
  --name orders-worker-1 \
  --concurrency 50 \
  --redelivery-timeout 10m \
  --healthcheck-port 8080 \
  --metrics-port 9090 \
  --logging-format json \
  --tasks myapp.tasks:production_tasks
```

### testing and development

```bash
# run tests with fast polling for quick feedback
shadows worker \
  --concurrency 10 \
  --minimum-check-interval 50ms \
  --scheduling-resolution 100ms \
  --tasks tests.tasks:test_tasks
```

## testing with shadows

shadows includes powerful testing utilities that make it easy to test complex distributed workflows:

### pytest integration

```python
import pytest
from shadows import Shadow, Worker

@pytest.fixture
async def test_shadows():
    async with Shadow(name=f"test-{uuid4()}") as shadows:
        yield shadows
        await shadows.clear()

async def test_order_processing(test_shadows: Shadow):
    test_shadows.register(process_order)
    test_shadows.register(send_confirmation)
    
    await test_shadows.add(process_order)(order_id=123)
    
    async with Worker(test_shadows) as worker:
        await worker.run_until_finished()
    
    assert order_is_processed(123)
    assert confirmation_was_sent(123)
```

### controlling perpetual tasks

```python
async def test_perpetual_monitoring(test_shadows: Shadow):
    test_shadows.register(health_check_service)
    
    await test_shadows.add(health_check_service)("https://api.example.com")
    
    async with Worker(test_shadows) as worker:
        # let health check run 3 times, then stop
        await worker.run_at_most({"health_check_service": 3})
    
    assert health_check_call_count == 3
```

## why shadows?

lightning-fast one-way background task processing devoid of unnecessary complexity

seamless scheduling of immediate or future workloads through a unified interface

bypass problematic tasks or parameters without requiring code redeployment

purpose-built architecture optimized for redis streams

comprehensive type safety and type awareness for all background task functions

sophisticated dependency injection system similar to fastapi, typer, and fastmcp for reusable resources


## hacking on shadows

we use [`uv`](https://docs.astral.sh/uv/) for project management, so getting set up
should be as simple as cloning the repo and running:

```bash
uv sync
```

then to run the test suite:

```bash
pytest
```

we aim to maintain 100% test coverage, which is required for all prs to `shadows`. we
believe that `shadows` should stay small, simple, understandable, and reliable, and that
begins with testing all the dusty branches and corners. and thanks dockettho. this will give us the
confidence to upgrade dependencies quickly and to adapt to new versions of redis over
time.
