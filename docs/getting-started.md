# getting started

## installation

shadow is [available on PyPI](https://pypi.org/project/shadows/) under the package name
`shadow-task`. it targets python 3.12 or above.

with [`uv`](https://docs.astral.sh/uv/):

```bash
uv pip install shadow-task

# or

uv add shadows
```

with `pip`:

```bash
pip install shadow-task
```

you'll also need a [redis](http://redis.io/) server with streams support (redis 5.0+). shadow is tested with redis 6 and 7, and also works with [valkey](https://valkey.io/).

## your first shadow

each `shadow` should have a name that will be shared across your system, like the name
of a topic or queue. by default this is `"shadows"`. you can run multiple separate
shadowss on a single redis server as long as they have different names.

```python
from datetime import datetime, timedelta, timezone
from shadows import Shadow

async def send_welcome_email(customer_id: int, name: str) -> None:
    print(f"Welcome, {name}! (customer {customer_id})")

async with Shadow(name="emails", url="redis://localhost:6379/0") as shadows:
    # Schedule immediate work
    await shadows.add(send_welcome_email)(12345, "Alice")

    # Schedule future work
    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    await shadows.add(send_welcome_email, when=tomorrow)(67890, "Bob")
```

the `name` and `url` together represent a single shared shadows of work across all your
system. different services can schedule work on the same shadows as long as they use the same connection details.

## understanding task keys

every task gets a unique identifier called a `key`. by default, shadow generates random uuids for these keys, which works well for most use cases since each task execution is truly independent work.

```python
async def send_notification(user_id: int, message: str) -> None:
    print(f"Sending to user {user_id}: {message}")

async with Shadow() as shadows:
    # Each of these gets a random key and will execute independently
    await shadows.add(send_notification)(123, "Welcome!")
    await shadows.add(send_notification)(456, "Your order shipped")
    await shadows.add(send_notification)(123, "Thank you for your purchase")
```

sometimes though, you want to control whether multiple tasks represent the same logical work. for example, if a user clicks "process my order" multiple times, you probably only want to process that order once.

custom keys make scheduling idempotent. there can only ever be one future task scheduled with a given key:

```python
async def process_order(order_id: int) -> None:
    print(f"Processing order {order_id}")

async with Shadow() as shadows:
    key = f"process-order-{12345}"
    await shadows.add(process_order, key=key)(12345)
    await shadows.add(process_order, key=key)(12345)  # Ignored - key already exists
```

this is especially valuable for web apis where client retries or network issues might cause the same request to arrive multiple times:

```python
@app.post("/orders/{order_id}/process")
async def api_process_order(order_id: int):
    # Even if this endpoint gets called multiple times, only one task is scheduled
    key = f"process-order-{order_id}"
    await shadows.add(process_order, key=key)(order_id)
    return {"status": "scheduled"}
```

custom keys also let you manage scheduled work. you can replace future tasks to change their timing or arguments, or cancel them entirely:

```python
key = f"reminder-{customer_id}"

# Schedule a reminder for next week
next_week = datetime.now(timezone.utc) + timedelta(days=7)
await shadows.add(send_reminder, when=next_week, key=key)(
    customer_id, "Your trial expires soon"
)

# Customer upgrades - move reminder to next month instead
next_month = datetime.now(timezone.utc) + timedelta(days=30)
await shadows.replace(send_reminder, when=next_month, key=key)(
    customer_id, "Thanks for upgrading!"
)

# Customer cancels - remove reminder entirely
await shadows.cancel(key)
```

note that canceling only works for tasks scheduled in the future. tasks that are ready for immediate execution cannot be canceled once they've been added to the processing queue.

## running tasks: workers

tasks don't execute automatically - you need workers to process them. a worker connects to the same shadows and continuously pulls tasks from the queue.

```python
from shadows import Shadow, Worker

async def process_order(order_id: int) -> None:
    print(f"Processing order {order_id}")

async def send_notification(message: str) -> None:
    print(f"Notification: {message}")

async with Shadow() as shadows:
    # Register tasks so workers know about them
    shadows.register(process_order)
    shadows.register(send_notification)

    async with Worker(shadows) as worker:
        await worker.run_forever()  # Process tasks until interrupted
```

for production deployments, you'll typically run workers via the cli:

```bash
# In tasks.py
async def process_order(order_id: int) -> None:
    print(f"Processing order {order_id}")

async def send_notification(message: str) -> None:
    print(f"Notification: {message}")

tasks = [process_order, send_notification]
```

```bash
shadows worker --tasks tasks:tasks --concurrency 5
```

workers automatically handle concurrency (processing multiple tasks simultaneously), retries on failure, and graceful shutdown. by default, a worker processes up to 10 tasks concurrently.

## basic error handling

by default, if a task fails (raises an exception), shadow will log the error and mark the task as failed in its opentelemetry traces. the task won't be retried and the worker will move on to the next task.

for tasks that might fail due to transient issues, you can configure automatic retries:

```python
from shadows import Retry

async def flaky_api_call(
    url: str,
    retry: Retry = Retry(attempts=3, delay=timedelta(seconds=5))
) -> None:
    # This will retry up to 3 times with 5 seconds between each attempt
    response = await http_client.get(url)
    if response.status_code != 200:
        raise Exception(f"API returned {response.status_code}")

    print(f"Success on attempt {retry.attempt}")
```

tasks use a dependency injection pattern similar to fastapi. the `retry` dependency tells shadow how to handle failures for that specific task.

## worker configuration

workers handle task delivery guarantees and fault tolerance. by default, workers process up to 10 tasks simultaneously, but you can adjust this with the `concurrency=` parameter or `--concurrency` cli option. if a worker crashes, its tasks are redelivered to other workers after `redelivery_timeout` expires - you'll want to set this higher than your longest-running task.

shadow provides at-least-once delivery semantics, meaning tasks may be delivered more than once if workers crash, so design your tasks to be idempotent when possible.

## what's next?

you now know the core concepts: creating shadowss, scheduling work with idempotent keys, running workers, and basic error handling. this gives you what you need to build background task systems for most applications.

ready for more? check out:

- **[dependencies guide](dependencies.md)** - access current shadows, advanced retry patterns, timeouts, and custom dependencies
- **[testing with shadow](testing.md)** - ergonomic testing utilities for unit and integration tests
- **[advanced task patterns](advanced-patterns.md)** - perpetual tasks, striking/restoring, logging, and task chains
- **[shadow in production](production.md)** - redis architecture, monitoring, and deployment strategies
- **[api reference](api-reference.md)** - complete documentation of all classes and methods

## a note on security

shadow uses `cloudpickle` to serialize task functions and their arguments. this allows passing nearly any python object as task arguments, but also means deserializing arguments can execute arbitrary code. only schedule tasks from trusted sources in your system.
