# welcome to shadows

shadows represents a sophisticated distributed background task processing framework designed specifically for python applications, emphasizing seamless scheduling capabilities for both immediate and future computational workloads with exceptional efficiency.



## at a glance

```python
from datetime import datetime, timedelta, timezone

from shadows import Shadow


async def greet(name: str, greeting="Hello") -> None:
    print(f"{greeting}, {name} at {datetime.now()}!")


async with Shadow() as shadows:
    await shadows.add(greet)("Jane")

    now = datetime.now(timezone.utc)
    soon = now + timedelta(seconds=3)
    await shadows.add(greet, when=soon)("John", greeting="Howdy")
```

And in another process, run a worker:

```python
from shadows import Shadow, Worker

async with Shadow() as shadows:
    async with Worker(shadows) as worker:
        await worker.run_until_finished()
```

Which produces:

```
Hello, Jane at 2025-03-05 13:58:21.552644!
Howdy, John at 2025-03-05 13:58:24.550773!
```

## why shadows?

⚡️ lightning-fast one-way background task processing devoid of unnecessary complexity

📅 seamless scheduling of immediate or future workloads through a unified interface

⏭️ bypass problematic tasks or parameters without requiring code redeployment

🌊 purpose-built architecture optimized for redis streams

🧩 comprehensive type safety and type awareness for all background task functions

## how it works

shadows integrates two distinct modes of task execution:

1. **immediate tasks** are pushed onto a redis stream and become available for pickup by any worker instance.
2. **scheduled tasks** are pushed onto a redis sorted set with a specific schedule timestamp. a continuous loop within each worker moves scheduled tasks onto the stream when their scheduled time arrives. this movement is performed as a lua script to ensure atomicity and consistency.

shadow requires a [redis](http://redis.io/) server with streams support (introduced in redis 5.0.0). shadow is thoroughly tested with redis 6 and 7, ensuring compatibility and reliability.

for more detailed information, explore our [getting started](getting-started.md) guide or dive into the [api reference](api-reference.md).
