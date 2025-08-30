# shadow in production

running shadow at scale requires understanding its redis-based architecture, configuring workers appropriately, and monitoring system health. this guide covers everything you need for reliable production deployments.

## redis streams architecture

shadow uses redis streams and sorted sets to provide reliable task delivery with at-least-once semantics. note that shadow requires a single redis instance and does not support redis cluster.

### task lifecycle

understanding how tasks flow through the system helps with monitoring and troubleshooting:

1. **immediate tasks** go directly to the redis stream and are available to any worker in the consumer group
2. **future tasks** are stored in the sorted set with their execution time as the score
3. **workers continuously move** due tasks from the sorted set to the stream
4. **consumer groups** ensure each task is delivered to exactly one worker
5. **acknowledgment** removes completed tasks; unacknowledged tasks are redelivered

### redelivery behavior

when a worker crashes or fails to acknowledge a task within `redelivery_timeout`, redis automatically makes the task available to other workers. this ensures reliability but means tasks may execute more than once.

```python
# Configure redelivery timeout based on your longest-running tasks
async with Worker(
    shadows,
    redelivery_timeout=timedelta(minutes=10)  # Adjust for your workload
) as worker:
    await worker.run_forever()
```

set redelivery timeout to be longer than your 99th percentile task duration to minimize duplicate executions.

### redis data structures

shadow creates several redis data structures for each shadows:

- **stream (`{shadows}:stream`)**: ready-to-execute tasks using redis consumer groups
- **sorted set (`{shadows}:queue`)**: future tasks ordered by scheduled execution time
- **hashes (`{shadows}:{key}`)**: serialized task data for scheduled tasks
- **set (`{shadows}:workers`)**: active worker heartbeats with timestamps
- **set (`{shadows}:worker-tasks:{worker}`)**: tasks each worker can execute
- **stream (`{shadows}:strikes`)**: strike/restore commands for operational control

## worker configuration

### core settings

workers have several configuration knobs for different environments:

```python
async with Worker(
    shadows,
    name="worker-1",                                    # Unique worker identifier
    concurrency=20,                                     # Parallel task limit
    redelivery_timeout=timedelta(minutes=5),           # When to redeliver tasks
    reconnection_delay=timedelta(seconds=5),           # Redis reconnection backoff
    minimum_check_interval=timedelta(milliseconds=100), # Polling frequency
    scheduling_resolution=timedelta(milliseconds=250),  # Future task check frequency
    schedule_automatic_tasks=True                       # Enable perpetual task startup
) as worker:
    await worker.run_forever()
```

### environment variable configuration

all settings can be configured via environment variables for production deployments:

```bash
# Core shadows settings
export SHADOW_NAME=orders
export SHADOW_URL=redis://redis.production.com:6379/0

# Worker settings
export SHADOW_WORKER_NAME=orders-worker-1
export SHADOW_WORKER_CONCURRENCY=50
export SHADOW_WORKER_REDELIVERY_TIMEOUT=10m
export SHADOW_WORKER_RECONNECTION_DELAY=5s
export SHADOW_WORKER_MINIMUM_CHECK_INTERVAL=100ms
export SHADOW_WORKER_SCHEDULING_RESOLUTION=250ms

# Monitoring
export SHADOW_WORKER_HEALTHCHECK_PORT=8080
export SHADOW_WORKER_METRICS_PORT=9090

# Logging
export SHADOW_LOGGING_LEVEL=INFO
export SHADOW_LOGGING_FORMAT=json

# Task modules
export SHADOW_TASKS=myapp.tasks:production_tasks
```

### cli usage

run workers in production using the cli:

```bash
# Basic worker
shadows worker --tasks myapp.tasks:all_tasks

# Production worker with full configuration
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

### tuning for different workloads

**high-throughput, fast tasks:**

```bash
shadows worker \
  --concurrency 100 \
  --redelivery-timeout 30s \
  --minimum-check-interval 50ms \
  --scheduling-resolution 100ms
```

**long-running, resource-intensive tasks:**

```bash
shadows worker \
  --concurrency 5 \
  --redelivery-timeout 1h \
  --minimum-check-interval 1s \
  --scheduling-resolution 5s
```

**mixed workload with perpetual tasks:**

```bash
shadows worker \
  --concurrency 25 \
  --redelivery-timeout 5m \
  --schedule-automatic-tasks \
  --tasks myapp.tasks:all_tasks,myapp.monitoring:health_checks
```

## connection management

### redis connection pools

shadow automatically manages redis connection pools, but you can tune them for your environment:

```python
from redis.asyncio import ConnectionPool

# Custom connection pool for high-concurrency workers
pool = ConnectionPool.from_url(
    "redis://redis.prod.com:6379/0",
    max_connections=50,  # Match or exceed worker concurrency
    retry_on_timeout=True
)

async with Shadow(name="orders", connection_pool=pool) as shadows:
    # Use the custom pool
    pass
```

### redis requirements

shadow requires a single redis instance and does not currently support redis cluster. for high availability, consider:

- **managed redis services** like aws elasticache, google cloud memorystore, or redis cloud
- **redis replicas** with manual failover procedures

```python
# With authentication
shadows_url = "redis://:password@redis.prod.com:6379/0"
```

### valkey support

shadow also works with valkey (redis fork):

```bash
export SHADOW_URL=valkey://valkey.prod.com:6379/0
```

## monitoring and observability

### prometheus metrics

enable prometheus metrics with the `--metrics-port` option:

```bash
shadows worker --metrics-port 9090
```

available metrics include:

#### task counters

- `shadows_tasks_added` - tasks scheduled
- `shadows_tasks_started` - tasks begun execution
- `shadows_tasks_succeeded` - successfully completed tasks
- `shadows_tasks_failed` - failed tasks
- `shadows_tasks_retried` - retry attempts
- `shadows_tasks_stricken` - tasks blocked by strikes

#### task timing

- `shadows_task_duration` - histogram of task execution times
- `shadows_task_punctuality` - how close tasks run to their scheduled time

#### system health

- `shadows_queue_depth` - tasks ready for immediate execution
- `shadows_schedule_depth` - tasks scheduled for future execution
- `shadows_tasks_running` - currently executing tasks
- `shadows_redis_disruptions` - redis connection failures
- `shadows_strikes_in_effect` - active strike rules

all metrics include labels for shadows name, worker name, and task function name.

### health checks

enable health check endpoints:

```bash
shadows worker --healthcheck-port 8080
```

the health check endpoint (`/`) returns 200 ok when the worker is healthy and able to process tasks.

### opentelemetry traces

shadow automatically creates opentelemetry spans for task execution:

- **span name**: `shadows.task.{function_name}`
- **attributes**: shadows name, worker name, task key, attempt number
- **status**: success/failure with error details
- **duration**: complete task execution time

configure your opentelemetry exporter to send traces to your observability platform. see the [opentelemetry python documentation](https://opentelemetry.io/docs/languages/python/) for configuration examples with various backends like jaeger, zipkin, or cloud providers.

### structured logging

configure structured logging for production:

```bash
# JSON logs for log aggregation
shadows worker --logging-format json --logging-level info

# Plain logs for simple deployments
shadows worker --logging-format plain --logging-level warning
```

log entries include:

- task execution start/completion
- error details with stack traces
- worker lifecycle events
- redis connection status
- strike/restore operations

### example grafana dashboard

monitor shadow health with queries like:

```promql
# Task throughput
rate(shadows_tasks_completed[5m])

# Error rate
rate(shadows_tasks_failed[5m]) / rate(shadows_tasks_started[5m])

# Queue depth trending
shadows_queue_depth

# P95 task duration
histogram_quantile(0.95, rate(shadows_task_duration_bucket[5m]))

# Worker availability
up{job="shadows-workers"}
```

## production guidelines

### capacity planning

**estimate concurrent tasks:**

```
concurrent_tasks = avg_task_duration * tasks_per_second
worker_concurrency = concurrent_tasks * 1.2  # 20% buffer
```

**size worker pools:**

- start with 1-2 workers per cpu core
- monitor cpu and memory usage
- scale horizontally rather than increasing concurrency indefinitely

### deployment strategies

**blue-green deployments:**

```bash
# Deploy new workers with different name
shadows worker --name orders-worker-v2 --tasks myapp.tasks:v2_tasks

# Gradually strike old task versions
shadows strike old_task_function

# Scale down old workers after tasks drain
```

### error handling

**configure appropriate retries:**

```python
# Transient failures - short delays
async def api_call(
    retry: Retry = Retry(attempts=3, delay=timedelta(seconds=5))
): ...

# Infrastructure issues - exponential backoff
async def database_sync(
    retry: ExponentialRetry = ExponentialRetry(
        attempts=5,
        minimum_delay=timedelta(seconds=30),
        maximum_delay=timedelta(minutes=10)
    )
): ...

# Critical operations - unlimited retries
async def financial_transaction(
    retry: Retry = Retry(attempts=None, delay=timedelta(minutes=1))
): ...
```

**dead letter handling:**

```python
async def process_order(order_id: str) -> None:
    try:
        await handle_order(order_id)
    except CriticalError as e:
        # Send to dead letter queue for manual investigation
        await send_to_dead_letter_queue(order_id, str(e))
        raise
```

### operational procedures

**graceful shutdown:**

```bash
# Workers handle SIGTERM gracefully
kill -TERM $WORKER_PID

# Or use container orchestration stop signals
docker stop shadows-worker
```

**emergency task blocking:**

```bash
# Block problematic tasks immediately
shadows strike problematic_function

# Block tasks for specific customers
shadows strike process_order customer_id == "problematic-customer"

# Restore when issues are resolved
shadows restore problematic_function
```

**monitoring checklist:**

- queue depth alerts (tasks backing up)
- error rate alerts (> 5% failure rate)
- task duration alerts (p95 > expected)
- worker availability alerts
- redis connection health

### scaling considerations

**horizontal scaling:**

- add workers across multiple machines
- use consistent worker naming for monitoring
- monitor redis memory usage as task volume grows

**vertical scaling:**

- increase worker concurrency for i/o bound tasks
- increase memory limits for large task payloads
- monitor cpu usage to avoid oversubscription

**redis scaling:**

- use managed redis services for high availability (redis cluster is not supported)
- monitor memory usage and eviction policies
- scale vertically for larger workloads

running shadow in production requires attention to these operational details, but the redis-based architecture and monitoring support can help with demanding production workloads.
