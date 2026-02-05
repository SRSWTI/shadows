import asyncio
import os
import socket
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from docker import DockerClient

# Add src to path
sys.path.insert(0, os.path.abspath("src"))
from shadows import Shadow, Worker, TaskLogger

@asynccontextmanager
async def run_redis(version: str) -> AsyncGenerator[str, None]:
    def get_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    port = get_free_port()

    client = DockerClient.from_env()
    container = client.containers.run(
        f"redis:{version}",
        detach=True,
        ports={"6379/tcp": port},
        auto_remove=True,
    )

    # Wait for Redis to be ready
    for line in container.logs(stream=True):
        if b"Ready to accept connections" in line:
            break

    url = f"redis://localhost:{port}/0"
    print(f"Redis running at {url}")
    try:
        yield url
    finally:
        container.stop()

async def noop():
    pass

async def benchmark(count: int = 1000):
    print(f"Benchmarking {count} tasks...")
    
    async with run_redis("7.4.2") as redis_url:
        # Check if local queue is supported (via kwargs or introspection)
        # For now, we just pass standard args. 
        # When we add the feature, we might need to enable it if it's opt-in.
        # But user requested `enable_local_queue=True`.
        
        # We can try/except or inspect to see if we can pass enable_local_queue
        # But for the "before" test, we just use standard init.
        
        try:
            shadow = Shadow(url=redis_url, enable_local_queue=True)
            print("With enable_local_queue=True")
        except TypeError:
            shadow = Shadow(url=redis_url)
            print("Standard Shadow (no local queue support yet)")
            
        async with shadow as s:
            s.register(noop)
            
            # Pre-fill tasks
            print("Producing tasks...")
            prod_start = time.time()
            # We usegather for speed in production if possible, but s.add is async
            # s.add returns a wrapper, calling it schedules the task.
            # await s.add(noop)()
            
            tasks = []
            for i in range(count):
                # Use unique keys to avoid de-duplication if any
                tasks.append(s.add(noop, key=f"task-{i}")())
            
            await asyncio.gather(*tasks)
            prod_time = time.time() - prod_start
            print(f"Production time: {prod_time:.4f}s ({count/prod_time:.2f} tasks/s)")
            
            print("Consuming tasks...")
            consume_start = time.time()
            async with Worker(s, concurrency=50, schedule_automatic_tasks=False) as worker:
                await worker.run_until_finished()
            
            consume_time = time.time() - consume_start
            print(f"Consumption time: {consume_time:.4f}s ({count/consume_time:.2f} tasks/s)")
            print(f"Total time (serial prod+cons): {prod_time + consume_time:.4f}s")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    asyncio.run(benchmark(count))
