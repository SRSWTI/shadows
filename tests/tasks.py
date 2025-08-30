"""
Tasks module for testing shadows CLI functionality
"""

import asyncio
from datetime import datetime
from shadows import CurrentShadow

async def hello_world(name: str = "World") -> None:
    """A simple hello world task."""
    print(f"Hello, {name}! Executed at {datetime.now()}")

async def process_data(data_id: int, shadows: CurrentShadow = CurrentShadow()) -> None:
    """A task that processes data and schedules follow-up work."""
    print(f"Processing data {data_id}")
    
    # Simulate some work
    await asyncio.sleep(0.1)
    
    # Schedule a follow-up task
    await shadows.add(hello_world)(f"Data {data_id} Processor")

async def batch_processor(batch_size: int = 5) -> None:
    """A task that processes batches of work."""
    print(f"Processing batch of size {batch_size}")
    
    for i in range(batch_size):
        print(f"  Processing item {i} in batch")

# List of tasks that can be used with the CLI
all_tasks = [hello_world, process_data, batch_processor] 