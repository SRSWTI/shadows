"""
advanced test script for shadow-task package demonstrating advanced features
"""

import asyncio
import random
from datetime import datetime, timedelta, timezone
from shadows import Shadow, Worker, Perpetual, Retry, CurrentShadow

async def flaky_task(task_id: int, retry: Retry = Retry(attempts=3, delay=timedelta(seconds=1))) -> None:
    """A task that sometimes fails to test retry functionality."""
    if random.random() < 0.7:  # 70% chance of failure, flaky like i said 
        print(f"Task {task_id} failed on attempt {retry.attempt}")
        raise Exception(f"Simulated failure for task {task_id}")
    
    print(f"Task {task_id} succeeded on attempt {retry.attempt}!")

async def perpetual_counter(
    counter: int = 0,
    perpetual: Perpetual = Perpetual(every=timedelta(seconds=1))
) -> None:
    """A perpetual task that counts up and stops after 3 iterations."""
    print(f"Perpetual counter: {counter}")
    
    if counter >= 2:
        print("Perpetual task completed, stopping...")
        perpetual.cancel()
        return
    
    # Schedule next iteration with incremented counter
    perpetual.perpetuate(counter + 1)

async def task_with_dependencies(
    user_id: int,
    shadows: Shadow = CurrentShadow()
) -> None:
    """A task that demonstrates dependency injection and scheduling more work."""
    print(f"Processing user {user_id}")
    
    # Simulate some work
    await asyncio.sleep(0.1)
    
    # Schedule a follow-up task
    await shadows.add(simple_followup)(user_id, "completed")

async def simple_followup(user_id: int, status: str) -> None:
    """A simple follow-up task."""
    print(f"Follow-up for user {user_id}: {status}")

async def test_retry_functionality():
    """Test retry functionality with flaky tasks."""
    print("\n" + "=" * 50)
    print("TESTING RETRY FUNCTIONALITY")
    print("=" * 50)
    
    async with Shadow(name="test-retries") as shadows:
        shadows.register(flaky_task)
        
        # Schedule multiple flaky tasks
        for i in range(3):
            await shadows.add(flaky_task)(i)
        
        async with Worker(shadows) as worker:
            await worker.run_until_finished()
            print("✓ Retry functionality test completed")

async def test_perpetual_tasks():
    """Test perpetual task functionality."""
    print("\n" + "=" * 50)
    print("TESTING PERPETUAL TASKS")
    print("=" * 50)
    
    async with Shadow(name="test-perpetual") as shadows:
        shadows.register(perpetual_counter)
        
        # Start the perpetual task
        await shadows.add(perpetual_counter)(0)
        
        async with Worker(shadows) as worker:
            # Let it run for a limited time
            await worker.run_at_most({"perpetual_counter": 3})
            print("✓ Perpetual task test completed")

async def test_dependencies_and_chaining():
    """Test dependency injection and task chaining."""
    print("\n" + "=" * 50)
    print("TESTING DEPENDENCIES AND CHAINING")
    print("=" * 50)
    
    async with Shadow(name="test-dependencies") as shadows:
        shadows.register(task_with_dependencies)
        shadows.register(simple_followup)
        
        # Schedule tasks that will create follow-up tasks
        for i in range(3):
            await shadows.add(task_with_dependencies)(i)
        
        async with Worker(shadows) as worker:
            await worker.run_until_finished()
            print("✓ Dependencies and chaining test completed")

async def test_snapshot_functionality():
    """Test snapshot functionality to inspect the shadows."""
    print("\n" + "=" * 50)
    print("TESTING SNAPSHOT FUNCTIONALITY")
    print("=" * 50)
    
    async with Shadow(name="test-snapshot") as shadows:
        shadows.register(simple_task)
        
        # Schedule some immediate and future tasks
        await shadows.add(simple_task)("Immediate")
        
        future_time = datetime.now(timezone.utc) + timedelta(seconds=5)
        await shadows.add(simple_task, when=future_time, key="future-task")("Future")
        
        # Take a snapshot
        snapshot = await shadows.snapshot()
        print(f"✓ Snapshot taken: {len(snapshot.running)} running, {len(snapshot.future)} future tasks")
        
        # Process the tasks
        async with Worker(shadows) as worker:
            await worker.run_until_finished()
            print("✓ Snapshot functionality test completed")

async def simple_task(name: str) -> None:
    """A simple task for testing."""
    print(f"Simple task executed: {name}")

async def main():
    """Run all advanced tests."""
    print("=" * 50)
    print("SHADOW-TASK ADVANCED FEATURES TEST")
    print("=" * 50)
    
    try:
        await test_retry_functionality()
        await test_perpetual_tasks()
        await test_dependencies_and_chaining()
        await test_snapshot_functionality()
        
        print("\n" + "=" * 50)
        print("ALL ADVANCED TESTS PASSED! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Advanced test failed with error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 