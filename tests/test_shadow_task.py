#!/usr/bin/env python3
"""
Test script for shadow-task package
"""

import asyncio
from datetime import datetime, timedelta, timezone
from shadows import Shadow, Worker

async def simple_task(name: str) -> None:
    """A simple task that prints a greeting."""
    print(f"Hello, {name}! Task executed at {datetime.now()}")

async def test_basic_functionality():
    """Test basic shadow-task functionality."""
    print("Testing shadow-task package...")
    
    # Create a shadow instance
    async with Shadow(name="test-shadows") as shadows:
        print("✓ Shadow instance created successfully")
        
        # Register the task
        shadows.register(simple_task)
        print("✓ Task registered successfully")
        
        # Schedule an immediate task
        await shadows.add(simple_task)("World")
        print("✓ Task scheduled successfully")
        
        # Create a worker and run it
        async with Worker(shadows) as worker:
            print("✓ Worker created successfully")
            
            # Run until finished
            await worker.run_until_finished()
            print("✓ Worker completed successfully")

async def test_scheduled_task():
    """Test scheduled task functionality."""
    print("\nTesting scheduled tasks...")
    
    async with Shadow(name="test-scheduled") as shadows:
        shadows.register(simple_task)
        
        # Schedule a task for 2 seconds from now
        future_time = datetime.now(timezone.utc) + timedelta(seconds=2)
        await shadows.add(simple_task, when=future_time)("Scheduled World")
        print("✓ Scheduled task created successfully")
        
        async with Worker(shadows) as worker:
            print("Waiting for scheduled task to execute...")
            await worker.run_until_finished()
            print("✓ Scheduled task completed successfully")

async def test_task_with_key():
    """Test task with custom key functionality."""
    print("\nTesting tasks with custom keys...")
    
    async with Shadow(name="test-keys") as shadows:
        shadows.register(simple_task)
        
        # Schedule task with custom key
        await shadows.add(simple_task, key="test-key-1")("Keyed World")
        print("✓ Task with custom key scheduled")
        
        # Try to schedule another task with the same key (should be ignored)
        await shadows.add(simple_task, key="test-key-1")("Duplicate Key")
        print("✓ Duplicate key task attempted (should be ignored)")
        
        async with Worker(shadows) as worker:
            await worker.run_until_finished()
            print("✓ Keyed task completed successfully")

async def main():
    """Run all tests."""
    print("=" * 50)
    print("SHADOW-TASK PACKAGE TEST")
    print("=" * 50)
    
    try:
        await test_basic_functionality()
        await test_scheduled_task()
        await test_task_with_key()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 