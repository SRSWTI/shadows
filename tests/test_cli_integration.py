#!/usr/bin/env python3
"""
Test script for CLI integration with shadow-task
"""

import asyncio
import subprocess
import time
from datetime import datetime, timedelta, timezone
from shadows import Shadow

async def add_tasks_for_cli_test():
    """Add tasks that can be processed by the CLI worker."""
    print("Adding tasks for CLI worker test...")
    
    async with Shadow(name="test-cli") as shadows:
        # Register tasks
        from tasks import hello_world, process_data, batch_processor
        shadows.register(hello_world)
        shadows.register(process_data)
        shadows.register(batch_processor)
        
        # Add some immediate tasks
        await shadows.add(hello_world)("CLI Test 1")
        await shadows.add(hello_world)("CLI Test 2")
        await shadows.add(process_data)(1)
        await shadows.add(batch_processor)(3)
        
        # Add a scheduled task
        future_time = datetime.now(timezone.utc) + timedelta(seconds=3)
        await shadows.add(hello_world, when=future_time, key="scheduled-task")("Scheduled CLI Test")
        
        print("✓ Tasks added successfully")
        
        # Take a snapshot to show the tasks
        snapshot = await shadows.snapshot()
        print(f"✓ Snapshot: {len(snapshot.running)} running, {len(snapshot.future)} future tasks")

async def test_worker_processing():
    """Test that the worker can process the tasks we added."""
    print("\nTesting worker processing...")
    
    # Start a worker in a subprocess
    worker_process = subprocess.Popen([
        "shadows", "worker", 
        "--tasks", "tasks:all_tasks",
        "--shadows", "test-cli",
        "--concurrency", "2"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    try:
        # Let the worker run for a few seconds
        print("Worker started, letting it process tasks...")
        time.sleep(5)
        
        # Check the snapshot again
        async with Shadow(name="test-cli") as shadows:
            snapshot = await shadows.snapshot()
            print(f"✓ After worker processing: {len(snapshot.running)} running, {len(snapshot.future)} future tasks")
        
    finally:
        # Stop the worker
        worker_process.terminate()
        worker_process.wait()
        print("✓ Worker stopped")

async def test_cli_commands():
    """Test various CLI commands."""
    print("\nTesting CLI commands...")
    
    # Test snapshot command
    result = subprocess.run([
        "shadows", "snapshot", "--shadows", "test-cli"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Snapshot command works")
        print("Snapshot output:")
        print(result.stdout)
    else:
        print(f"❌ Snapshot command failed: {result.stderr}")
    
    # Test workers ls command
    result = subprocess.run([
        "shadows", "workers", "ls", "--shadows", "test-cli"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Workers ls command works")
        print("Workers output:")
        print(result.stdout)
    else:
        print(f"❌ Workers ls command failed: {result.stderr}")

async def main():
    """Run the CLI integration test."""
    print("=" * 50)
    print("SHADOW-TASK CLI INTEGRATION TEST")
    print("=" * 50)
    
    try:
        await add_tasks_for_cli_test()
        await test_worker_processing()
        await test_cli_commands()
        
        print("\n" + "=" * 50)
        print("CLI INTEGRATION TEST COMPLETED! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ CLI integration test failed with error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 