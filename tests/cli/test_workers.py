import asyncio
from datetime import timedelta

from typer.testing import CliRunner

from shadows.cli import app
from shadows.shadows import Shadow
from shadows.worker import Worker


async def test_list_workers_command(shadows: Shadow, runner: CliRunner):
    """Should list all active workers"""
    heartbeat = timedelta(milliseconds=20)
    shadows.heartbeat_interval = heartbeat
    shadows.missed_heartbeats = 3

    async with Worker(shadows, name="worker-1"), Worker(shadows, name="worker-2"):
        await asyncio.sleep(heartbeat.total_seconds() * 5)

        result = await asyncio.get_running_loop().run_in_executor(
            None,
            runner.invoke,
            app,
            [
                "workers",
                "ls",
                "--url",
                shadows.url,
                "--shadows",
                shadows.name,
            ],
        )
        assert result.exit_code == 0, result.output

        assert "worker-1" in result.output
        assert "worker-2" in result.output


async def test_list_workers_for_task(shadows: Shadow, runner: CliRunner):
    """Should list workers that can handle a specific task"""
    heartbeat = timedelta(milliseconds=20)
    shadows.heartbeat_interval = heartbeat
    shadows.missed_heartbeats = 3

    async with Worker(shadows, name="worker-1"), Worker(shadows, name="worker-2"):
        await asyncio.sleep(heartbeat.total_seconds() * 5)

        result = await asyncio.get_running_loop().run_in_executor(
            None,
            runner.invoke,
            app,
            [
                "workers",
                "for-task",
                "trace",
                "--url",
                shadows.url,
                "--shadows",
                shadows.name,
            ],
        )
        assert result.exit_code == 0, result.output

        assert "worker-1" in result.output
        assert "worker-2" in result.output
