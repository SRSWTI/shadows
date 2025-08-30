import asyncio
import logging

import pytest
from typer.testing import CliRunner

from shadows.cli import app
from shadows.shadows import Shadow
from shadows.worker import Worker


async def test_trace_command(
    runner: CliRunner,
    shadows: Shadow,
    worker: Worker,
    caplog: pytest.LogCaptureFixture,
):
    """Should add a trace task to the shadows"""
    result = await asyncio.get_running_loop().run_in_executor(
        None,
        runner.invoke,
        app,
        [
            "tasks",
            "trace",
            "hiya!",
            "--url",
            shadows.url,
            "--shadows",
            shadows.name,
        ],
    )
    assert result.exit_code == 0
    assert "Added trace task" in result.stdout.strip()

    with caplog.at_level(logging.INFO):
        await worker.run_until_finished()

    assert "hiya!" in caplog.text
    assert "ERROR" not in caplog.text


async def test_fail_command(
    runner: CliRunner,
    shadows: Shadow,
    worker: Worker,
    caplog: pytest.LogCaptureFixture,
):
    """Should add a trace task to the shadows"""
    result = await asyncio.get_running_loop().run_in_executor(
        None,
        runner.invoke,
        app,
        [
            "tasks",
            "fail",
            "hiya!",
            "--url",
            shadows.url,
            "--shadows",
            shadows.name,
        ],
    )
    assert result.exit_code == 0
    assert "Added fail task" in result.stdout.strip()

    with caplog.at_level(logging.INFO):
        await worker.run_until_finished()

    assert "hiya!" in caplog.text
    assert "ERROR" in caplog.text


async def test_sleep_command(
    runner: CliRunner,
    shadows: Shadow,
    worker: Worker,
    caplog: pytest.LogCaptureFixture,
):
    """Should add a trace task to the shadows"""
    result = await asyncio.get_running_loop().run_in_executor(
        None,
        runner.invoke,
        app,
        [
            "tasks",
            "sleep",
            "0.1",
            "--url",
            shadows.url,
            "--shadows",
            shadows.name,
        ],
    )
    assert result.exit_code == 0
    assert "Added sleep task" in result.stdout.strip()

    with caplog.at_level(logging.INFO):
        await worker.run_until_finished()

    assert "Sleeping for 0.1 seconds" in caplog.text
    assert "ERROR" not in caplog.text
