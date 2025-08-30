import asyncio
import subprocess
import sys

import shadows


async def test_module_invocation_as_cli_entrypoint():
    """Should allow invoking shadows as a module with python -m shadows."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "shadows",
        "version",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    assert process.returncode == 0, stderr.decode()
    assert stdout.decode().strip() == shadows.__version__
