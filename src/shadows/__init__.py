"""
shadows - A distributed background task system for Python functions.

shadows focuses on scheduling future work as seamlessly and efficiently as immediate work.
"""

from importlib.metadata import version

__version__ = version("shadow-task")

from .annotations import Logged
from .dependencies import (
    ConcurrencyLimit,
    CurrentShadow,
    CurrentExecution,
    CurrentWorker,
    Depends,
    ExponentialRetry,
    Perpetual,
    Retry,
    TaskArgument,
    TaskKey,
    TaskLogger,
    Timeout,
)
from .shadows import Shadow
from .execution import Execution
from .worker import Worker

__all__ = [
    "__version__",
    "ConcurrencyLimit",
    "CurrentShadow",
    "CurrentExecution",
    "CurrentWorker",
    "Depends",
    "Shadow",
    "Execution",
    "ExponentialRetry",
    "Logged",
    "Perpetual",
    "Retry",
    "TaskArgument",
    "TaskKey",
    "TaskLogger",
    "Timeout",
    "Worker",
]
