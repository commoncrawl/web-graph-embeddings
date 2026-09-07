"""Structured logging and run-directory management.

Every training/eval invocation creates a timestamped run directory under ``runs/<name>/<ts>/``
containing the resolved ``config.yaml``, a human-readable ``log.txt``, and a machine-readable
``metrics.jsonl`` (one JSON object per logged step). Logging goes to both stdout (rich) and the
log file, so the workflow is friendly to humans and agents alike.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

_LOGGER_NAME = "wgl"


def get_logger() -> logging.Logger:
    """Return the package logger (configured by :func:`setup_logging`)."""
    return logging.getLogger(_LOGGER_NAME)


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the package logger with a rich stdout handler and optional file handler.

    Args:
        log_file: Optional path to also write plain-text logs to.
        level: Logging level.

    Returns:
        The configured logger.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    rich_handler = RichHandler(rich_tracebacks=True, show_path=False, omit_repeated_times=False)
    rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    logger.addHandler(rich_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(file_handler)
    return logger


def timestamp() -> str:
    """Return a filesystem-safe timestamp string."""
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


class RunDirectory:
    """A timestamped output directory for a single experiment run.

    Attributes:
        path: The run directory path.
    """

    def __init__(self, runs_dir: Path, name: str) -> None:
        """Create (or address) a run directory under ``runs_dir/name/<ts>``.

        Args:
            runs_dir: Root directory holding all runs.
            name: Experiment name (a subdirectory).
        """
        self.path = Path(runs_dir) / name / timestamp()
        self.path.mkdir(parents=True, exist_ok=True)
        self._metrics_file = self.path / "metrics.jsonl"

    @property
    def config_path(self) -> Path:
        """Path to the serialized resolved config."""
        return self.path / "config.yaml"

    @property
    def log_path(self) -> Path:
        """Path to the plain-text log file."""
        return self.path / "log.txt"

    @property
    def checkpoint_path(self) -> Path:
        """Path to the model checkpoint."""
        return self.path / "model.pt"

    def log_metrics(self, step: int, **metrics: Any) -> None:
        """Append a metrics record to ``metrics.jsonl``.

        Args:
            step: Step or epoch index.
            **metrics: Scalar metric values to record.
        """
        record = {"step": step, "wall_time": time.time(), **metrics}
        with self._metrics_file.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def read_metrics(self) -> list[dict[str, Any]]:
        """Read all recorded metric rows back."""
        if not self._metrics_file.exists():
            return []
        return [json.loads(line) for line in self._metrics_file.read_text().splitlines() if line]


def mark_last_run(reference_dir: Path, run_path: Path) -> None:
    """Write a ``.last_run`` pointer file so the CLI/Makefile can find the latest run.

    Args:
        reference_dir: Directory to write the pointer into (typically the data dir).
        run_path: The run directory to record.
    """
    reference_dir.mkdir(parents=True, exist_ok=True)
    (reference_dir / ".last_run").write_text(str(run_path) + os.linesep)
