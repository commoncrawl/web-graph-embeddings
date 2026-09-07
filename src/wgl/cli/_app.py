"""The root ``wgl`` Typer application.

Command groups live in sibling modules and attach themselves here on import (see
:mod:`wgl.cli`), so a subset of the CLI can be shipped by dropping modules from that import list
without touching any command body.
"""

from __future__ import annotations

import typer

from wgl.logging import get_logger, setup_logging

app = typer.Typer(add_completion=False, help="Directed Web graph representation learning.")


@app.command()
def info() -> None:
    """Print environment and framework information."""
    setup_logging()
    logger = get_logger()
    import torch

    logger.info("torch %s | CUDA available: %s", torch.__version__, torch.cuda.is_available())
    try:
        import wgl.models  # noqa: F401  (registers models)
        from wgl.registry import MODELS
    except ImportError:
        # A minimal reproduction checkout ships the data/release pipeline without the model zoo.
        logger.info("registered models: (wgl.models not installed)")
    else:
        logger.info("registered models: %s", ", ".join(MODELS.names()))
