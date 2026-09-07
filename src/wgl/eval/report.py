"""JSON report serialization for the downstream eval commands.

Every ``wgl eval`` subcommand ends by writing a JSON payload whose leaves are a mix of Python
scalars, numpy scalars (``np.float64`` from a metric) and small numpy arrays (per-class support).
``json.dumps`` rejects all of the numpy ones, so each eval script grew its own ``_jsonable``
default. This is that default, once.
"""

from __future__ import annotations

import json
from pathlib import Path


def jsonable(obj: object) -> object:
    """``json.dumps`` default: unwrap numpy scalars/arrays into plain Python.

    Args:
        obj: The object ``json.dumps`` could not serialize.

    Returns:
        A JSON-serializable equivalent (list for arrays, Python scalar for numpy scalars, else the
        ``str`` repr as a last resort so a report never fails to write).
    """
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def write_json_report(path: str | Path, payload: object) -> Path:
    """Write an eval payload as indented JSON, creating the parent directory.

    Args:
        path: Destination ``.json`` path.
        payload: The report payload (numpy scalars/arrays are unwrapped via :func:`jsonable`).

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=jsonable))
    return out
