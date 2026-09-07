"""Shared pytest setup.

Colour is disabled process-wide before any CLI test runs: with colour on, rich's option
highlighter splits ``--emb`` into ``-`` and ``-emb`` with an escape sequence between them, so
plain substring assertions on ``result.output`` fail. Local runs are not a terminal and stay
uncoloured; CI is, which is why this only ever broke there.
"""

from __future__ import annotations

import os

for _forced in ("FORCE_COLOR", "CLICOLOR_FORCE"):
    os.environ.pop(_forced, None)
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
