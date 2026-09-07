"""Modules of the cuGraph-PyG link-prediction trainer (``scripts/cupyg_lp.py``).

Deliberately ``wgl``-free: this package runs inside the ``cupyg`` container image (NGC PyG base +
RAPIDS + WholeGraph), which does not install the project package. ``scripts/`` is mounted at
``/gs-scripts``, so the entry point's directory puts this package on ``sys.path``.

The published full-CC artifact uses the shallow encoder on the sampler-free path; ``sage.py`` and
``loader_sampled.py`` are the alternative-model modules and can be dropped together -- see
``docs/public-release.md``.
"""
