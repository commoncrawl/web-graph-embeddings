"""Command-line interface for web-graph-learning.

The CLI exposes the full workflow as explicit subcommands with structured stdout logging, so it is
straightforward for both humans and agents to drive:

    wgl info
    wgl data toy --out data/toy
    wgl data prepare --vertices v.txt.gz --edges e.txt.gz --out data/cc_host
    wgl data split --graph data/toy --task link_prediction --out data/toy
    wgl train --config configs/toy_linkpred.yaml --set train.epochs=5
    wgl eval run --run runs/<...>
    wgl eval topic --graph data/cc_deg8 --emb emb.npy --host-topics ht.parquet --out t.json
    wgl embed --run runs/<...> --out emb.npy
    wgl sweep --config configs/sweep_scaling.yaml
    wgl knn build --hf-repo malteos/web-graph-embeddings --shards 8 --row-groups 1 \
        --out data/knn/cc_deg8_poc
    wgl knn query --index data/knn/cc_deg8_poc --host www.example.de

``torch`` is imported lazily inside the commands that need it, so the CLI stays importable in the
torch-free CPU environments the label-prep Slurm jobs run in (``slurm/topic_cpu.sbatch``); see
``tests/test_cli_no_torch.py``.

Command groups live in one module each and attach themselves to :data:`app` on import, so the
import list below is the whole assembly. A minimal public reproduction checkout keeps only the
modules under "published pipeline" and deletes the rest -- see ``docs/public-release.md``.
"""

from __future__ import annotations

from wgl.cli._app import app

# Command groups attach themselves to `app` on import. Import order is the assembly; deleting a
# line here (and its module) removes that group from the CLI. isort is off so the two tiers stay
# visually separate -- see docs/public-release.md.
# isort: off

# --- published embedding pipeline ---------------------------------------------------------------
from wgl.cli import data_cc  # noqa: F401  (registers `wgl data`: steps 2-4)
from wgl.cli import release_cmds  # noqa: F401  (registers `wgl release`: step 6)

# --- everything else: experiment slices, in-house trainer, evaluation, figures, kNN viewer -------

# isort: on

__all__ = ["app"]
