"""The link-prediction model: a learnable node-embedding table + dot decoder.

This is the published artifact's architecture -- a shallow, featureless, transductive first-order
embedding (LINE-1st / PyTorch-BigGraph-dot family). ``--encoder sage`` swaps in the optional
GraphSAGE stack from :mod:`cupyg.sage`, which the full-CC run does not use.
"""

from __future__ import annotations

import torch


class LPModel(torch.nn.Module):
    """Featureless LP: learnable node-embedding table + (optional) GraphSAGE + dot decoder.

    ``shallow`` scores ``emb[src] · emb[dst]`` directly (apples-to-apples with the in-house dot).
    ``sage`` message-passes the table through ``num_layers`` ``SAGEConv`` layers over the sampled
    subgraph before the dot — this is where GPU neighbour sampling actually earns its keep.
    """

    def __init__(
        self, num_nodes: int, dim: int, encoder: str, num_layers: int, wg_module: object = None
    ) -> None:
        """Build the embedding table and, for ``sage``, the ``SAGEConv`` stack.

        ``wg_module`` (a ``WholeMemoryEmbeddingModule``) shards the node-embedding table across the
        WholeGraph communicator (multi-GPU path); when ``None`` the table is a plain single-GPU
        ``nn.Embedding``. The sharded table trains through its *own* WholeMemory sparse optimizer
        (not autograd/DDP), so it registers no ``nn.Parameter`` here — only the ``sage`` convs do.
        """
        super().__init__()
        self.encoder = encoder
        self.sharded = wg_module is not None
        if self.sharded:
            self.emb_module = wg_module
        else:
            self.emb = torch.nn.Embedding(num_nodes, dim)
            torch.nn.init.normal_(self.emb.weight, std=0.1)
        if encoder == "sage":
            from cupyg.sage import build_convs  # optional model variant; see docs/public-release.md

            self.convs = build_convs(dim, num_layers)

    def _lookup(self, n_id: torch.Tensor) -> torch.Tensor:
        """Gather the (sharded or local) embedding rows for ``n_id``."""
        return self.emb_module(n_id) if self.sharded else self.emb(n_id)

    def _encode_batch(self, batch: object) -> torch.Tensor:
        # Single node/edge type -> cuGraph-PyG yields a homogeneous ``Data`` batch, so node ids and
        # message-passing edges are plain attributes (not keyed by node/edge type as in the hetero
        # ``mag_lp_mnmg.py`` example).
        x = self._lookup(batch.n_id)
        if self.encoder == "sage":
            from cupyg.sage import apply_convs

            return apply_convs(self.convs, x, batch.edge_index)
        return x

    def forward(self, batch: object) -> torch.Tensor:
        """Encode the sampled subgraph -> per-node reps (the DDP entry point for ``sage`` convs).

        Scoring lives in the loss functions (:func:`_inbatch_loss` / :func:`_binary_loss`) so the
        in-batch-negative path can reuse these node reps; routing the encode through ``forward``
        keeps DDP's gradient reducer engaged for the convs under ``world_size>1``.
        """
        return self._encode_batch(batch)


def _base(model: object) -> LPModel:
    """Unwrap DDP to the underlying ``LPModel`` (a no-op when not DDP-wrapped)."""
    return model.module if hasattr(model, "module") else model

