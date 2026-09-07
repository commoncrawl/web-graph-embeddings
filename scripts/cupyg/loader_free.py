"""Seed-edge minibatches WITHOUT cuGraph neighbour sampling -- the shallow full-CC path.

A shallow encoder never message-passes, so sampling a neighbourhood is pure waste: the graph does
not need to be on the GPU at all. Dropping the GraphStore frees ~63 GB/rank at full CC and
sidesteps the cuGraph MG sampler's 32-bit edge-count overflow past ~2B edges/rank.
"""

from __future__ import annotations

import numpy as np
import torch



class _SFBatch:
    """Minimal homogeneous batch for the sampler-free shallow path.

    Mirrors the fields ``LPModel.forward`` / ``_inbatch_loss`` read off a cuGraph ``Data`` batch
    (``n_id`` global node ids, ``edge_label_index`` LOCAL indices into ``n_id``, all-positive
    ``edge_label``), so the model + loss are byte-for-byte unchanged.
    """

    def __init__(self, seed_edges: np.ndarray) -> None:
        src = torch.as_tensor(seed_edges[0], dtype=torch.int64)
        dst = torch.as_tensor(seed_edges[1], dtype=torch.int64)
        self.n_id, inv = torch.unique(torch.cat([src, dst]), return_inverse=True)
        self.edge_label_index = inv.view(2, -1)  # local (into n_id) src/dst of each positive edge
        self.edge_label = torch.ones(src.numel(), dtype=torch.float32)  # all pos; loss adds negs

    def cuda(self) -> _SFBatch:
        self.n_id = self.n_id.cuda()
        self.edge_label_index = self.edge_label_index.cuda()
        return self  # edge_label stays on CPU; the loss moves it (parity with the cuGraph batch)



class _SamplerFreeEdgeLoader:
    """Seed-edge minibatches WITHOUT cuGraph neighbour sampling — the shallow full-CC path.

    cuGraph's MG sampler 32-bit-overflows beyond ~2 B edges/GPU (documented limit; 13 B/4 ranks =
    3.27 B > INT32_MAX -> SIGSEGV on the first sample), and a shallow dot model discards sampled
    neighbours anyway. So we skip the GraphStore + LinkNeighborLoader entirely: iterate this rank's
    seed shard in ``bs`` chunks (drop_last, chunk-free, reshuffled per epoch) and emit an
    :class:`_SFBatch`. No graph on GPU (frees ~63 GB/rank); embeddings gather by GLOBAL id from the
    WholeMemory table; ``_inbatch_loss`` corrupts each positive with in-batch negatives. Batch count
    is exactly ``n // bs`` (no per-chunk drop), so ``epoch_cap`` stays exact for lockstep.
    """

    def __init__(self, edges: np.ndarray, bs: int, shuffle: bool, seed: int = 0) -> None:
        self._edges, self._bs, self._shuffle, self._seed = edges, bs, shuffle, seed
        self._epoch = 0

    def __iter__(self) -> object:
        n = int(self._edges.shape[1])
        order = np.arange(n)
        if self._shuffle:  # reshuffle seed order per epoch (same seed on every rank; own shard)
            np.random.default_rng(self._seed + self._epoch).shuffle(order)
            self._epoch += 1
        for start in range(0, n - self._bs + 1, self._bs):  # drop_last
            yield _SFBatch(self._edges[:, order[start : start + self._bs]])


# ── eval (MRR vs N negatives, parity with the in-house protocol) ─────────────────────────────

