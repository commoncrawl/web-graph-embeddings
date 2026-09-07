"""Held-out MRR/hits and the frozen embedding-table export.

``eval_mrr_gathered`` is the sharded path: it gathers only the node ids a scoring chunk needs, so
the ~27 GB full-CC table never has to fit on one GPU.
"""

from __future__ import annotations

import numpy as np
import torch

from cupyg.model import _base


@torch.no_grad()
def eval_mrr(
    emb: torch.Tensor,
    pos: np.ndarray,
    num_neg: int,
    hits_k: tuple[int, ...],
    gen: torch.Generator,
) -> dict[str, float]:
    """Rank each held-out positive dst against ``num_neg`` uniform negatives -> MRR/Hits@K."""
    n_nodes = emb.shape[0]
    src = torch.as_tensor(pos[0], device="cuda")
    dst = torch.as_tensor(pos[1], device="cuda")
    ranks: list[torch.Tensor] = []
    chunk = 4096
    for start in range(0, src.shape[0], chunk):
        s, d = src[start : start + chunk], dst[start : start + chunk]
        hs = emb[s]
        pos_score = (hs * emb[d]).sum(dim=-1)
        neg = torch.randint(0, n_nodes, (s.shape[0], num_neg), device="cuda", generator=gen)
        neg_score = torch.bmm(hs.unsqueeze(1), emb[neg].transpose(1, 2)).squeeze(1)
        ranks.append((neg_score > pos_score[:, None]).sum(dim=1) + 1)
    r = torch.cat(ranks).float()
    hits = {f"hits@{k}": (r <= k).float().mean().item() for k in hits_k}
    return {"mrr": (1.0 / r).mean().item(), **hits}



@torch.no_grad()
def _gather_emb(model: object, ids: torch.Tensor) -> torch.Tensor:
    """Embeddings for global node ``ids`` from the shallow table (never materializes it whole).

    Sharded ``--wg-embed``: ``wm_embedding.gather`` is a COLLECTIVE — every rank must call it the
    same number of times; same ids -> identical rows on all ranks. Dense: a plain index.
    """
    base = _base(model)
    if base.sharded:
        return base.emb_module.wm_embedding.gather(ids, force_dtype=torch.float32)
    return base.emb.weight.detach()[ids]



@torch.no_grad()
def eval_mrr_gathered(
    model: object,
    pos: np.ndarray,
    num_nodes: int,
    num_neg: int,
    hits_k: tuple[int, ...],
    gen: torch.Generator,
) -> dict[str, float]:
    """MRR/Hits@K gathering ONLY the rows each chunk needs — the full-CC shallow path.

    The full ``[num_nodes, dim]`` table is ~27 GB at full CC and OOMs on one GPU; ``eval_mrr`` only
    indexes it, so instead gather ``emb[src|dst|neg]`` per chunk (bounded to ``chunk*(2+num_neg)``
    rows) from the WholeMemory table and score locally. Per-chunk gathers are collectives, so all
    ranks run the SAME chunk count / ids (eval edges replicated, ``gen`` seeded identically) →
    lockstep. Identical result to ``eval_mrr`` over the full table.
    """
    src = torch.as_tensor(pos[0], device="cuda")
    dst = torch.as_tensor(pos[1], device="cuda")
    ranks: list[torch.Tensor] = []
    chunk = 4096
    for start in range(0, src.shape[0], chunk):
        s, d = src[start : start + chunk], dst[start : start + chunk]
        neg = torch.randint(0, num_nodes, (s.shape[0], num_neg), device="cuda", generator=gen)
        hs = _gather_emb(model, s)
        pos_score = (hs * _gather_emb(model, d)).sum(dim=-1)
        hn = _gather_emb(model, neg.reshape(-1)).reshape(s.shape[0], num_neg, -1)
        neg_score = torch.bmm(hs.unsqueeze(1), hn.transpose(1, 2)).squeeze(1)
        ranks.append((neg_score > pos_score[:, None]).sum(dim=1) + 1)
    r = torch.cat(ranks).float()
    hits = {f"hits@{k}": (r <= k).float().mean().item() for k in hits_k}
    return {"mrr": (1.0 / r).mean().item(), **hits}



@torch.no_grad()
def export_table_chunked(
    model: object, num_nodes: int, dim: int, keep: bool, chunk: int = 2_000_000
) -> np.ndarray | None:
    """Assemble the full ``[num_nodes, dim]`` shallow table on CPU by gathering in chunks.

    The one-shot ``gather(arange(num_nodes))`` peaks at ~27 GB VRAM and OOMs at full CC; gather
    ``chunk`` rows at a time instead (peak ~``chunk*dim*4`` on device). The gathers are COLLECTIVE,
    so ALL ranks call this in lockstep; only ``keep`` (rank 0) assembles + returns the array (the
    others gather and discard, returning ``None``).
    """
    out = np.empty((num_nodes, dim), dtype=np.float32) if keep else None
    for start in range(0, num_nodes, chunk):
        ids = torch.arange(start, min(start + chunk, num_nodes), device="cuda")
        rows = _gather_emb(model, ids).detach().cpu().numpy().astype(np.float32)
        if keep:
            out[start : start + rows.shape[0]] = rows
    return out

