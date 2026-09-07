"""Deterministic, permutation-free 80/10/10 edge split + the precomputed split cache.

Hashes each edge by its global index so ranks agree without ever materializing a permutation of
the ~13B-edge list. ``load_split_cache`` reads the cache ``wgl data stream-split-cache`` wrote.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


# Knuth multiplicative hash (mirrors wgl.data.cc_split_cache / wgl.data.splits) for a
# deterministic, permutation-free 80/10/10 edge split.
_SPLIT_HASH_MULT = np.uint64(0x9E3779B97F4A7C15)



def load_split(
    graph_dir: Path, seed: int, max_edges: int = 0, induced: bool = False
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Load our directed graph and split edges 80/10/10 (train message+label / val / test).

    ``max_edges`` subsamples the first N edges for a fast validation run; ``induced`` then keeps
    only the involved nodes and remaps to 0..M-1 (density-preserving, matching gs_export_graph
    --induced), so the small graph stays representative for neighbour sampling, not near-isolated.
    """
    # Slice path: load the whole COO into RAM. (mmap_mode is a no-op here: np.load ignores it for an
    # .npz MEMBER; the full graph uses --stream-split, which memmaps a standalone .npy.)
    with np.load(graph_dir / "graph.npz") as npz:
        edge_index = np.ascontiguousarray(npz["edge_index"]).astype(np.int64)
    if max_edges and edge_index.shape[1] > max_edges:
        edge_index = np.ascontiguousarray(edge_index[:, :max_edges])
    meta = graph_dir / "meta.json"
    if induced:
        involved = np.unique(edge_index)
        edge_index = np.searchsorted(involved, edge_index).astype(np.int64)
        num_nodes = int(involved.shape[0])
    elif meta.exists():
        num_nodes = int(json.loads(meta.read_text())["num_nodes"])
    else:
        num_nodes = int(edge_index.max()) + 1
    n_edges = edge_index.shape[1]
    perm = np.random.default_rng(seed).permutation(n_edges)
    n_hold = n_edges // 10
    test = edge_index[:, perm[:n_hold]]
    val = edge_index[:, perm[n_hold : 2 * n_hold]]
    train = edge_index[:, perm[2 * n_hold :]]
    print(
        f"nodes={num_nodes:,} edges={n_edges:,} -> train={train.shape[1]:,} "
        f"val={val.shape[1]:,} test={test.shape[1]:,}",
        flush=True,
    )
    return num_nodes, train, val, test


# Knuth multiplicative hash (mirrors wgl.data.cc_split_cache / wgl.data.splits) for a deterministic,
# streaming, permutation-free split. Splitting by edge *index* means each edge's fate is a pure
# function of its position + seed — no O(E) permutation, no full materialization.



def _split_buckets(start: int, count: int, seed: int) -> np.ndarray:
    """Deterministic split bucket in ``[0, 10)`` for edge indices ``[start, start+count)``.

    ``0`` -> test, ``1`` -> val, ``>=2`` -> train (an 80/10/10 split). Vectorized over the chunk.
    """
    idx = (np.arange(start, start + count, dtype=np.uint64) + np.uint64(seed)) * _SPLIT_HASH_MULT
    return (idx % np.uint64(10)).astype(np.int64)



def _eval_keep(global_idx: np.ndarray, keep_frac: float, seed: int) -> np.ndarray:
    """Deterministic ~``keep_frac`` subsample mask over held-out edges, by hash of their index.

    Uses the HIGH bits of the multiplicative hash (well-mixed, unlike the low-bit ``mod`` the split
    uses) so the kept fraction is ~uniform and independent of index parity. Index-keyed => identical
    on every rank + chunk-invariant, so val/test stay replicated across ranks after subsampling.
    """
    if keep_frac >= 1.0:
        return np.ones(global_idx.shape[0], dtype=bool)
    h = (global_idx.astype(np.uint64) + np.uint64(seed)) * _SPLIT_HASH_MULT
    frac = (h >> np.uint64(40)).astype(np.float64) / float(1 << 24)  # top 24 bits -> [0, 1)
    return frac < keep_frac



def _open_edge_index(graph_dir: Path) -> np.ndarray:
    """Return the COO ``edge_index`` as a TRUE memmap (never materialized in full).

    Reads the standalone ``edge_index.npy`` (not the ``graph.npz`` member): ``np.load``'s
    ``mmap_mode`` is silently ignored for an ``.npz`` member, so loading through the zip would pull
    the entire ~105 GB array into host RAM per rank. Run ``wgl data extract-edge-index`` once on any
    slice written before the writer emitted this file.
    """
    npy = graph_dir / "edge_index.npy"
    if not npy.exists():
        msg = (
            f"{npy} missing. The COO edge_index must be a standalone .npy for a true memmap "
            "(np.load ignores mmap_mode on an .npz member -> full array in RAM per rank). "
            f"Run once: wgl data extract-edge-index --graph {graph_dir}"
        )
        raise FileNotFoundError(msg)
    arr = np.load(npy, mmap_mode="r")
    if not isinstance(arr, np.memmap):  # guard against a silent regression to full-load
        msg = f"{npy} did not memmap (got {type(arr).__name__}); refusing to load ~105 GB in RAM"
        raise RuntimeError(msg)
    return arr



def stream_split_and_shard(
    graph_dir: Path,
    seed: int,
    world_size: int,
    global_rank: int,
    max_edges: int = 0,
    chunk: int = 50_000_000,
    eval_cap: int = 0,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Memory-bounded 80/10/10 split + **per-rank train shard** for the full-CC cut (§3.1/§3.2).

    ``load_split`` can't scale to the full graph: ``np.random.permutation(13.4e9)`` alone needs
    ~107 GB and it materializes the whole int64 edge_index (~214 GB). Here the on-disk int32
    standalone ``edge_index.npy`` is **mmapped** (via :func:`_open_edge_index` -- NOT the graph.npz
    member, whose ``mmap_mode`` np.load silently ignores) and streamed in ``chunk``-column passes;
    each edge's split is a hash of its **index** (no permutation), and train edges are striped by
    train ordinal. So every rank holds only **its own** train shard (~1/world_size of 80 %) plus
    val/test — never the whole edge list. This shard is used BOTH as the rank's ``GraphStore``
    partition (each rank ingests its own edges; cuGraph redistributes) and as its loader
    ``edge_label_index``; the union over ranks is exactly the leakage-safe train set (val/test out).

    Assumes an already-compact graph (e.g. a ``wgl data min-degree-slice`` output); ``--induced``
    node remapping is not applied here. Returns ``(num_nodes, train_shard, val, test)`` (int64 COO).
    With ``eval_cap > 0``, val/test are **subsampled inside the streaming loop** (deterministic,
    identical on every rank) to ~``1.5 * eval_cap`` each, so they never materialize in full: at
    full-CC the untrimmed splits are ~1.3 B edges each (~21 GB/rank). ``eval_cap = 0`` keeps them
    whole (the validated slice behaviour).
    """
    meta = graph_dir / "meta.json"
    edge_index = _open_edge_index(graph_dir)  # TRUE memmap; chunks below load one slice at a time
    n_edges = int(edge_index.shape[1])
    if max_edges and n_edges > max_edges:
        n_edges = max_edges
    num_nodes = (
        int(json.loads(meta.read_text())["num_nodes"])
        if meta.exists()
        else int(np.asarray(edge_index[:, :n_edges]).max()) + 1
    )
    # Keep ~1.5x the cap so the post-hoc exact-cap trim (main's _subsample_eval) has slack; each
    # bucket is ~n_edges/10, so keep_frac targets eval_cap out of that expected size.
    expected_split = max(1, n_edges // 10)
    keep_frac = 1.0 if eval_cap <= 0 else min(1.0, 1.5 * eval_cap / expected_split)
    train_cols: list[np.ndarray] = []
    val_cols: list[np.ndarray] = []
    test_cols: list[np.ndarray] = []
    train_ordinal = 0
    # The full-CC split streams ~260 chunks over the ~105 GB memmap (~30 min, no GPU engaged yet).
    # Log periodic progress on rank 0 so this long, otherwise-silent startup is distinguishable
    # from a hang (and gives a rough read on split throughput before training even begins).
    n_chunks = (n_edges + chunk - 1) // chunk
    log_every_c = max(1, n_chunks // 20)
    t0 = time.monotonic()
    tr_kept = va_kept = te_kept = 0
    for ci, start in enumerate(range(0, n_edges, chunk)):
        stop = min(start + chunk, n_edges)
        cols = np.asarray(edge_index[:, start:stop]).astype(np.int64)  # one chunk in RAM
        buckets = _split_buckets(start, stop - start, seed)
        test_pos = np.flatnonzero(buckets == 0)
        val_pos = np.flatnonzero(buckets == 1)
        if keep_frac < 1.0:  # subsample held-out edges in-loop (never materialize the full split)
            test_pos = test_pos[_eval_keep(start + test_pos, keep_frac, seed + 7)]
            val_pos = val_pos[_eval_keep(start + val_pos, keep_frac, seed + 11)]
        test_cols.append(cols[:, test_pos])
        val_cols.append(cols[:, val_pos])
        tcols = cols[:, buckets >= 2]
        # Stripe this chunk's train edges by GLOBAL train ordinal so ranks partition train.
        ordinals = train_ordinal + np.arange(tcols.shape[1])
        kept = tcols[:, ordinals % world_size == global_rank]
        train_cols.append(kept)
        train_ordinal += tcols.shape[1]
        tr_kept, va_kept, te_kept = (
            tr_kept + kept.shape[1],
            va_kept + val_pos.size,
            te_kept + test_pos.size,
        )
        if global_rank == 0 and (ci % log_every_c == 0 or stop == n_edges):
            el = time.monotonic() - t0
            print(
                f"[stream-split] rank 0 {stop / n_edges:5.1%} (chunk {ci + 1}/{n_chunks}) "
                f"train_shard={tr_kept:,} val={va_kept:,} test={te_kept:,} {el:,.0f}s",
                flush=True,
            )

    def _cat(parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts, axis=1) if parts else np.zeros((2, 0), dtype=np.int64)

    train, val, test = _cat(train_cols), _cat(val_cols), _cat(test_cols)
    print(
        f"[stream-split] rank {global_rank}/{world_size} nodes={num_nodes:,} edges={n_edges:,} "
        f"-> train_shard={train.shape[1]:,} (of ~{train_ordinal:,}) val={val.shape[1]:,} "
        f"test={test.shape[1]:,}",
        flush=True,
    )
    return num_nodes, train, val, test



def load_split_cache(
    cache_dir: Path, world_size: int, global_rank: int, eval_cap: int = 0
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Load a precomputed split (``wgl data stream-split-cache``) -- fast, re-hash-free startup.

    Replaces :func:`stream_split_and_shard` when ``--split-cache-dir`` is set. The in-line split has
    every rank re-read + re-hash the whole ~105 GB ``edge_index.npy`` on every (re)launch (~1 h,
    cache-thrashing, GPU-idle at full CC, paid again on each 24 h-cap resume). Here each rank
    instead memmaps ``train.npy`` and reads only its CONTIGUOUS ``1/world_size`` slice of the train
    columns -- disjoint across ranks (no redundant scan, no re-hash, no thrash) -- plus the small
    shared val/test. The union of the per-rank slices is exactly the cached train set; cuGraph
    redistributes the partition, so a contiguous shard is equivalent to the in-line ordinal-striped
    one for sampling. ``train.npy`` is ``[2, n_edges]`` with only ``[:, :train_total]`` valid (tail
    slack), so the slice bounds come from ``split_meta.json``, never the file's column count.
    """
    import json as _json

    cache_dir = Path(cache_dir)
    meta = _json.loads((cache_dir / "split_meta.json").read_text())
    num_nodes, train_total = int(meta["num_nodes"]), int(meta["train_total"])
    train_mm = np.load(cache_dir / "train.npy", mmap_mode="r")
    if not isinstance(train_mm, np.memmap):  # guard against a silent full-load regression
        msg = f"{cache_dir / 'train.npy'} did not memmap (got {type(train_mm).__name__})"
        raise RuntimeError(msg)
    lo = train_total * global_rank // world_size
    hi = train_total * (global_rank + 1) // world_size
    train_shard = np.ascontiguousarray(train_mm[:, lo:hi])  # this rank's contiguous train partition
    val, test = np.load(cache_dir / "val.npy"), np.load(cache_dir / "test.npy")
    cached_eval = int(meta.get("eval_edges", 0))
    if eval_cap and cached_eval and eval_cap > cached_eval and global_rank == 0:
        print(
            f"[split-cache] --eval-edges {eval_cap:,} > cached {cached_eval:,}; using cached "
            f"val={val.shape[1]:,}/test={test.shape[1]:,} (re-cache for more).",
            flush=True,
        )
    print(
        f"[split-cache] rank {global_rank}/{world_size} nodes={num_nodes:,} "
        f"train_shard={train_shard.shape[1]:,} (of {train_total:,}) "
        f"val={val.shape[1]:,} test={test.shape[1]:,}",
        flush=True,
    )
    return num_nodes, train_shard, val, test



def _subsample_eval(edges: np.ndarray, max_edges: int, seed: int) -> np.ndarray:
    """Fixed random subsample of held-out edges for a bounded periodic MRR (§3.3).

    Full-CC val/test are each ~10 % of ~13 B edges (~1.3 B), so ranking every one of them every
    eval is infeasible. Cap to a deterministic (seeded) sample so the MRR trajectory stays
    comparable across epochs and identical across ranks (eval is replicated). Samples with
    collision + ``unique`` (O(max_edges), never allocates O(num_edges)); ``0`` or a set already
    within the cap is returned unchanged, so the validated slice path is untouched.
    """
    n = edges.shape[1]
    if max_edges <= 0 or n <= max_edges:
        return edges
    rng = np.random.default_rng(seed)
    cols = np.unique(rng.integers(0, n, size=max_edges, dtype=np.int64))
    return np.ascontiguousarray(edges[:, cols])


# ── train ────────────────────────────────────────────────────────────────────────────────────

