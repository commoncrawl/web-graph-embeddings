r"""Memmap-able edge index + the precomputed deterministic edge split cache.

Steps 3 and 4 of the reproduction runbook: pull ``graph.npz``'s COO into a standalone
``edge_index.npy`` the trainer can memory-map per rank, then hash the 80/10/10 edge split
once into ``split_cache/`` so each rank slices a contiguous shard at launch instead of
re-hashing the whole edge list on every (re)start.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from wgl.logging import get_logger

logger = get_logger()


def extract_edge_index_npy(
    graph_dir: str | Path, *, member: str = "edge_index", out_name: str = "edge_index.npy"
) -> tuple[Path, tuple[int, ...]]:
    """Extract an ``.npz`` member into a standalone, memmap-able ``.npy`` (bounded RAM).

    A one-time fix-up for slices written before ``write_min_degree_slice`` emitted
    ``edge_index.npy`` directly: ``np.savez`` stores each member as an uncompressed (ZIP_STORED)
    ``.npy`` inside the zip, so the member can be **byte-streamed** out with ``zipfile`` (peak RAM
    = one copy buffer) into a real ``.npy`` that ``np.load(..., mmap_mode="r")`` can memmap. The
    original ``graph.npz`` is left untouched for provenance.

    Args:
        graph_dir: Slice directory holding ``graph.npz``.
        member: Array name inside the npz (``np.savez(edge_index=...)`` -> member ``edge_index``).
        out_name: Output filename written next to ``graph.npz``.

    Returns:
        ``(path, shape)`` of the extracted standalone array.
    """
    import shutil
    import zipfile

    graph_dir = Path(graph_dir)
    dst = graph_dir / out_name
    npz = graph_dir / "graph.npz"
    if not npz.exists():
        msg = f"{npz} not found"
        raise FileNotFoundError(msg)
    arc = f"{member}.npy"
    with zipfile.ZipFile(npz) as zf:
        names = zf.namelist()
        if arc not in names:
            arc = next((n for n in names if n.startswith(member)), None)
            if arc is None:
                msg = f"member '{member}' not in {npz} (has {names})"
                raise KeyError(msg)
        with zf.open(arc) as src, dst.open("wb") as out:
            shutil.copyfileobj(src, out, length=64 * 1024 * 1024)  # 64 MB streaming copy
    arr = np.load(dst, mmap_mode="r")
    if not isinstance(arr, np.memmap):
        msg = f"{dst} did not memmap after extraction (got {type(arr).__name__})"
        raise RuntimeError(msg)
    logger.info("Extracted %s -> %s %s (memmap-able)", arc, dst, tuple(arr.shape))
    return dst, tuple(arr.shape)


# ── offline split cache (fast, resume-cheap trainer startup) ─────────────────────────────────────
# Mirror the trainer's split hash (scripts/cupyg_lp.py `_split_buckets`/`_eval_keep`) byte-for-byte
# so a cached split is identical to the in-line one. The two live in separate images (wgl vs cupyg,
# which has no wgl), so the logic is intentionally duplicated -- a test guards against drift
# against drift.
_SPLIT_HASH_MULT = np.uint64(0x9E3779B97F4A7C15)


def _split_buckets(start: int, count: int, seed: int) -> np.ndarray:
    """Deterministic split bucket in ``[0, 10)`` for edge indices ``[start, start+count)``.

    ``0`` -> test, ``1`` -> val, ``>=2`` -> train (an 80/10/10 split).
    """
    idx = (np.arange(start, start + count, dtype=np.uint64) + np.uint64(seed)) * _SPLIT_HASH_MULT
    return (idx % np.uint64(10)).astype(np.int64)


def _eval_keep(global_idx: np.ndarray, keep_frac: float, seed: int) -> np.ndarray:
    """~``keep_frac`` subsample mask over held-out edges, via the HIGH bits of the index hash."""
    if keep_frac >= 1.0:
        return np.ones(global_idx.shape[0], dtype=bool)
    h = (global_idx.astype(np.uint64) + np.uint64(seed)) * _SPLIT_HASH_MULT
    frac = (h >> np.uint64(40)).astype(np.float64) / float(1 << 24)  # top 24 bits -> [0, 1)
    return frac < keep_frac


def write_split_cache(
    graph_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    seed: int = 42,
    eval_edges: int = 2_000_000,
    chunk: int = 50_000_000,
) -> dict:
    """Precompute the deterministic 80/10/10 edge split ONCE and cache it to disk (bounded RAM).

    The trainer's in-line ``stream_split_and_shard`` re-reads the whole ~105 GB ``edge_index.npy``
    on EVERY rank and EVERY (re)launch, then hashes it -- at full CC a ~1 h, cache-thrashing,
    GPU-idle startup paid again on every 24 h-cap resume. This does the read+hash exactly once,
    single-process, and writes a train ``.npy`` (in edge order) plus subsampled val/test that the
    trainer memmap-slices per rank (``--split-cache-dir``), so a relaunch's startup drops to a
    contiguous ~1/world_size read with no re-hash.

    Train is written straight into a ``[2, n_edges]`` memmap; only ``[:, :train_total]`` is valid
    (the tail is unused slack -- avoids a two-pass count for the exact train size). Val/test are
    subsampled to ~``eval_edges`` each (the trainer's ``--eval-edges`` must be <= this) and saved
    whole; they are tiny and identical on every rank. Bounded RAM: one ``chunk`` of edges plus the
    subsampled val/test, never the full edge list.

    Returns the ``split_meta`` dict (also written to ``<out_dir>/split_meta.json``).
    """
    import json as _json

    from numpy.lib.format import open_memmap

    graph_dir = Path(graph_dir)
    out_dir = Path(out_dir) if out_dir is not None else graph_dir / "split_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    npy = graph_dir / "edge_index.npy"
    if not npy.exists():
        msg = f"{npy} missing; run `wgl data extract-edge-index --graph {graph_dir}` first"
        raise FileNotFoundError(msg)
    edge_index = np.load(npy, mmap_mode="r")
    if not isinstance(edge_index, np.memmap):
        msg = f"{npy} did not memmap (got {type(edge_index).__name__})"
        raise RuntimeError(msg)
    n_edges = int(edge_index.shape[1])

    meta_path = graph_dir / "meta.json"
    num_nodes = (
        int(_json.loads(meta_path.read_text())["num_nodes"])
        if meta_path.exists()
        else int(np.asarray(edge_index).max()) + 1
    )

    # ~1.5x the cap so the trainer's exact-cap trim keeps slack; each held-out bucket is ~n_edges/10
    expected_split = max(1, n_edges // 10)
    keep_frac = 1.0 if eval_edges <= 0 else min(1.0, 1.5 * eval_edges / expected_split)

    train_mm = open_memmap(out_dir / "train.npy", mode="w+", dtype=np.int32, shape=(2, n_edges))
    val_cols: list[np.ndarray] = []
    test_cols: list[np.ndarray] = []
    cursor = 0
    n_chunks = (n_edges + chunk - 1) // chunk
    t0 = time.monotonic()
    for ci, start in enumerate(range(0, n_edges, chunk)):
        stop = min(start + chunk, n_edges)
        cols = np.asarray(edge_index[:, start:stop]).astype(np.int32)  # one chunk in RAM
        buckets = _split_buckets(start, stop - start, seed)
        test_pos = np.flatnonzero(buckets == 0)
        val_pos = np.flatnonzero(buckets == 1)
        if keep_frac < 1.0:  # subsample held-out edges in-loop (never materialize the full split)
            test_pos = test_pos[_eval_keep(start + test_pos, keep_frac, seed + 7)]
            val_pos = val_pos[_eval_keep(start + val_pos, keep_frac, seed + 11)]
        test_cols.append(cols[:, test_pos])
        val_cols.append(cols[:, val_pos])
        tcols = cols[:, buckets >= 2]
        k = tcols.shape[1]
        train_mm[:, cursor : cursor + k] = tcols  # write straight to the disk-backed train memmap
        cursor += k
        if ci % max(1, n_chunks // 20) == 0 or stop == n_edges:
            logger.info(
                "split-cache %5.1f%% (chunk %d/%d) train=%d val=%d test=%d %.0fs",
                100 * stop / n_edges,
                ci + 1,
                n_chunks,
                cursor,
                sum(c.shape[1] for c in val_cols),
                sum(c.shape[1] for c in test_cols),
                time.monotonic() - t0,
            )
    train_mm.flush()
    del train_mm  # release the memmap handle
    train_total = cursor
    val = np.concatenate(val_cols, axis=1) if val_cols else np.zeros((2, 0), np.int32)
    test = np.concatenate(test_cols, axis=1) if test_cols else np.zeros((2, 0), np.int32)
    np.save(out_dir / "val.npy", val)
    np.save(out_dir / "test.npy", test)

    split_meta = {
        "num_nodes": num_nodes,
        "n_edges": n_edges,
        "train_total": int(train_total),
        "val_count": int(val.shape[1]),
        "test_count": int(test.shape[1]),
        "seed": seed,
        "eval_edges": eval_edges,
        "chunk": chunk,
    }
    (out_dir / "split_meta.json").write_text(_json.dumps(split_meta, indent=2))
    logger.info(
        "Wrote split cache -> %s (train=%d val=%d test=%d; train.npy slack=%d cols)",
        out_dir,
        train_total,
        val.shape[1],
        test.shape[1],
        n_edges - train_total,
    )
    return split_meta
