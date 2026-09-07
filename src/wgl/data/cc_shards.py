r"""Shard-level primitives for Common Crawl host/domain web-graph text dumps.

The vertex/edge shards CC publishes are gzipped TSV; every ingest path in this package
(the release slice writer, the degree report, the sampling loaders) resolves and parses them
through the helpers here. Kept dependency-free of the higher-level loaders so that a minimal
reproduction checkout can ship this module plus :mod:`wgl.data.cc_min_degree` alone.
"""

from __future__ import annotations

import glob as globmod
import gzip
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from wgl.logging import get_logger

logger = get_logger()

# Knuth multiplicative-hash constant (fixed point of the golden ratio in 64-bit).
_HASH_MULT = 0x9E3779B97F4A7C15
_U64 = np.uint64
_MASK64 = 0xFFFFFFFFFFFFFFFF

ShardSpec = str | Path | list[str | Path]


def _open_text(path: Path):  # noqa: ANN202 - thin file-handle helper
    """Open a possibly-gzipped text file for reading."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("rt")


def resolve_shards(spec: ShardSpec) -> list[Path]:
    """Resolve a shard spec to a sorted list of files.

    Args:
        spec: A single file path, a glob pattern, a directory (its ``*.gz``/``*`` are used), or an
            explicit list of any of those.

    Returns:
        Sorted list of shard file paths.

    Raises:
        FileNotFoundError: If the spec resolves to no files.
    """
    if isinstance(spec, (list, tuple)):
        out: list[Path] = []
        for item in spec:
            out.extend(resolve_shards(item))
        return out
    text = str(spec)
    if any(ch in text for ch in "*?["):
        # glob.glob handles absolute patterns with wildcards in any position (Path.glob cannot).
        files = [Path(p) for p in sorted(globmod.glob(text))]  # noqa: PTH207
    else:
        path = Path(spec)
        if path.is_dir():
            files = sorted(path.glob("*.gz")) or sorted(p for p in path.glob("*") if p.is_file())
        else:
            files = [path]
    if not files:
        msg = f"No shard files matched: {spec!r}"
        raise FileNotFoundError(msg)
    return files


def _count_vertices(vshards: list[Path]) -> int:
    """Count vertex lines across shards (the id bound) when a ``.stats`` count isn't supplied."""
    total = 0
    for shard in vshards:
        with _open_text(shard) as fh:
            for line in fh:
                if line.strip():
                    total += 1
    return total


def accumulate_degrees(
    edges: ShardSpec, num_nodes: int, degree_cache: str | Path | None = None
) -> np.ndarray:
    """Stream every edge shard once and return per-node **total degree** (in + out), int64.

    The full-CC scan (13.4 B edges) is ~1.5-2 h, so an optional ``degree_cache`` ``.npy`` lets a
    later min-degree-threshold choice skip the pass entirely (the degrees don't depend on the
    threshold). Mirrors ``scripts/degree_distribution.py``'s accumulation.
    """
    if degree_cache is not None and Path(degree_cache).exists():
        logger.info("Loaded cached degrees from %s", degree_cache)
        return np.load(degree_cache)
    eshards = resolve_shards(edges)
    deg = np.zeros(num_nodes, dtype=np.int64)
    total = 0
    for i, shard in enumerate(eshards):
        for src, dst in _parse_edge_shard(shard):
            deg += np.bincount(src, minlength=num_nodes)
            deg += np.bincount(dst, minlength=num_nodes)
            total += src.shape[0]
        logger.info(
            "  degree pass shard %d/%d (%s): cum %d edges", i + 1, len(eshards), shard.name, total
        )
    if degree_cache is not None:
        np.save(degree_cache, deg)
        logger.info("Cached degrees -> %s", degree_cache)
    return deg


def accumulate_in_out_degrees(
    edges: ShardSpec, num_nodes: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Stream every edge shard once and return per-node in-degree, out-degree and the edge count.

    Kept separate from :func:`accumulate_degrees` (which returns only the *total* and supports a
    ``degree_cache``) because the distribution report needs the two directions apart — the dangling
    count is ``(out_degree == 0).sum()``, which a cached total cannot reconstruct. The cheaper
    single-accumulator version stays the one on the slice-building hot path.

    Args:
        edges: Edge shard spec (file, glob, directory, or list).
        num_nodes: Node count / id bound.

    Returns:
        ``(in_degree, out_degree, total_edges)``.
    """
    eshards = resolve_shards(edges)
    indeg = np.zeros(num_nodes, dtype=np.int64)
    outdeg = np.zeros(num_nodes, dtype=np.int64)
    total = 0
    for i, shard in enumerate(eshards):
        for src, dst in _parse_edge_shard(shard):
            outdeg += np.bincount(src, minlength=num_nodes)
            indeg += np.bincount(dst, minlength=num_nodes)
            total += src.shape[0]
        logger.info(
            "  degree pass shard %d/%d (%s): cum %d edges", i + 1, len(eshards), shard.name, total
        )
    return indeg, outdeg, total


def _induced_edge_counts(
    edges: ShardSpec, deg: np.ndarray, thresholds: tuple[int, ...]
) -> np.ndarray:
    """Second pass: exact induced-subgraph edge count per min-degree threshold.

    An edge survives the ``deg >= thr`` cut iff *both* endpoints do, i.e. iff
    ``min(deg[src], deg[dst]) >= thr``. Bucketing each edge by how many thresholds its
    min-endpoint-degree clears turns the per-threshold counts into one suffix sum.

    Args:
        edges: Edge shard spec.
        deg: Per-node total degree from pass 1.
        thresholds: The min-degree ladder.

    Returns:
        Induced edge count per threshold, shape ``(len(thresholds),)``.
    """
    eshards = resolve_shards(edges)
    hist = np.zeros(len(thresholds) + 1, dtype=np.int64)
    thr_arr = np.asarray(thresholds)
    for i, shard in enumerate(eshards):
        for src, dst in _parse_edge_shard(shard):
            med = np.minimum(deg[src], deg[dst])
            hist += np.bincount(
                np.searchsorted(thr_arr, med, side="right"), minlength=len(thresholds) + 1
            )
        logger.info("  edge-coverage pass shard %d/%d", i + 1, len(eshards))
    suffix = np.cumsum(hist[::-1])[::-1]
    return suffix[1:]


def _read_vertices_for_ids(shards: list[Path], remap: np.ndarray, n_kept: int) -> list[str]:
    """Stream vertex shards, materializing names only for kept endpoints (positioned by compact id).

    Node ids are the global line order across shards (CC ids are 0-based contiguous), matching
    :func:`_read_vertices`. ``remap[old_id]`` gives the compact id (or ``-1`` if the vertex is
    not an endpoint of any kept edge).
    """
    names: list[str | None] = [None] * n_kept
    max_old = remap.shape[0] - 1
    gid = 0
    for shard in shards:
        with _open_text(shard) as fh:
            for line in fh:
                if not line.strip():
                    continue
                if gid <= max_old and remap[gid] >= 0:
                    parts = line.rstrip("\n").split("\t")
                    names[remap[gid]] = parts[1] if len(parts) > 1 else parts[0]
                gid += 1
    # Every kept endpoint is an edge endpoint, so it must appear in the vertex shards.
    missing = [i for i, nm in enumerate(names) if nm is None]
    if missing:
        msg = f"{len(missing)} kept edge endpoint(s) had no vertex record (corrupt shards?)"
        raise ValueError(msg)
    return [nm for nm in names if nm is not None]


def _read_edges(shards: list[Path], remap: np.ndarray) -> np.ndarray:
    """Stream edge shards, remapping endpoints and dropping edges touching removed nodes."""
    kept_src: list[np.ndarray] = []
    kept_dst: list[np.ndarray] = []
    for shard in shards:
        for src, dst in _parse_edge_shard(shard):
            new_src, new_dst = remap[src], remap[dst]
            keep = (new_src >= 0) & (new_dst >= 0)
            if keep.any():
                kept_src.append(new_src[keep])
                kept_dst.append(new_dst[keep])
    if not kept_src:
        return np.zeros((2, 0), dtype=np.int64)
    return np.stack([np.concatenate(kept_src), np.concatenate(kept_dst)])


def _parse_edge_shard(
    shard: Path, chunk_lines: int = 8_000_000
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Parse an edge shard into ``(src, dst)`` int arrays in bounded-size chunks.

    Uses PyArrow's C++ multithreaded CSV reader when available (~10x faster than the pure-Python
    path on the multi-billion-edge full graph; the ``[cc]`` extra ships pyarrow). Falls back to a
    NumPy split-and-parse otherwise. Both yield identical ``(src, dst)`` int64 chunks.

    Args:
        shard: A ``from-id<TAB>to-id`` (optionally gzipped) shard.
        chunk_lines: Approximate lines per yielded chunk (caps peak memory per shard).

    Yields:
        ``(src, dst)`` int64 arrays for each chunk.
    """
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
    except ImportError:
        yield from _parse_edge_shard_numpy(shard, chunk_lines)
        return

    compression = "gzip" if shard.suffix == ".gz" else None
    read_opts = pacsv.ReadOptions(
        column_names=["src", "dst"],  # shards have no header
        block_size=chunk_lines * 24,  # ~24 bytes/line -> bounded per-batch memory
    )
    parse_opts = pacsv.ParseOptions(delimiter="\t")
    convert_opts = pacsv.ConvertOptions(column_types={"src": pa.int64(), "dst": pa.int64()})
    with pa.input_stream(str(shard), compression=compression) as stream:
        reader = pacsv.open_csv(
            stream, read_options=read_opts, parse_options=parse_opts, convert_options=convert_opts
        )
        for batch in reader:
            if batch.num_rows:
                src = batch.column(0).to_numpy(zero_copy_only=False)
                dst = batch.column(1).to_numpy(zero_copy_only=False)
                yield src, dst


def _parse_edge_shard_numpy(
    shard: Path, chunk_lines: int = 8_000_000
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """NumPy fallback for :func:`_parse_edge_shard` (no pyarrow)."""
    with _open_text(shard) as fh:
        while True:
            block = fh.readlines(chunk_lines * 24)  # ~24 bytes/line heuristic
            if not block:
                break
            flat = np.array("".join(block).split(), dtype=np.int64)  # splits on tabs + newlines
            if flat.size:
                pairs = flat.reshape(-1, 2)
                yield pairs[:, 0], pairs[:, 1]
