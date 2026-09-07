r"""Min-degree induced training slice + degree distribution over raw Common Crawl shards.

This is the preprocessing stage of the published embedding pipeline
(``wgl data degree-distribution`` and ``wgl data min-degree-slice``): stream the raw shards,
keep the well-connected core, and write a compact :class:`DirectedGraph` plus the
``orig_ids.npy`` join key back to the full host id-space.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wgl.data.cc_shards import (
    ShardSpec,
    _count_vertices,
    _induced_edge_counts,
    _parse_edge_shard,
    _read_edges,
    _read_vertices_for_ids,
    accumulate_degrees,
    accumulate_in_out_degrees,
    resolve_shards,
)
from wgl.data.graph import DirectedGraph
from wgl.logging import get_logger

logger = get_logger()

# The min-degree ladder reported by :func:`degree_distribution_report` — the thresholds the Phase-4
# "train on a min-degree subgraph, serve the tail via fallback" decision was made from (§7).
MIN_DEGREE_THRESHOLDS: tuple[int, ...] = (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

# Log-spaced degree-histogram edges (right-open); the last bin is open-ended.
_HISTOGRAM_EDGES: tuple[int, ...] = (0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025, 1 << 62)


def degree_distribution_report(
    edges: ShardSpec,
    num_nodes: int,
    *,
    edge_coverage: bool = False,
    thresholds: tuple[int, ...] = MIN_DEGREE_THRESHOLDS,
) -> dict:
    """Full-graph degree distribution + min-degree node/edge retention ladder (Phase-4 §7).

    Streams the raw edge shards once to accumulate every node's degree, then reports — for a ladder
    of min total-degree thresholds — how many nodes survive (the candidate training set), what
    fraction fall below (served via the fallback ladder), and the endpoint-incidence edge coverage.
    This is the measurement the ``--min-degree 8`` cut was chosen from.

    Args:
        edges: Edge shard spec (file, glob, directory, or list).
        num_nodes: Node count / id bound (from the graph's ``.stats``).
        edge_coverage: Run a second pass for the exact induced-subgraph edge count per threshold.
        thresholds: The min-degree ladder to report.

    Returns:
        The report payload: totals, the ``retention`` ladder, and a log-binned ``degree_histogram``.
    """
    logger.info("Pass 1/%d: degrees over the edge shards", 2 if edge_coverage else 1)
    indeg, outdeg, total_edges = accumulate_in_out_degrees(edges, num_nodes)
    deg = indeg + outdeg
    n_dangling = int((outdeg == 0).sum())
    endpoint_total = int(deg.sum())  # = 2 * total_edges

    rows = []
    for thr in thresholds:
        keep = deg >= thr
        n_keep = int(keep.sum())
        rows.append(
            {
                "min_degree": thr,
                "n_nodes": n_keep,
                "frac_nodes": round(n_keep / num_nodes, 5),
                "frac_fallback": round(1 - n_keep / num_nodes, 5),
                "endpoint_incidence": round(int(deg[keep].sum()) / max(endpoint_total, 1), 5),
            }
        )
        logger.info(
            "deg>=%d: %d nodes (%.2f%%), fallback %.2f%%, endpoint-incidence %.3f",
            thr,
            n_keep,
            100 * n_keep / num_nodes,
            100 * (1 - n_keep / num_nodes),
            rows[-1]["endpoint_incidence"],
        )

    hist, _ = np.histogram(deg, bins=list(_HISTOGRAM_EDGES))
    payload = {
        "num_nodes": num_nodes,
        "total_edges": total_edges,
        "n_dangling": n_dangling,
        "frac_dangling": round(n_dangling / num_nodes, 5),
        "mean_degree": round(endpoint_total / num_nodes, 4),
        "retention": rows,
        "degree_histogram": [
            {
                "deg_lo": int(_HISTOGRAM_EDGES[j]),
                "deg_hi": int(_HISTOGRAM_EDGES[j + 1] - 1),
                "n_nodes": int(hist[j]),
            }
            for j in range(len(hist))
        ],
    }

    if edge_coverage:
        logger.info("Pass 2/2: induced-subgraph edge counts (min endpoint degree)")
        induced = _induced_edge_counts(edges, deg, thresholds)
        for j in range(len(thresholds)):
            payload["retention"][j]["induced_edges"] = int(induced[j])
            payload["retention"][j]["frac_edges"] = round(int(induced[j]) / max(total_edges, 1), 5)
    return payload


def load_min_degree_subgraph(
    vertices: ShardSpec,
    edges: ShardSpec,
    min_degree: int,
    num_nodes: int | None = None,
    degree_cache: str | Path | None = None,
) -> tuple[DirectedGraph, np.ndarray, int]:
    """Build the **min-degree induced subgraph** — the Phase-4 training slice — from raw CC shards.

    Keeps every node whose *total* degree (in + out over the full graph) is ``>= min_degree`` and
    the edges between two kept nodes, remapping to compact ids ``0..M-1``. This is the §7 decision:
    train on a well-connected subgraph (the degree-1 sea is served by the fallback ladder), which
    both improves quality and - at ``min_degree`` in ~8-16 - shrinks the table to fit one GPU. The
    intermediate-slice bring-up is just a *higher* threshold (a smaller, cheaper cut) of the same
    code path.

    Memory scales with the **kept edge count** (like :func:`_load_edge_induced`): the endpoints are
    accumulated in RAM, then handed to :meth:`DirectedGraph.from_edges`. For the full deg≥8 cut
    (~11-12 B edges) this needs the out-of-core streaming writer (deferred, §3.1); for the
    intermediate slice pick ``min_degree`` high enough that the kept edges fit host RAM.

    Args:
        vertices: Vertex shard spec (``id<TAB>reversed-host``).
        edges: Edge shard spec (``from-id<TAB>to-id``).
        min_degree: Keep nodes with total degree ``>= min_degree``.
        num_nodes: Id bound / node count; counted from the vertex shards when omitted.
        degree_cache: Optional ``.npy`` to cache/reuse the (threshold-independent) degree vector.

    Returns:
        ``(graph, orig_ids, source_num_nodes)`` — the compact induced :class:`DirectedGraph`
        (node_names attached); ``orig_ids`` mapping ``compact_id -> original global CC node id``
        (int64, sorted); and the full source node count (the id space the fallback must cover).
        ``orig_ids`` is the join key that lets the release cover **all** source hosts: kept hosts
        get a trained vector, the dropped low-degree tail gets a fallback-ladder vector.
    """
    if min_degree < 1:
        msg = f"min_degree must be >= 1, got {min_degree}"
        raise ValueError(msg)
    vshards = resolve_shards(vertices)
    eshards = resolve_shards(edges)
    if num_nodes is None:
        num_nodes = _count_vertices(vshards)

    orig_ids, remap = _min_degree_keep_set(eshards, min_degree, num_nodes, degree_cache)
    edge_index = _read_edges(eshards, remap)
    names = _read_vertices_for_ids(vshards, remap, orig_ids.size)
    graph = DirectedGraph.from_edges(edge_index, num_nodes=orig_ids.size, node_names=names)
    logger.info(
        "Min-degree slice: %d nodes / %d induced edges (avg degree %.2f)",
        graph.num_nodes,
        graph.num_edges,
        2 * graph.num_edges / max(graph.num_nodes, 1),
    )
    return graph, orig_ids, num_nodes


def _min_degree_keep_set(
    eshards: list[Path], min_degree: int, num_nodes: int, degree_cache: str | Path | None
) -> tuple[np.ndarray, np.ndarray]:
    """Degree pass -> ``(orig_ids, remap)`` for the deg>=min_degree keep set (both build paths)."""
    deg = accumulate_degrees(eshards, num_nodes, degree_cache)
    orig_ids = np.flatnonzero(deg >= min_degree).astype(np.int64)
    if orig_ids.size == 0:
        max_deg = int(deg.max()) if deg.size else 0
        msg = f"min_degree={min_degree} keeps 0 nodes (max degree {max_deg})"
        raise ValueError(msg)
    remap = np.full(num_nodes, -1, dtype=np.int64)
    remap[orig_ids] = np.arange(orig_ids.size, dtype=np.int64)
    logger.info(
        "min_degree=%d: keep %d/%d nodes (%.2f%%), fallback %.2f%%",
        min_degree,
        orig_ids.size,
        num_nodes,
        100 * orig_ids.size / num_nodes,
        100 * (1 - orig_ids.size / num_nodes),
    )
    return orig_ids, remap


def _count_induced_edges(eshards: list[Path], remap: np.ndarray) -> int:
    """Stream edges once, counting those with **both** endpoints kept (remap >= 0)."""
    total = 0
    for shard in eshards:
        for src, dst in _parse_edge_shard(shard):
            total += int(((remap[src] >= 0) & (remap[dst] >= 0)).sum())
    return total


def _fill_induced_edges(eshards: list[Path], remap: np.ndarray, edge_index: np.ndarray) -> int:
    """Stream edges, writing each kept+remapped edge sequentially into ``edge_index`` (a memmap).

    Peak RAM is one parse chunk (not the whole edge list), so this scales to the full 13.4 B-edge
    graph — the point of the out-of-core path. Returns the number of edges written (== the count).
    """
    cursor = 0
    for shard in eshards:
        for src, dst in _parse_edge_shard(shard):
            ns, nd = remap[src], remap[dst]
            keep = (ns >= 0) & (nd >= 0)
            k = int(keep.sum())
            if k:
                edge_index[0, cursor : cursor + k] = ns[keep].astype(np.int32)
                edge_index[1, cursor : cursor + k] = nd[keep].astype(np.int32)
                cursor += k
    return cursor


def write_min_degree_slice(
    vertices: ShardSpec,
    edges: ShardSpec,
    min_degree: int,
    out_dir: str | Path,
    *,
    num_nodes: int | None = None,
    degree_cache: str | Path | None = None,
    build_csr: bool = True,
    csr_max_edges: int = 3_000_000_000,
) -> tuple[int, int, int]:
    """Out-of-core min-degree slice writer — the full-CC-scale path (§3.1), bounded RAM.

    Same keep set / output as :func:`load_min_degree_subgraph`, but never holds the induced edge
    list in RAM: it streams the kept edges into a disk-backed int32 ``edge_index`` memmap (peak
    RAM = one parse chunk), then writes ``graph.npz`` with a **streaming** ``np.savez`` (chunked
    copy out of the memmap). This lifts the in-memory path's edge ceiling, so the full deg>=8 cut
    (~11-12 B edges) can be prepped for training.

    Edge streams: degrees (cached) + a count pass (size the memmap) + a fill pass = up to 3; with a
    warm ``--degree-cache`` it's 2. The COO ``edge_index`` is all the cuGraph-PyG training path
    reads; the CSR views (for the wgl eval harness / ``DirectedGraph.load``) are built **in RAM**
    only when ``build_csr`` and the induced edges fit ``csr_max_edges`` — otherwise the artifact is
    COO-only (``has_csr=false`` in meta) and eval-at-scale uses the distributed path (§3.3).

    Args:
        vertices: Raw CC vertex shard spec.
        edges: Raw CC edge shard spec.
        min_degree: Keep nodes with total degree >= this.
        out_dir: Output directory (``graph.npz`` + ``orig_ids.npy`` + ``node_names.txt`` + meta).
        num_nodes: Id bound; counted from the vertex shards when omitted.
        degree_cache: Optional ``.npy`` to cache/reuse the (threshold-independent) degree vector.
        build_csr: Build + persist the CSR views when the edges fit ``csr_max_edges``.
        csr_max_edges: In-RAM CSR-build ceiling; above it the slice is written COO-only.

    Returns:
        ``(num_kept_nodes, num_edges, source_num_nodes)``.
    """
    import json as _json

    from numpy.lib.format import open_memmap

    vshards = resolve_shards(vertices)
    eshards = resolve_shards(edges)
    if num_nodes is None:
        num_nodes = _count_vertices(vshards)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_ids, remap = _min_degree_keep_set(eshards, min_degree, num_nodes, degree_cache)
    n_edges = _count_induced_edges(eshards, remap)
    logger.info("Out-of-core writer: %d induced edges -> memmap (bounded RAM)", n_edges)

    scratch = out_dir / "_edge_index.scratch.npy"
    edge_index = open_memmap(scratch, mode="w+", dtype=np.int32, shape=(2, n_edges))
    filled = _fill_induced_edges(eshards, remap, edge_index)
    if (
        filled != n_edges
    ):  # count and fill must see the same edges (shards unchanged between passes)
        msg = f"filled {filled} edges but counted {n_edges} (shards changed mid-build?)"
        raise RuntimeError(msg)
    edge_index.flush()
    names = _read_vertices_for_ids(vshards, remap, orig_ids.size)

    has_csr = build_csr and n_edges <= csr_max_edges
    if has_csr:
        # Small enough to build CSR in RAM (int64 upcast inside from_edges): a full DirectedGraph.
        graph = DirectedGraph.from_edges(
            np.asarray(edge_index), num_nodes=orig_ids.size, node_names=names
        )
        graph.save(out_dir)  # writes graph.npz (with CSR) + meta.json + node_names.txt
    else:
        # COO-only: stream the memmap into graph.npz (np.savez chunks large arrays, no full load).
        np.savez(out_dir / "graph.npz", edge_index=edge_index)
        (out_dir / "node_names.txt").write_text("\n".join(names))
    del edge_index  # release the memmap handle before renaming the scratch file
    # Keep the COO as a standalone ``edge_index.npy``. Only a real ``.npy`` can be memmapped:
    # ``np.load(..., mmap_mode="r")`` on an ``.npz`` MEMBER is silently ignored (NpzFile always
    # materializes the array in full), which would blow host RAM at full-CC scale. The training
    # path (``cupyg_lp.py --stream-split``) reads this file, not the npz member.
    scratch.replace(out_dir / "edge_index.npy")

    np.save(out_dir / "orig_ids.npy", orig_ids)
    meta = {
        "num_nodes": int(orig_ids.size),
        "num_edges": int(n_edges),
        "min_degree": int(min_degree),
        "source_num_nodes": int(num_nodes),
        "has_csr": bool(has_csr),
        "has_edge_index_npy": True,
        "source_vertices": str(vertices),
        "source_edges": str(edges),
    }
    (out_dir / "meta.json").write_text(_json.dumps(meta, indent=2))
    logger.info(
        "Out-of-core min-degree slice (deg>=%d): %d nodes / %d edges, has_csr=%s -> %s",
        min_degree,
        orig_ids.size,
        n_edges,
        has_csr,
        out_dir,
    )
    return orig_ids.size, n_edges, num_nodes
