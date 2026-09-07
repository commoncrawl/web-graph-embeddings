"""Data commands for the published embedding pipeline.

Steps 2-4 of the reproduction runbook: the degree ladder the min-degree cut was chosen from,
the streaming induced-subgraph writer, the memmap-able edge index, and the split cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from wgl.cli._app import app
from wgl.logging import get_logger, setup_logging

data_app = typer.Typer(add_completion=False, help="Data preparation commands.")
app.add_typer(data_app, name="data")


@data_app.command("min-degree-slice")
def data_min_degree_slice(
    vertices: str = typer.Option(..., help="Vertex shards: file, glob, or directory."),
    edges: str = typer.Option(..., help="Edge shards: file, glob, or directory."),
    out: Path = typer.Option(..., help="Output directory for the min-degree training slice."),
    min_degree: int = typer.Option(
        8, help="Keep nodes with total (in+out) degree >= this; the rest are served by fallback."
    ),
    num_nodes: int = typer.Option(
        0, help="Id bound / node count (0 = count vertex lines). Full CC = 279356058."
    ),
    degree_cache: str = typer.Option(
        "", help="Optional .npy to cache/reuse the per-node degree vector across threshold choices."
    ),
    streaming: bool = typer.Option(
        False,
        help="Out-of-core writer (bounded RAM via a disk memmap) for edge counts that don't fit "
        "host RAM, e.g. the full deg>=8 cut (~11-12B edges). In-memory otherwise.",
    ),
    csr_max_edges: int = typer.Option(
        3_000_000_000,
        help="Streaming mode: build CSR in RAM only up to this many edges; above it, COO-only.",
    ),
) -> None:
    """Build the Phase-4 min-degree induced training slice from raw CC shards (§7).

    Keeps well-connected nodes (deg >= min_degree), drops the degree-1 sea (served via the fallback
    ladder), and writes the compact graph plus ``orig_ids.npy`` (compact_id -> global CC id) — the
    join key that lets the release still cover every source host. Use ``--streaming`` for the
    full-CC cut whose induced edges don't fit RAM.
    """
    setup_logging()
    logger = get_logger()
    import json as _json

    from wgl.data.cc_min_degree import load_min_degree_subgraph, write_min_degree_slice

    if streaming:
        n_nodes, n_edges, _ = write_min_degree_slice(
            vertices,
            edges,
            min_degree,
            out,
            num_nodes=num_nodes or None,
            degree_cache=degree_cache or None,
            csr_max_edges=csr_max_edges,
        )
        logger.info(
            "Min-degree slice [streaming] (deg>=%d): %d nodes / %d edges -> %s (+ orig_ids.npy)",
            min_degree,
            n_nodes,
            n_edges,
            out,
        )
        return

    graph, orig_ids, source_num_nodes = load_min_degree_subgraph(
        vertices,
        edges,
        min_degree,
        num_nodes=num_nodes or None,
        degree_cache=degree_cache or None,
    )
    graph.save(out)
    np.save(out / "orig_ids.npy", orig_ids)
    # Augment the graph.save() meta with the slice provenance (source graph + threshold), so the
    # release/fallback step and re-runs know exactly which cut this is.
    meta_path = out / "meta.json"
    meta = _json.loads(meta_path.read_text())
    meta.update(
        min_degree=min_degree,
        source_num_nodes=source_num_nodes,
        source_vertices=str(vertices),
        source_edges=str(edges),
    )
    meta_path.write_text(_json.dumps(meta, indent=2))
    logger.info(
        "Min-degree slice (deg>=%d): %d nodes / %d edges -> %s (+ orig_ids.npy)",
        min_degree,
        graph.num_nodes,
        graph.num_edges,
        out,
    )


@data_app.command("extract-edge-index")
def data_extract_edge_index(
    graph: Path = typer.Option(..., help="Slice dir with graph.npz (min-degree-slice output)."),
) -> None:
    """Extract graph.npz's edge_index into a standalone memmap-able edge_index.npy (bounded RAM).

    One-time fix-up for slices written before the writer emitted edge_index.npy: the cuGraph
    ``--stream-split`` training path needs a real ``.npy`` (np.load's ``mmap_mode`` is ignored for
    an ``.npz`` member, which would materialize the whole ~105 GB COO in RAM per rank).
    """
    setup_logging()
    logger = get_logger()
    from wgl.data.cc_split_cache import extract_edge_index_npy

    path, shape = extract_edge_index_npy(graph)
    logger.info("Wrote %s %s", path, shape)


@data_app.command("stream-split-cache")
def data_stream_split_cache(
    graph: Path = typer.Option(..., help="Slice dir with edge_index.npy (run extract-edge-index)."),
    out: Path = typer.Option(
        None,
        help="Cache dir (default <graph>/split_cache); pass to the trainer's --split-cache-dir.",
    ),
    seed: int = typer.Option(42, help="Split seed; must match the trainer's --seed."),
    eval_edges: int = typer.Option(
        2_000_000, help="Subsample val/test to ~this many each; trainer --eval-edges must be <=."
    ),
) -> None:
    """Precompute the deterministic 80/10/10 edge split ONCE so trainer startup is fast + resumable.

    The trainer's in-line ``--stream-split`` re-reads and re-hashes the whole ~105 GB edge_index on
    every rank and every (re)launch (~1 h GPU-idle, cache-thrashing at full CC, paid again on each
    24 h-cap resume). This runs that read+hash once, single-process, writing train.npy + subsampled
    val/test the trainer memmap-slices per rank (``--split-cache-dir``) -- relaunch startup becomes
    a contiguous ~1/world_size read with no re-hash. A CPU-only, one-time job.
    """
    setup_logging()
    logger = get_logger()
    from wgl.data.cc_split_cache import write_split_cache

    meta = write_split_cache(graph, out, seed=seed, eval_edges=eval_edges)
    logger.info("split cache: %s", meta)


@data_app.command("degree-distribution")
def data_degree_distribution(
    edges: str = typer.Option(..., help="Raw edge shards: file, glob, or directory."),
    num_nodes: int = typer.Option(..., help="Node count / id bound (from the graph's .stats)."),
    out: Path = typer.Option(..., help="Output JSON report path."),
    edge_coverage: bool = typer.Option(
        False, help="Second pass: exact induced-subgraph edge count (both endpoints >= thr)."
    ),
) -> None:
    """Report the full-graph degree distribution + min-degree retention ladder (Phase-4 §7).

    Streams the raw shards once and answers "train on a min-degree subgraph, serve the rest via
    fallback": per threshold, how many nodes survive, what fraction falls to the fallback ladder,
    and how much of the edge mass is retained. Run this before ``wgl data min-degree-slice`` to
    choose its ``--min-degree``.
    """
    setup_logging()
    logger = get_logger()
    from wgl.data.cc_min_degree import degree_distribution_report
    from wgl.eval.report import write_json_report

    payload = degree_distribution_report(edges, num_nodes, edge_coverage=edge_coverage)
    write_json_report(out, payload)
    logger.info(
        "Degree distribution: %d nodes / %d edges (%.2f%% dangling) -> %s",
        payload["num_nodes"],
        payload["total_edges"],
        100 * payload["frac_dangling"],
        out,
    )
