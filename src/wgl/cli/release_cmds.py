"""Release artifact commands: identity manifest, sizing, and the Parquet shard export.

``wgl release shard-export`` is the final stage of the published pipeline -- it turns the trained
``.npy`` table into the checksummed Parquet shard set uploaded to the Hugging Face hub.
"""

from __future__ import annotations

from pathlib import Path

import typer

from wgl.cli._app import app
from wgl.data.graph import DirectedGraph
from wgl.logging import get_logger, setup_logging

release_app = typer.Typer(add_completion=False, help="Release artifact identity/manifest commands.")
app.add_typer(release_app, name="release")


@release_app.command("manifest")
def release_manifest(
    graph: Path = typer.Option(..., help="Directory holding a prepared graph (with node_names)."),
    out: Path = typer.Option(..., help="Output directory for the release manifest."),
    release_id: str = typer.Option(..., help="Stable release id, e.g. cc-main-2026-mar-apr-may."),
    source_crawls: str = typer.Option(
        ..., help="Comma-separated constituent crawl ids (1-3 under the rolling-window policy)."
    ),
    reversed_host: bool = typer.Option(
        True, help="Node names are in Common Crawl reversed notation (com.example.www)."
    ),
) -> None:
    """Emit a release manifest (stable host/PLD keys + evidence) from a prepared graph.

    Writes ``manifest.json`` (release header) + ``hosts.tsv`` (per-host keys, local node id,
    degrees, cohort, confidence). The host/PLD keys are stable across releases; the id is not.
    """
    setup_logging()
    logger = get_logger()
    from wgl.release.manifest import ReleaseManifest

    crawls = [c.strip() for c in source_crawls.split(",") if c.strip()]
    if not 1 <= len(crawls) <= 3:
        msg = "source_crawls must list 1-3 crawl ids (rolling-window policy)."
        raise typer.BadParameter(msg)
    g = DirectedGraph.load(graph)
    manifest = ReleaseManifest.from_graph(
        g, release_id=release_id, source_crawls=crawls, reversed_host=reversed_host
    )
    manifest.save(out)
    logger.info(
        "Release manifest %s: %d hosts (%d crawls) -> %s",
        release_id,
        len(manifest),
        len(crawls),
        out,
    )


@release_app.command("sizing")
def release_sizing(
    host_nodes: int = typer.Option(
        None, help="Host node count (defaults to the cc-main-2026-mar-apr-may host count)."
    ),
    domain_nodes: int = typer.Option(
        None, help="Domain/PLD node count (defaults to the planning estimate)."
    ),
    dims: str = typer.Option("64,128,256", help="Comma-separated candidate dimensions."),
    precisions: str = typer.Option("fp32,fp16,int8", help="Comma-separated precisions."),
) -> None:
    """Print artifact size estimates (GiB) across dimensions/precisions for host and domain."""
    setup_logging()
    logger = get_logger()
    from wgl.release.sizing import (
        CC_DOMAIN_NODES_ESTIMATE,
        CC_HOST_NODES,
        format_table,
        size_table,
    )

    dim_tuple = tuple(int(d) for d in dims.split(",") if d.strip())
    prec_tuple = tuple(p.strip() for p in precisions.split(",") if p.strip())
    hn = host_nodes if host_nodes is not None else CC_HOST_NODES
    dn = domain_nodes if domain_nodes is not None else CC_DOMAIN_NODES_ESTIMATE
    for label, n in (("HOST", hn), ("DOMAIN", dn)):
        table = format_table(size_table(n, dims=dim_tuple, precisions=prec_tuple), label)
        for line in table.splitlines():
            logger.info("%s", line)


@release_app.command("shard-export")
def release_shard_export(
    emb: Path = typer.Option(..., help="Exported [N, dim] float32 .npy embedding table."),
    node_names: Path = typer.Option(
        ..., "--node-names", help="node_names.txt (one reversed CC host per row, aligned to emb)."
    ),
    out: Path = typer.Option(..., help="Output directory for the shard set(s) + manifest.json."),
    release_id: str = typer.Option(..., help="Stable release id (the crawl window)."),
    source_crawls: str = typer.Option(..., help="Comma-separated constituent crawl ids."),
    model_space_version: str = typer.Option(..., help="Embedding model-space tag."),
    meta: Path = typer.Option(None, help="Export sidecar JSON (encoder + metrics) to fold in."),
    rows_per_shard: int = typer.Option(4_000_000, help="Rows per contiguous shard (default 4M)."),
    precision: str = typer.Option(
        "both",
        help="Published tier(s): fp32 (canonical), fp16 (primary half-precision), or both "
        "(fp32 + fp16 sidecar).",
    ),
    normalize: bool = typer.Option(
        True, help="L2-normalize vectors (contract default); --no-normalize keeps raw magnitudes."
    ),
    limit: int = typer.Option(None, help="Process only the first N rows (smoke test)."),
) -> None:
    """Shard an exported embedding table into Parquet + a checksummed manifest for HF release."""
    setup_logging()
    import json

    from wgl.release.export import export_parquet_shards

    export_parquet_shards(
        emb,
        node_names,
        out,
        rows_per_shard=rows_per_shard,
        normalize=normalize,
        precision=precision,
        release_id=release_id,
        source_crawls=[c.strip() for c in source_crawls.split(",") if c.strip()],
        model_space_version=model_space_version,
        meta=json.loads(meta.read_text()) if meta else None,
        limit=limit,
    )
