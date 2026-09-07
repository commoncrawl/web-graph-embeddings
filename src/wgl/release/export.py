"""Shard an exported embedding table into Parquet for a HuggingFace publication.

Turns the raw single-file export (e.g. ``cc_deg8_shallow_full_emb.npy`` — ``[52,913,544, 128]``
fp32, ~27 GB) into the **publishable, contract-compliant** layout: contiguous row-range Parquet
shards of
bounded size, each row co-locating the stable identity keys with its vector, plus a top-level
``manifest.json`` (schema/model-space version, per-shard row ranges + SHA-256). See
[`docs/release-contract.md`](../../../docs/release-contract.md) — "Shard layout & checksums".

Why Parquet (vs a single ``.npy`` / safetensors): the identity keys and vector live in the same row
(no separate join file, no row-``i`` alignment discipline), it streams via ``datasets``, and a
consumer can pull one shard — or project only ``host_key`` — without downloading the whole table.

**Contiguous, not hash-of-key.** Shards are row ranges (``shard = row // rows_per_shard``), which
preserves the row-``i`` <-> ``node_names[i]`` <-> ``coords[i]`` alignment the eval + viewer pipeline
depends on. "Locate one host without loading all vectors" is still O(1): look ``host_key`` up in any
shard's ``host_key``/``row_id`` columns, then ``shard = row_id // rows_per_shard``.

Published vectors are **L2-normalized** (cosine = dot on unit vectors) unless ``normalize=False``.
``precision`` picks the published tier(s): ``fp32`` (canonical lossless), ``fp16`` (half-precision
as the *primary* tier — negligible cosine error on unit vectors, half the download), or ``both``
(fp32 canonical + an fp16 sidecar set). The primary tier is always written to ``vectors/``; ``both``
adds the fp16 sidecar under ``vectors_fp16/``.

``pyarrow`` is required (an optional ``cc``/``topic`` extra) — this module is imported lazily by the
CLI so the base package stays pyarrow-free.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from wgl.logging import get_logger
from wgl.release.identity import domain_key, host_key
from wgl.release.manifest import SCHEMA_VERSION

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = get_logger()

_SHA_CHUNK = 16 << 20  # 16 MiB read blocks for checksumming

# Published shard set(s) for each precision choice: (dirname, numpy dtype). The primary tier is
# always written to "vectors/"; "both" additionally emits an fp16 sidecar under "vectors_fp16/".
PRECISION_SETS: dict[str, list[tuple[str, np.dtype]]] = {
    "fp32": [("vectors", np.dtype("float32"))],
    "fp16": [("vectors", np.dtype("float16"))],
    "both": [("vectors", np.dtype("float32")), ("vectors_fp16", np.dtype("float16"))],
}


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in bounded chunks (never loads it whole)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_SHA_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _iter_names(path: Path) -> Iterator[str]:
    """Yield one node name per line (reversed CC notation), no trailing newline.

    Python file iteration yields the final line even without a trailing newline, so this returns
    exactly one name per node (``wc -l`` under-counts by one when the last line is unterminated).
    """
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            yield line.rstrip("\n")


def _keys_for(names: list[str]) -> tuple[list[str], list[str]]:
    """Derive (host_key, domain_key) for a block of reversed CC names via the contract functions."""
    hk = [host_key(n) for n in names]
    dk = [domain_key(n) for n in names]
    return hk, dk


def _write_shard(
    out_dir: Path,
    kind: str,
    shard_idx: int,
    num_shards: int,
    row_start: int,
    block: np.ndarray,
    host_keys: list[str],
    domain_keys: list[str],
    dim: int,
) -> dict:
    """Write one Parquet shard (row_id, host_key, domain_key, embedding); return its manifest row.

    ``block`` is the ``[n, dim]`` vector slice already cast to the target dtype (fp32 or fp16).
    Emits a fixed-size-list embedding column so the on-disk layout is dense and unambiguous.
    """
    n = block.shape[0]
    row_ids = np.arange(row_start, row_start + n, dtype=np.int64)
    flat = pa.array(block.reshape(-1), type=pa.from_numpy_dtype(block.dtype))
    emb_col = pa.FixedSizeListArray.from_arrays(flat, dim)
    table = pa.table(
        {
            "row_id": pa.array(row_ids, type=pa.int64()),
            "host_key": pa.array(host_keys, type=pa.string()),
            "domain_key": pa.array(domain_keys, type=pa.string()),
            "embedding": emb_col,
        }
    )
    fname = f"{kind}/emb-{shard_idx:05d}-of-{num_shards:05d}.parquet"
    path = out_dir / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return {
        "filename": fname,
        "row_start": int(row_start),
        "row_end": int(row_start + n),
        "n_rows": int(n),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _report_input_norms(emb: np.ndarray, sample: int = 4096) -> tuple[float, float, float]:
    """Return (min, mean, max) L2 norm over a leading sample of rows (cheap normalization check)."""
    rows = np.asarray(emb[: min(sample, emb.shape[0])], dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1)
    return float(norms.min()), float(norms.mean()), float(norms.max())


def export_parquet_shards(
    emb_path: Path,
    names_path: Path,
    out_dir: Path,
    *,
    rows_per_shard: int = 4_000_000,
    normalize: bool = True,
    precision: str = "both",
    release_id: str,
    source_crawls: list[str],
    model_space_version: str,
    meta: dict | None = None,
    limit: int | None = None,
) -> Path:
    """Stream the memmapped table into Parquet shards + a manifest; return the manifest path.

    Reads the ``.npy`` via ``mmap_mode`` and the names via a line iterator, materializing only one
    shard's rows at a time, so peak memory is ~one shard (not the whole 27 GB table). ``precision``
    selects the published tier(s) per :data:`PRECISION_SETS`: ``fp32`` / ``fp16`` write a single
    ``vectors/`` set at that dtype; ``both`` writes fp32 ``vectors/`` plus an fp16 ``vectors_fp16/``
    sidecar.

    Args:
        emb_path: Exported ``[N, dim]`` float32 ``.npy`` table.
        names_path: ``node_names.txt`` — one reversed CC host name per row, aligned to ``emb`` rows.
        out_dir: Destination directory for the shard set(s) + ``manifest.json``.
        rows_per_shard: Rows per contiguous shard (shard = ``row_id // rows_per_shard``).
        normalize: L2-normalize each vector (the contract default); ``False`` keeps raw magnitudes.
        precision: One of ``fp32`` / ``fp16`` / ``both`` (see :data:`PRECISION_SETS`).
        release_id: Stable release id (the crawl window).
        source_crawls: Constituent monthly crawl ids recorded in the manifest.
        model_space_version: Embedding model-space tag (vectors only compare within one tag).
        meta: Optional export sidecar dict (``encoder`` + metrics) folded into the manifest.
        limit: Process only the first N rows (smoke test); disables the trailing-alignment check.

    Returns:
        Path to the written ``manifest.json``.
    """
    if precision not in PRECISION_SETS:
        raise ValueError(f"precision must be one of {sorted(PRECISION_SETS)}, got {precision!r}")
    meta = meta or {}
    sets = PRECISION_SETS[precision]
    emb = np.load(emb_path, mmap_mode="r")
    if emb.ndim != 2:
        raise ValueError(f"expected a 2-D embedding table, got shape {emb.shape}")
    num_rows, dim = int(emb.shape[0]), int(emb.shape[1])
    if limit is not None:
        num_rows = min(num_rows, limit)
    num_shards = (num_rows + rows_per_shard - 1) // rows_per_shard

    nmin, nmean, nmax = _report_input_norms(emb)
    logger.info(
        "table %s rows=%d dim=%d input L2 norm(min/mean/max)=%.4f/%.4f/%.4f -> normalize=%s",
        emb.dtype,
        num_rows,
        dim,
        nmin,
        nmean,
        nmax,
        normalize,
    )

    names_iter = _iter_names(names_path)
    shards: dict[str, list[dict]] = {kind: [] for kind, _ in sets}
    seen = 0
    for shard_idx in range(num_shards):
        start = shard_idx * rows_per_shard
        end = min(start + rows_per_shard, num_rows)
        n = end - start
        names = [next(names_iter) for _ in range(n)]
        block = np.array(emb[start:end], dtype=np.float32)  # writable copy (memmap is read-only)
        if normalize:
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            np.divide(block, norms, out=block, where=norms > 0)  # leave all-zero rows untouched
        hk, dk = _keys_for(names)
        for kind, dtype in sets:
            cast = block if dtype == np.float32 else block.astype(dtype)
            shards[kind].append(
                _write_shard(out_dir, kind, shard_idx, num_shards, start, cast, hk, dk, dim)
            )
        seen += n
        logger.info("shard %d/%d rows [%d,%d) written", shard_idx + 1, num_shards, start, end)

    if seen != num_rows:
        raise ValueError(f"wrote {seen} rows, expected {num_rows}")
    # Guard the emb<->names alignment: after consuming num_rows names the iterator must be exhausted
    # (unless we deliberately truncated with limit).
    if limit is None and next(names_iter, None) is not None:
        raise ValueError(
            f"{names_path.name} has more lines than the {num_rows} embedding rows — misaligned"
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_space_version": model_space_version,
        "release_id": release_id,
        "source_crawls": source_crawls,
        "encoder": meta.get("encoder"),
        "dim": dim,
        "num_rows": num_rows,
        "rows_per_shard": rows_per_shard,
        "num_shards": num_shards,
        "distance": "cosine",
        "normalized": normalize,
        "input_norm_min_mean_max": [nmin, nmean, nmax],
        "columns": {
            "row_id": "int64 (global row index; shard = row_id // rows_per_shard)",
            "host_key": "str (forward host, contract identity key)",
            "domain_key": "str (registrable domain / PLD)",
            "embedding": "fixed_size_list<the set's dtype>[dim]",
        },
        "canonical_precision": sets[0][1].name,  # dtype of the primary "vectors/" set
        "precisions": {kind: dtype.name for kind, dtype in sets},
        "metrics": {
            k: meta[k]
            for k in ("best_val_mrr", "test_mrr", "test_hits@1", "test_hits@10")
            if k in meta
        },
        "shards": shards,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    kinds = "+".join(k for k, _ in sets)
    logger.info("wrote %s (%d shards x %s)", manifest_path, num_shards, kinds)
    return manifest_path
