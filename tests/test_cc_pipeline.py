"""Tests for the published embedding pipeline's ingestion stages.

Covers the three modules a minimal reproduction checkout ships: shard resolution/parsing
(:mod:`wgl.data.cc_shards`), the min-degree induced training slice
(:mod:`wgl.data.cc_min_degree`), and the memmap-able edge index + split cache
(:mod:`wgl.data.cc_split_cache`).
"""

import gzip

import numpy as np
import pytest

from _cc_helpers import _collect_edges, _write_single
from wgl.data.cc_min_degree import load_min_degree_subgraph, write_min_degree_slice
from wgl.data.cc_shards import (
    _parse_edge_shard,
    _parse_edge_shard_numpy,
    accumulate_degrees,
    resolve_shards,
)


def test_edge_shard_parsers_agree_and_match_source(tmp_path):
    # The pyarrow fast path (when available) and the numpy fallback must yield identical edges;
    # without pyarrow, _parse_edge_shard falls back to numpy so this is trivially true.
    for name in ("edges.txt", "edges.txt.gz"):
        path = tmp_path / name
        src = np.arange(250, dtype=np.int64)
        dst = (src * 7) % 100
        text = "".join(f"{s}\t{d}\n" for s, d in zip(src, dst, strict=True))
        if name.endswith(".gz"):
            with gzip.open(path, "wt") as fh:
                fh.write(text)
        else:
            path.write_text(text)
        ns, nd = _collect_edges(_parse_edge_shard_numpy(path, chunk_lines=50))
        ps, pd = _collect_edges(_parse_edge_shard(path, chunk_lines=50))
        assert np.array_equal(ns, src) and np.array_equal(nd, dst)
        assert np.array_equal(ps, ns) and np.array_equal(pd, nd)


def test_resolve_shards_glob_and_list(tmp_path):
    (tmp_path / "a.txt.gz").write_bytes(b"")
    (tmp_path / "b.txt.gz").write_bytes(b"")
    matched = resolve_shards(str(tmp_path / "*.txt.gz"))
    assert len(matched) == 2
    assert resolve_shards([tmp_path / "a.txt.gz", tmp_path / "b.txt.gz"]) == matched


def _edges_as_pairs(g):
    return {(int(s), int(d)) for s, d in zip(*g.edge_index, strict=True)}


def test_min_degree_slice_keeps_well_connected_nodes(tmp_path):
    # Degrees on the _write_single graph: 0->3, 1->3, 2->2, 3->1, 4->1 (in+out).
    vpath, epath = _write_single(tmp_path)
    g, orig_ids, source_n = load_min_degree_subgraph(vpath, epath, min_degree=2)

    assert source_n == 5
    # deg>=2 keeps original nodes {0,1,2}; the degree-1 tail {3,4} drops to fallback.
    assert orig_ids.tolist() == [0, 1, 2]
    assert g.num_nodes == 3
    assert g.node_names == ["com.a", "com.b", "org.c"]
    # Induced edges: only those with BOTH endpoints kept, remapped to compact ids (identity here).
    assert _edges_as_pairs(g) == {(0, 1), (0, 2), (1, 2)}


def test_min_degree_slice_threshold_is_nested_and_names_align(tmp_path):
    vpath, epath = _write_single(tmp_path)
    g3, ids3, _ = load_min_degree_subgraph(vpath, epath, min_degree=3)
    # deg>=3 keeps {0,1}; the single edge between them survives.
    assert ids3.tolist() == [0, 1]
    assert _edges_as_pairs(g3) == {(0, 1)}
    # orig_ids[compact] must recover the right name at full resolution (the serving join key).
    g1, ids1, _ = load_min_degree_subgraph(vpath, epath, min_degree=1)
    assert ids1.tolist() == [0, 1, 2, 3, 4]  # nothing dropped at deg>=1
    for compact, gid in enumerate(ids3):
        assert g3.node_names[compact] == g1.node_names[int(gid)]


def test_min_degree_slice_degree_cache_roundtrip(tmp_path):
    vpath, epath = _write_single(tmp_path)
    cache = tmp_path / "deg.npy"
    g_a, ids_a, _ = load_min_degree_subgraph(vpath, epath, min_degree=2, degree_cache=cache)
    assert cache.exists()
    # Second call reuses the cached degrees (same result, threshold-independent cache).
    g_b, ids_b, _ = load_min_degree_subgraph(vpath, epath, min_degree=2, degree_cache=cache)
    assert ids_a.tolist() == ids_b.tolist()
    assert _edges_as_pairs(g_a) == _edges_as_pairs(g_b)
    # Cached degrees match a fresh accumulation.
    assert np.array_equal(np.load(cache), accumulate_degrees(epath, 5))


def test_min_degree_slice_empty_keep_raises(tmp_path):
    vpath, epath = _write_single(tmp_path)
    with pytest.raises(ValueError, match="keeps 0 nodes"):
        load_min_degree_subgraph(vpath, epath, min_degree=99)


def test_streaming_writer_matches_in_memory_with_csr(tmp_path):
    # The out-of-core writer must produce the SAME slice as the in-memory path (edges, ids, names).
    vpath, epath = _write_single(tmp_path)
    g_mem, ids_mem, _ = load_min_degree_subgraph(vpath, epath, min_degree=2)

    out = tmp_path / "slice"
    n_nodes, n_edges, source_n = write_min_degree_slice(vpath, epath, 2, out)
    assert (n_nodes, n_edges, source_n) == (g_mem.num_nodes, g_mem.num_edges, 5)

    import json

    meta = json.loads((out / "meta.json").read_text())
    assert meta["has_csr"] is True and meta["min_degree"] == 2 and meta["source_num_nodes"] == 5
    assert np.load(out / "orig_ids.npy").tolist() == ids_mem.tolist()
    assert not (out / "_edge_index.scratch.npy").exists()  # scratch renamed, not left behind
    # The standalone edge_index.npy (the memmap-able COO the trainer reads) is emitted + memmaps.
    assert meta["has_edge_index_npy"] is True
    ei_mm = np.load(out / "edge_index.npy", mmap_mode="r")
    assert isinstance(ei_mm, np.memmap)
    assert {(int(s), int(d)) for s, d in zip(*np.asarray(ei_mm), strict=True)} == _edges_as_pairs(
        g_mem
    )

    # With CSR the artifact loads as a full DirectedGraph identical to the in-memory build.
    from wgl.data.graph import DirectedGraph

    g_stream = DirectedGraph.load(out)
    assert g_stream.node_names == g_mem.node_names
    assert _edges_as_pairs(g_stream) == _edges_as_pairs(g_mem)


def test_streaming_writer_coo_only_when_over_csr_budget(tmp_path):
    # csr_max_edges below the induced edge count forces the COO-only path (no CSR, still trainable).
    vpath, epath = _write_single(tmp_path)
    g_mem, _, _ = load_min_degree_subgraph(vpath, epath, min_degree=2)
    out = tmp_path / "coo"
    write_min_degree_slice(vpath, epath, 2, out, csr_max_edges=1)

    import json

    meta = json.loads((out / "meta.json").read_text())
    assert meta["has_csr"] is False
    # graph.npz holds only edge_index; the COO edge set still matches the in-memory build (the
    # cuGraph-PyG training path reads exactly this array).
    with np.load(out / "graph.npz") as npz:
        assert set(npz.files) == {"edge_index"}
        ei = npz["edge_index"]
    assert ei.dtype == np.int32
    pairs = {(int(s), int(d)) for s, d in zip(*ei, strict=True)}
    assert pairs == _edges_as_pairs(g_mem)
    assert (out / "node_names.txt").read_text().splitlines() == g_mem.node_names
    # Even COO-only, the memmap-able standalone .npy is emitted with the same edges.
    ei_mm = np.load(out / "edge_index.npy", mmap_mode="r")
    assert isinstance(ei_mm, np.memmap) and ei_mm.dtype == np.int32
    assert {(int(s), int(d)) for s, d in zip(*np.asarray(ei_mm), strict=True)} == pairs


def test_extract_edge_index_npy_roundtrips_and_memmaps(tmp_path):
    # One-time fix-up for a slice that only has graph.npz: extract a memmap-able standalone .npy.
    from wgl.data.cc_split_cache import extract_edge_index_npy

    ei = np.arange(24, dtype=np.int32).reshape(2, 12)
    (tmp_path / "graph.npz").write_bytes(b"")  # placeholder, overwritten by savez below
    np.savez(tmp_path / "graph.npz", edge_index=ei)
    path, shape = extract_edge_index_npy(tmp_path)
    assert path == tmp_path / "edge_index.npy" and shape == (2, 12)
    out = np.load(path, mmap_mode="r")
    assert isinstance(out, np.memmap)
    assert np.array_equal(np.asarray(out), ei)


def test_write_split_cache_roundtrips_and_is_leakage_safe(tmp_path):
    # Identity graph: column i == (i, i), so a split maps 1:1 back to edge index i.
    import json

    from wgl.data.cc_split_cache import _split_buckets, write_split_cache

    n = 5000
    ei = np.stack([np.arange(n), np.arange(n)]).astype(np.int32)
    np.save(tmp_path / "edge_index.npy", ei)
    (tmp_path / "meta.json").write_text(json.dumps({"num_nodes": n}))

    # eval_edges=0 keeps val/test whole, so the three splits partition every edge exactly.
    meta = write_split_cache(tmp_path, eval_edges=0, chunk=997)
    cache = tmp_path / "split_cache"
    assert meta["num_nodes"] == n and meta["n_edges"] == n

    train_mm = np.load(cache / "train.npy", mmap_mode="r")
    assert train_mm.shape == (2, n)  # [2, n_edges] with only [:, :train_total] valid
    train = np.asarray(train_mm[:, : meta["train_total"]])
    val = np.load(cache / "val.npy")
    test = np.load(cache / "test.npy")

    buckets = _split_buckets(0, n, seed=42)
    assert {int(c) for c in train[0]} == set(np.flatnonzero(buckets >= 2).tolist())
    assert {int(c) for c in val[0]} == set(np.flatnonzero(buckets == 1).tolist())
    assert {int(c) for c in test[0]} == set(np.flatnonzero(buckets == 0).tolist())
    # Exact partition (no leakage, no drops): train + val + test == all edges, disjoint.
    assert meta["train_total"] + val.shape[1] + test.shape[1] == n
    assert set(train[0]).isdisjoint(val[0]) and set(train[0]).isdisjoint(test[0])


def test_write_split_cache_subsamples_val_test(tmp_path):
    # With an eval cap, val/test are subsampled in-loop; train is untouched (all buckets>=2 kept).
    import json

    from wgl.data.cc_split_cache import _split_buckets, write_split_cache

    n = 200_000
    ei = np.stack([np.arange(n), np.arange(n)]).astype(np.int32)
    np.save(tmp_path / "edge_index.npy", ei)
    (tmp_path / "meta.json").write_text(json.dumps({"num_nodes": n}))

    cap = 2000
    meta = write_split_cache(tmp_path, eval_edges=cap, chunk=50_000)
    n_train_full = int(np.count_nonzero(_split_buckets(0, n, seed=42) >= 2))
    assert meta["train_total"] == n_train_full  # train never subsampled
    # ~1.5x cap kept per held-out split (well under the untrimmed ~n/10), deterministic.
    assert cap < meta["val_count"] < 4 * cap
    assert cap < meta["test_count"] < 4 * cap
