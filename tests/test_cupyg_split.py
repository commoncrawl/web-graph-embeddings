"""Tests for the streaming permutation-free split + per-rank shard of the cupyg trainer.

The ``cupyg`` package lives in the cuGraph-PyG container image (wgl is intentionally absent there),
so ``scripts/`` is put on ``sys.path`` here -- exactly as ``python /gs-scripts/cupyg_lp.py`` does --
and only the numpy-only helpers are exercised: the split assignment and per-rank sharding, which
must partition train exactly and be reproducible regardless of chunk size.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")  # the cupyg modules import torch at module top

_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from cupyg import loader_free  # noqa: E402
from cupyg import split as cupyg_split  # noqa: E402


def _write_identity_graph(tmp_path, n_edges, *, edge_index_npy=True):
    """graph.npz whose edge i = (i, i): columns are unique so a split maps 1:1 back to its index.

    Also writes the standalone ``edge_index.npy`` the trainer memmaps (unless ``edge_index_npy`` is
    False, to exercise the missing-file guard).
    """
    ei = np.stack([np.arange(n_edges), np.arange(n_edges)]).astype(np.int32)
    np.savez(tmp_path / "graph.npz", edge_index=ei)
    if edge_index_npy:
        np.save(tmp_path / "edge_index.npy", ei)
    (tmp_path / "meta.json").write_text(json.dumps({"num_nodes": n_edges}))
    return tmp_path


def _cols(edges):
    return {(int(s), int(d)) for s, d in zip(*edges, strict=True)}


def test_split_buckets_fractions_and_determinism():
    b = cupyg_split._split_buckets(0, 200_000, seed=42)
    # ~80/10/10 over the 10 hash buckets (>=2 train, 1 val, 0 test).
    assert 0.09 < np.mean(b == 0) < 0.11
    assert 0.09 < np.mean(b == 1) < 0.11
    assert 0.78 < np.mean(b >= 2) < 0.82
    # Splitting by index in chunks == splitting the whole range at once (chunk-invariant).
    whole = cupyg_split._split_buckets(0, 1000, seed=7)
    piece = np.concatenate(
        [cupyg_split._split_buckets(0, 400, 7), cupyg_split._split_buckets(400, 600, 7)]
    )
    assert np.array_equal(whole, piece)


def test_stream_split_partitions_train_and_covers_all_edges(tmp_path):
    n = 5000
    g = _write_identity_graph(tmp_path, n)
    buckets = cupyg_split._split_buckets(0, n, seed=42)
    exp_test = {(i, i) for i in np.flatnonzero(buckets == 0)}
    exp_val = {(i, i) for i in np.flatnonzero(buckets == 1)}
    exp_train = {(i, i) for i in np.flatnonzero(buckets >= 2)}

    world = 4
    shards = [
        cupyg_split.stream_split_and_shard(g, seed=42, world_size=world, global_rank=r)
        for r in range(world)
    ]
    num_nodes = shards[0][0]
    assert num_nodes == n

    train_sets = [_cols(t) for _, t, _, _ in shards]
    # Every rank sees the SAME full held-out sets (val/test are replicated for the current eval).
    for _, _, val, test in shards:
        assert _cols(val) == exp_val
        assert _cols(test) == exp_test
    # Train shards are DISJOINT and their union is exactly the train split (no drop/dup edge).
    union = set().union(*train_sets)
    assert union == exp_train
    assert sum(len(s) for s in train_sets) == len(exp_train)  # disjoint (counts add up)
    # Round-robin striping keeps the shards within one edge of each other.
    assert max(map(len, train_sets)) - min(map(len, train_sets)) <= 1
    # Full coverage, no leakage: train / val / test partition all edges.
    assert exp_train | exp_val | exp_test == {(i, i) for i in range(n)}


def test_stream_split_is_chunk_invariant_and_reproducible(tmp_path):
    n = 3000
    g = _write_identity_graph(tmp_path, n)
    a = cupyg_split.stream_split_and_shard(g, seed=1, world_size=3, global_rank=1, chunk=1_000_000)
    b = cupyg_split.stream_split_and_shard(
        g, seed=1, world_size=3, global_rank=1, chunk=250
    )  # 12 chunks
    assert _cols(a[1]) == _cols(b[1])  # train shard identical across chunk sizes
    assert _cols(a[2]) == _cols(b[2]) and _cols(a[3]) == _cols(b[3])


def test_subsample_eval_bounded_deterministic_and_lossless():
    edges = np.stack([np.arange(10_000), np.arange(10_000) + 100_000]).astype(np.int64)
    # 0 or a cap >= size returns the array unchanged (the validated slice path).
    assert cupyg_split._subsample_eval(edges, 0, seed=1) is edges
    assert cupyg_split._subsample_eval(edges, 10_000, seed=1) is edges
    assert cupyg_split._subsample_eval(edges, 50_000, seed=1) is edges
    # A real cap is bounded and deterministic for a given seed.
    a = cupyg_split._subsample_eval(edges, 1_000, seed=7)
    b = cupyg_split._subsample_eval(edges, 1_000, seed=7)
    assert a.shape[0] == 2 and a.shape[1] <= 1_000  # <= because collision+unique may drop a few
    assert np.array_equal(a, b)  # same seed -> identical subsample (comparable across epochs/ranks)
    assert not np.array_equal(a, cupyg_split._subsample_eval(edges, 1_000, seed=8))  # seed matters
    # Every sampled column is an intact (src, dst) pair from the original (paired, not shuffled).
    picked = {(int(s), int(d)) for s, d in zip(*a, strict=True)}
    orig = {(i, i + 100_000) for i in range(10_000)}
    assert picked <= orig


def test_stream_split_reads_a_true_memmap(tmp_path):
    # The edge_index the loop streams must be a real memmap, never a full in-RAM array (an .npz
    # member silently ignores mmap_mode -> ~105 GB/rank at full scale). The helper enforces it.
    g = _write_identity_graph(tmp_path, 2000)
    arr = cupyg_split._open_edge_index(g)
    assert isinstance(arr, np.memmap)
    # Missing standalone edge_index.npy -> a clear, actionable error (not a silent full load).
    sub = tmp_path / "no_npy"
    sub.mkdir()
    g2 = _write_identity_graph(sub, 2000, edge_index_npy=False)
    with pytest.raises(FileNotFoundError, match="extract-edge-index"):
        cupyg_split.stream_split_and_shard(g2, seed=1, world_size=1, global_rank=0)


def test_stream_split_eval_cap_subsamples_val_test_not_train(tmp_path):
    n = 20_000
    g = _write_identity_graph(tmp_path, n)
    # eval_cap=0 keeps the full held-out sets (the validated slice behaviour).
    _, tr_full, val_full, test_full = cupyg_split.stream_split_and_shard(
        g, seed=3, world_size=1, global_rank=0, eval_cap=0
    )
    # eval_cap>0 shrinks val/test in-loop (never materialized in full) but leaves TRAIN untouched.
    _, tr_cap, val_cap, test_cap = cupyg_split.stream_split_and_shard(
        g, seed=3, world_size=1, global_rank=0, eval_cap=200
    )
    assert _cols(tr_cap) == _cols(tr_full)  # train split identical
    assert val_cap.shape[1] < val_full.shape[1] and test_cap.shape[1] < test_full.shape[1]
    # ~1.5x cap kept (slack for the exact-cap trim); the kept edges are a subset of the full set.
    assert val_cap.shape[1] <= 3 * 200 and test_cap.shape[1] <= 3 * 200
    assert _cols(val_cap) <= _cols(val_full) and _cols(test_cap) <= _cols(test_full)
    # Deterministic + identical across ranks (index-keyed): same seed -> same subsample.
    _, _, val_cap2, _ = cupyg_split.stream_split_and_shard(
        g, seed=3, world_size=1, global_rank=0, eval_cap=200
    )
    assert _cols(val_cap) == _cols(val_cap2)


def test_stream_split_single_rank_and_max_edges(tmp_path):
    n = 2000
    g = _write_identity_graph(tmp_path, n)
    # world_size 1: the one rank ingests the entire train split.
    _, train, _val, _test = cupyg_split.stream_split_and_shard(
        g, seed=5, world_size=1, global_rank=0
    )
    buckets = cupyg_split._split_buckets(0, n, seed=5)
    assert _cols(train) == {(i, i) for i in np.flatnonzero(buckets >= 2)}
    # max_edges caps the stream to the first N edges only.
    _, tr2, v2, te2 = cupyg_split.stream_split_and_shard(
        g, seed=5, world_size=1, global_rank=0, max_edges=500
    )
    assert len(_cols(tr2)) + len(_cols(v2)) + len(_cols(te2)) == 500


def _import_wgl_split_cache():
    """The wgl writer (available in the test env, though not in the cupyg image)."""
    return pytest.importorskip("wgl.data.cc_split_cache")


def test_split_cache_hash_matches_trainer_no_drift():
    # The split hash is duplicated across images (wgl writer vs cupyg trainer); they MUST agree, or
    # a cached split would silently disagree with the in-line one.
    ccw = _import_wgl_split_cache()
    assert np.array_equal(
        ccw._split_buckets(0, 50_000, seed=13), cupyg_split._split_buckets(0, 50_000, seed=13)
    )
    assert np.array_equal(
        ccw._split_buckets(1234, 9_000, seed=7), cupyg_split._split_buckets(1234, 9_000, seed=7)
    )
    idx = np.arange(5, 40_005, dtype=np.int64)
    assert np.array_equal(
        ccw._eval_keep(idx, 0.3, seed=49), cupyg_split._eval_keep(idx, 0.3, seed=49)
    )


def test_load_split_cache_partitions_train_and_matches_streaming(tmp_path):
    # End-to-end: the wgl writer's cache, read back by the trainer's loader, must partition train
    # contiguously across ranks (union == the streamed split's train), with identical val/test.
    ccw = _import_wgl_split_cache()
    n = 6000
    g = _write_identity_graph(tmp_path, n)
    ccw.write_split_cache(g, eval_edges=0, chunk=701)  # eval_edges=0 -> whole val/test, exact
    cache = g / "split_cache"

    # The in-line streamed split (world_size 1) is the reference train/val/test membership.
    num_nodes, tr_ref, val_ref, test_ref = cupyg_split.stream_split_and_shard(
        g, seed=42, world_size=1, global_rank=0, eval_cap=0
    )

    world = 4
    union: set = set()
    total = 0
    for r in range(world):
        nn, shard, val, test = cupyg_split.load_split_cache(cache, world, r, eval_cap=0)
        assert nn == num_nodes == n
        assert shard.shape[0] == 2 and shard.flags["C_CONTIGUOUS"]
        union |= _cols(shard)
        total += shard.shape[1]
        # val/test are the shared held-out sets, identical on every rank.
        assert _cols(val) == _cols(val_ref) and _cols(test) == _cols(test_ref)
    # Disjoint contiguous shards whose union is exactly the streamed train split.
    assert total == len(union) == tr_ref.shape[1]
    assert union == _cols(tr_ref)


def test_sampler_free_loader_local_index_recovers_edges_and_batch_count():
    # The sampler-free loader (shallow full-CC path) must yield n//bs batches (drop_last) whose
    # LOCAL edge_label_index, indexed back through n_id, EXACTLY recovers the original edges —
    # the correctness contract that lets the model gather-by-id and the loss score locally.

    n, bs = 100, 16
    edges = np.stack([np.arange(n), np.arange(n) + 1000]).astype(np.int64)  # unique global ids

    def _all_edge_cols(loader):  # global (src,dst) pairs recovered from every batch via n_id
        cols = set()
        for b in loader:
            assert b.edge_label_index.shape == (2, bs) and (b.edge_label == 1).all()
            assert set(b.n_id.tolist()) == set(  # n_id = unique endpoints of this batch
                b.n_id[b.edge_label_index].flatten().tolist()
            )
            recon = b.n_id[b.edge_label_index].numpy()  # LOCAL indices -> GLOBAL ids
            cols |= {(int(s), int(d)) for s, d in zip(recon[0], recon[1], strict=True)}
        return cols

    no_shuf = list(loader_free._SamplerFreeEdgeLoader(edges, bs, shuffle=False))
    assert len(no_shuf) == n // bs  # drop_last drops the ragged tail
    for k, b in enumerate(no_shuf):  # shuffle=False -> contiguous slices, exact round-trip
        assert np.array_equal(b.n_id[b.edge_label_index].numpy(), edges[:, k * bs : (k + 1) * bs])
    all_edges = {(i, i + 1000) for i in range(n)}
    # Shuffled loader covers the SAME edge set (drop_last drops n%bs=4) and re-iterates cleanly.
    shuf = loader_free._SamplerFreeEdgeLoader(edges, bs, shuffle=True, seed=1)
    ep0, ep1 = _all_edge_cols(shuf), _all_edge_cols(shuf)
    assert ep0 <= all_edges and ep1 <= all_edges and len(ep0) == (n // bs) * bs
