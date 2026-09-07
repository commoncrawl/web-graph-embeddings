"""Shared shard-writing helpers for the Common Crawl ingestion tests.

Not named ``test_*`` so pytest does not collect it; imported by both the release-path tests
(:mod:`test_cc_pipeline`) and the downsampling tests (:mod:`test_cc_sampling`).
"""

import gzip

import numpy as np


def _collect_edges(gen):
    srcs, dsts = [], []
    for s, d in gen:
        srcs.append(np.asarray(s, dtype=np.int64))
        dsts.append(np.asarray(d, dtype=np.int64))
    return np.concatenate(srcs), np.concatenate(dsts)


def _write_single(tmp_path):
    vpath = tmp_path / "vertices.txt.gz"
    epath = tmp_path / "edges.txt.gz"
    with gzip.open(vpath, "wt") as f:
        for i, h in enumerate(["com.a", "com.b", "org.c", "net.d", "uk.co.bbc"]):
            f.write(f"{i}\t{h}\n")
    with gzip.open(epath, "wt") as f:
        for s, t in [(0, 1), (0, 2), (1, 2), (3, 0), (4, 1)]:
            f.write(f"{s}\t{t}\n")
    return vpath, epath


def _write_dense(tmp_path, n_nodes=200, fan=8):
    """Two-shard graph where every node has ``fan`` out-edges (avg out-degree == fan)."""
    vdir = tmp_path / "v"
    edir = tmp_path / "e"
    vdir.mkdir()
    edir.mkdir()
    half = n_nodes // 2
    with gzip.open(vdir / "part-0.txt.gz", "wt") as f:
        for i in range(half):
            f.write(f"{i}\thost-{i}\n")
    with gzip.open(vdir / "part-1.txt.gz", "wt") as f:
        for i in range(half, n_nodes):
            f.write(f"{i}\thost-{i}\n")
    edges = [(i, (i * k + 1) % n_nodes) for i in range(n_nodes) for k in range(1, fan + 1)]
    with gzip.open(edir / "part-0.txt.gz", "wt") as f:
        for s, t in edges[: len(edges) // 2]:
            f.write(f"{s}\t{t}\n")
    with gzip.open(edir / "part-1.txt.gz", "wt") as f:
        for s, t in edges[len(edges) // 2 :]:
            f.write(f"{s}\t{t}\n")
    return vdir, edir, n_nodes, fan
