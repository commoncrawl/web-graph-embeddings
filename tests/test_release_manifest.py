"""Phase 0.1 gate: release manifests join stably across persistent/new/missing/reappearing hosts."""

import numpy as np

from wgl.data.graph import DirectedGraph
from wgl.release.manifest import (
    CrossReleaseIndex,
    HostState,
    ReleaseManifest,
)


def _graph(names, edges=()):
    """Build a DirectedGraph from reversed-host node names and optional (src, dst) edges."""
    ei = np.array(edges, dtype=np.int64).T.reshape(2, -1) if edges else np.zeros((2, 0), np.int64)
    return DirectedGraph.from_edges(ei, num_nodes=len(names), node_names=list(names))


def _manifest(release_id, crawl, names, edges=()):
    return ReleaseManifest.from_graph(
        _graph(names, edges), release_id=release_id, source_crawls=[crawl]
    )


# Three synthetic releases. Node *order* (hence local node id) deliberately differs across releases,
# so a correct join must use the stable host key, not the integer id.
#   R1: A B C D
#   R2: B A E         (C, D dropped; E new; A/B reordered)
#   R3: F C A E       (B dropped; C reappears after being absent in R2; F new)
R1 = _manifest("rel-1", "crawl-jan", ["com.a", "com.b", "com.c", "com.d"])
R2 = _manifest("rel-2", "crawl-feb", ["com.b", "com.a", "com.e"])
R3 = _manifest("rel-3", "crawl-mar", ["com.f", "com.c", "com.a", "com.e"])


def test_from_graph_uses_stable_keys_and_local_ids():
    # Keys are canonical (un-reversed); local id is this release's node order.
    assert R1.host_keys == {"a.com", "b.com", "c.com", "d.com"}
    assert R1.records["a.com"].local_node_id == 0
    assert R2.records["a.com"].local_node_id == 1  # different id, same host
    assert R3.records["a.com"].local_node_id == 2


def test_cross_release_states_persistent_new_missing_reappeared():
    idx = CrossReleaseIndex.build([R1, R2, R3])

    # Persistent (active): present in R3 and in the immediately-preceding R2.
    assert idx.entries["a.com"].state is HostState.ACTIVE
    assert idx.entries["e.com"].state is HostState.ACTIVE
    # New: first appears in the latest release.
    assert idx.entries["f.com"].state is HostState.NEW
    # Reappeared: in R1, absent in R2, back in R3.
    assert idx.entries["c.com"].state is HostState.REAPPEARED
    # Missing: present earlier, absent from the latest release.
    assert idx.entries["b.com"].state is HostState.MISSING  # in R1+R2, gone in R3
    assert idx.entries["d.com"].state is HostState.MISSING  # only in R1

    assert idx.state_counts() == {"active": 2, "new": 1, "reappeared": 1, "missing": 2}
    assert idx.keys_in_state(HostState.MISSING) == {"b.com", "d.com"}


def test_cross_release_first_and_last_seen():
    idx = CrossReleaseIndex.build([R1, R2, R3])
    a = idx.entries["a.com"]
    assert a.first_release == "rel-1"
    assert a.last_seen_release == "rel-3"
    # The stable identity table carries each release's local node id for the same host.
    assert a.local_node_ids == {"rel-1": 0, "rel-2": 1, "rel-3": 2}

    c = idx.entries["c.com"]  # reappeared: present in R1 and R3, not R2
    assert c.local_node_ids == {"rel-1": 2, "rel-3": 1}
    assert "rel-2" not in c.local_node_ids


def test_two_release_gate_persistent_new_missing():
    # The runbook's literal gate uses two manifests (reappearing needs >=3).
    idx = CrossReleaseIndex.build([R1, R2])
    assert idx.entries["a.com"].state is HostState.ACTIVE  # persistent
    assert idx.entries["e.com"].state is HostState.NEW
    assert idx.entries["c.com"].state is HostState.MISSING
    assert idx.entries["d.com"].state is HostState.MISSING
    assert idx.state_counts() == {"active": 2, "new": 1, "missing": 2}


def test_single_release_all_new():
    idx = CrossReleaseIndex.build([R1])
    assert all(e.state is HostState.NEW for e in idx.entries.values())


def test_manifest_save_load_roundtrip(tmp_path):
    # A release with edges so cohort/confidence/dangling fields are non-trivial.
    #   edges: a->b, a->c, b->a  => a: in1/out2 (bidirectional/high), b: in1/out1 (bidi/high),
    #   c: in1/out0 (in_only/medium, dangling), d: isolated/low
    names = ["com.a", "com.b", "com.c", "com.d"]
    m = _manifest("rel-x", "crawl-x", names, edges=[(0, 1), (0, 2), (1, 0)])
    m.save(tmp_path)
    loaded = ReleaseManifest.load(tmp_path)
    assert loaded.release_id == "rel-x"
    assert loaded.source_crawls == ["crawl-x"]
    assert loaded.host_keys == m.host_keys
    rc = loaded.records["c.com"]
    assert rc.cohort == "in_only" and rc.confidence == "medium" and rc.is_dangling
    rd = loaded.records["d.com"]
    assert rd.cohort == "isolated" and rd.confidence == "low"
    # Identical join behavior after a round-trip.
    assert CrossReleaseIndex.build([m]).state_counts() == {"new": 4}


def test_duplicate_host_key_raises():
    # Two distinct reversed names that canonicalize to the same host key must be rejected.
    g = _graph(["com.a", "com.a"])
    try:
        ReleaseManifest.from_graph(g, release_id="r", source_crawls=["c"])
    except ValueError as e:
        assert "duplicate host key" in str(e)
    else:
        raise AssertionError("expected ValueError on duplicate host key")
