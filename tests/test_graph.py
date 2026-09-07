"""Tests for the DirectedGraph CSR structures and round-trip persistence."""

import numpy as np

from wgl.data.graph import DirectedGraph


def _toy_edges():
    # 0->1, 0->2, 1->2, 3->0
    return np.array([[0, 0, 1, 3], [1, 2, 2, 0]], dtype=np.int64)


def test_csr_neighbors_directed():
    g = DirectedGraph.from_edges(_toy_edges(), num_nodes=4)
    assert sorted(g.out_neighbors(0).tolist()) == [1, 2]
    assert g.out_neighbors(2).tolist() == []
    # in-neighbors are predecessors, not symmetrized
    assert g.in_neighbors(2).tolist() == [0, 1]
    assert g.in_neighbors(0).tolist() == [3]


def test_degrees():
    g = DirectedGraph.from_edges(_toy_edges(), num_nodes=4)
    assert g.out_degree().tolist() == [2, 1, 0, 1]
    assert g.in_degree().tolist() == [1, 1, 2, 0]


def test_subgraph_from_edge_mask():
    g = DirectedGraph.from_edges(_toy_edges(), num_nodes=4)
    mask = np.array([True, False, True, False])
    sub = g.subgraph_from_edge_mask(mask)
    assert sub.num_edges == 2
    assert sub.num_nodes == 4
    assert set(map(tuple, sub.edge_index.T.tolist())) == {(0, 1), (1, 2)}


def test_to_undirected_reciprocates_edges():
    g = DirectedGraph.from_edges(_toy_edges(), num_nodes=4)
    u = g.to_undirected()
    edges = set(map(tuple, u.edge_index.T.tolist()))
    # Every original edge and its reverse must be present; node count unchanged.
    for s, d in _toy_edges().T.tolist():
        assert (s, d) in edges
        assert (d, s) in edges
    assert u.num_nodes == 4
    # Symmetrized graph has equal in/out degree per node.
    assert u.in_degree().tolist() == u.out_degree().tolist()


def test_save_load_roundtrip(tmp_path):
    g = DirectedGraph.from_edges(_toy_edges(), num_nodes=4, node_names=["a", "b", "c", "d"])
    g.save(tmp_path)
    g2 = DirectedGraph.load(tmp_path)
    assert g2.num_nodes == 4
    assert g2.num_edges == 4
    assert g2.node_names == ["a", "b", "c", "d"]
    np.testing.assert_array_equal(g.out_indices, g2.out_indices)
    assert g2.features is None  # no node_features.npy written when features is None


def test_save_load_roundtrip_with_features(tmp_path):
    g = DirectedGraph.from_edges(_toy_edges(), num_nodes=4)
    g.features = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    g.save(tmp_path)
    assert (tmp_path / "node_features.npy").exists()
    g2 = DirectedGraph.load(tmp_path)
    assert g2.features is not None and g2.features.shape == (4, 5)
    np.testing.assert_array_equal(g.features, g2.features)
