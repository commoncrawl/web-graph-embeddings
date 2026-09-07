"""Directed graph stored as out-edge and in-edge CSR structures.

Web graphs are fundamentally directed and the in/out asymmetry is the key signal for tasks like
spam detection (see ``docs/brainstorming/edge-direction.md``). We therefore keep two CSR views and
never symmetrize implicitly:

* **out-CSR** (``out_indptr``, ``out_indices``): successors of each node (who it links to).
* **in-CSR**  (``in_indptr``,  ``in_indices``):  predecessors of each node (who links to it).

The COO ``edge_index`` of shape ``(2, num_edges)`` (rows: ``src``, ``dst``) is the canonical edge
list. Arrays are plain NumPy so they save/load with ``np.savez`` and can be memory-mapped at scale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _build_csr(
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
    num_nodes: int,
    index_dtype: np.dtype = np.int64,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Build a CSR ``(indptr, indices)`` grouping ``cols`` by ``rows``.

    Args:
        rows: Source endpoints (the CSR row of each entry).
        cols: Destination endpoints (the value stored in each row).
        num_nodes: Total number of nodes (CSR has ``num_nodes + 1`` indptr entries).
        index_dtype: dtype of the returned ``indices`` (the node-id column). ``int32`` halves the
            footprint when node ids fit (``num_nodes < 2**31``) — worth it at 10^10-edge scale.
            ``indptr`` is always ``int64`` (it accumulates to ``num_edges``, which can exceed 2^31).

    Returns:
        A tuple ``(indptr, indices)``.
    """
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    sorted_cols = cols[order]
    counts = np.bincount(sorted_rows, minlength=num_nodes)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, sorted_cols.astype(index_dtype, copy=False)


@dataclass
class DirectedGraph:
    """An immutable directed graph with cached out/in CSR adjacency.

    Attributes:
        num_nodes: Number of nodes.
        edge_index: COO edges of shape ``(2, num_edges)`` (row 0 = src, row 1 = dst).
        out_indptr: Out-CSR row pointers, shape ``(num_nodes + 1,)``.
        out_indices: Out-CSR successors.
        in_indptr: In-CSR row pointers, shape ``(num_nodes + 1,)``.
        in_indices: In-CSR predecessors.
        node_names: Optional per-node string identifiers (e.g. reversed hostnames).
        features: Optional per-node feature matrix, shape ``(num_nodes, feature_dim)`` (e.g.
            ogbn-arxiv's word2vec vectors). Persisted as ``node_features.npy``; ``None`` for
            structure-only graphs (the CC web graph has no such features).
    """

    num_nodes: int
    edge_index: NDArray[np.int64]
    out_indptr: NDArray[np.int64]
    out_indices: NDArray[np.int64]
    in_indptr: NDArray[np.int64]
    in_indices: NDArray[np.int64]
    node_names: list[str] | None = None
    features: NDArray[np.float32] | None = None

    @property
    def num_edges(self) -> int:
        """Number of directed edges."""
        return int(self.edge_index.shape[1])

    @classmethod
    def from_edges(
        cls,
        edge_index: NDArray[np.int64],
        num_nodes: int | None = None,
        node_names: list[str] | None = None,
    ) -> DirectedGraph:
        """Construct a graph from a COO edge list, building both CSR views.

        Args:
            edge_index: Array of shape ``(2, num_edges)`` with src in row 0 and dst in row 1.
            num_nodes: Node count; inferred from the max id + 1 when omitted.
            node_names: Optional per-node identifiers.

        Returns:
            A new :class:`DirectedGraph`.
        """
        edge_index = np.ascontiguousarray(edge_index, dtype=np.int64)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            msg = f"edge_index must have shape (2, E), got {edge_index.shape}"
            raise ValueError(msg)
        src, dst = edge_index[0], edge_index[1]
        if num_nodes is None:
            num_nodes = int(edge_index.max()) + 1 if edge_index.size else 0
        out_indptr, out_indices = _build_csr(src, dst, num_nodes)
        in_indptr, in_indices = _build_csr(dst, src, num_nodes)
        return cls(
            num_nodes=num_nodes,
            edge_index=edge_index,
            out_indptr=out_indptr,
            out_indices=out_indices,
            in_indptr=in_indptr,
            in_indices=in_indices,
            node_names=node_names,
        )

    def out_neighbors(self, node: int) -> NDArray[np.int64]:
        """Return the successors of ``node`` (nodes it links to)."""
        return self.out_indices[self.out_indptr[node] : self.out_indptr[node + 1]]

    def in_neighbors(self, node: int) -> NDArray[np.int64]:
        """Return the predecessors of ``node`` (nodes that link to it)."""
        return self.in_indices[self.in_indptr[node] : self.in_indptr[node + 1]]

    def out_degree(self) -> NDArray[np.int64]:
        """Return the out-degree of every node."""
        return np.diff(self.out_indptr)

    def in_degree(self) -> NDArray[np.int64]:
        """Return the in-degree of every node."""
        return np.diff(self.in_indptr)

    def to_undirected(self) -> DirectedGraph:
        """Return a symmetrized copy with both ``(u, v)`` and ``(v, u)`` for every edge.

        Used only to build the *undirected baseline* for direction ablations — it deliberately
        discards the in/out asymmetry that the directed models exploit, so it should never be used
        for the primary directed experiments.

        Returns:
            A new :class:`DirectedGraph` over the same nodes with reciprocated, de-duplicated edges.
        """
        both = np.concatenate([self.edge_index, self.edge_index[::-1]], axis=1)
        undirected = np.unique(both, axis=1) if both.size else both
        return DirectedGraph.from_edges(
            undirected, num_nodes=self.num_nodes, node_names=self.node_names
        )

    def subgraph_from_edge_mask(self, mask: NDArray[np.bool_]) -> DirectedGraph:
        """Return a graph over the same node set keeping only the masked edges.

        Used to build the *message* graph for link prediction (a structural view that excludes
        held-out evaluation edges).

        Args:
            mask: Boolean mask over edges of length ``num_edges``.

        Returns:
            A new :class:`DirectedGraph` with the same ``num_nodes`` and node names.
        """
        return DirectedGraph.from_edges(
            self.edge_index[:, mask], num_nodes=self.num_nodes, node_names=self.node_names
        )

    def save(self, directory: str | Path) -> None:
        """Persist the graph to ``directory`` (``graph.npz`` + ``meta.json``).

        Args:
            directory: Output directory (created if needed).
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "graph.npz",
            edge_index=self.edge_index,
            out_indptr=self.out_indptr,
            out_indices=self.out_indices,
            in_indptr=self.in_indptr,
            in_indices=self.in_indices,
        )
        meta = {"num_nodes": self.num_nodes, "num_edges": self.num_edges}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))
        if self.node_names is not None:
            (directory / "node_names.txt").write_text("\n".join(self.node_names))
        if self.features is not None:
            np.save(directory / "node_features.npy", self.features)

    @classmethod
    def load(cls, directory: str | Path, mmap: bool = False) -> DirectedGraph:
        """Load a graph previously written by :meth:`save`.

        Args:
            directory: Directory containing ``graph.npz`` / ``meta.json``.
            mmap: If True, memory-map the arrays (recommended for large graphs).

        Returns:
            The loaded :class:`DirectedGraph`.
        """
        directory = Path(directory)
        data = np.load(directory / "graph.npz", mmap_mode="r" if mmap else None)
        meta = json.loads((directory / "meta.json").read_text())
        names_path = directory / "node_names.txt"
        node_names = names_path.read_text().splitlines() if names_path.exists() else None
        features_path = directory / "node_features.npy"
        features = np.load(features_path) if features_path.exists() else None
        return cls(
            num_nodes=int(meta["num_nodes"]),
            edge_index=data["edge_index"],
            out_indptr=data["out_indptr"],
            out_indices=data["out_indices"],
            in_indptr=data["in_indptr"],
            in_indices=data["in_indices"],
            node_names=node_names,
            features=features,
        )
