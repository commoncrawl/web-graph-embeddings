"""Declarative schema for the v1 release artifact (runbook-v2 Phase 0.2).

The default download stays **compact**: one row per host (or per domain) with the stable key, the
default embedding, and lightweight evidence/graph-context metadata. Larger diagnostics and any
component/role vectors live in optional **sidecars** so they do not inflate the default artifact.

Fields are split by *location*:

* ``HEADER``      — one value per release (lives in the manifest header, not repeated per row);
* ``ROW_DEFAULT`` — per-host/-domain, shipped in the default artifact;
* ``ROW_SIDECAR`` — per-host/-domain, optional, shipped only on demand.

This module is the single source of truth for which fields exist, their on-disk width, and where
they live; :mod:`wgl.release.sizing` consumes the fixed-width metadata cost to estimate artifact
sizes. See [`docs/release-contract.md`](../../../docs/release-contract.md) §0.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FieldLocation(StrEnum):
    """Where a schema field lives in the released artifact."""

    HEADER = "header"  # one value per release (manifest header)
    ROW_DEFAULT = "row_default"  # per node, in the default artifact
    ROW_SIDECAR = "row_sidecar"  # per node, optional sidecar


class FieldKind(StrEnum):
    """Coarse value kind, used by the size model."""

    KEY = "key"  # variable-length string identifier
    VECTOR = "vector"  # the embedding (size = dim x precision, handled by sizing)
    SCALAR = "scalar"  # fixed-width numeric / categorical
    META = "meta"  # release-level metadata (header)


@dataclass(frozen=True)
class Field:
    """One artifact field.

    Attributes:
        name: Column name.
        dtype: On-disk dtype label (e.g. ``uint32``, ``float32``, ``uint8``, ``str``).
        location: Where the field lives (:class:`FieldLocation`).
        kind: Value kind (:class:`FieldKind`).
        nbytes: Fixed on-disk width in bytes for a single value, or ``None`` for variable-width
            (the key) and the embedding (whose width is ``dim x precision``, computed by sizing).
        doc: One-line description.
    """

    name: str
    dtype: str
    location: FieldLocation
    kind: FieldKind
    nbytes: int | None
    doc: str


# --- The v1 schema (compact default + optional sidecar) -----------------------------------------
#
# Degree maxima from the cc-main-2026-mar-apr-may host stats (maxindegree ~19.6M, maxoutdegree
# ~7.0M, nodes ~262.35M) all fit uint32, so no field needs 64-bit width.

SCHEMA: tuple[Field, ...] = (
    # Release-level (header): one value per release, not repeated per row.
    Field("release_id", "str", FieldLocation.HEADER, FieldKind.META, None, "Stable release id."),
    Field(
        "source_crawls",
        "list[str]",
        FieldLocation.HEADER,
        FieldKind.META,
        None,
        "Constituent crawl ids (<=3, rolling-window policy).",
    ),
    Field(
        "model_space_version",
        "str",
        FieldLocation.HEADER,
        FieldKind.META,
        None,
        "Embedding space tag; vectors comparable only within the same value.",
    ),
    # Per-node default artifact.
    Field(
        "host_or_domain_key",
        "str",
        FieldLocation.ROW_DEFAULT,
        FieldKind.KEY,
        None,
        "Stable host or domain/PLD key (the join identity).",
    ),
    Field(
        "release_local_node_id",
        "uint32",
        FieldLocation.ROW_DEFAULT,
        FieldKind.SCALAR,
        4,
        "Ephemeral per-release integer id (for joining to this release's graph).",
    ),
    Field(
        "embedding",
        "float32",
        FieldLocation.ROW_DEFAULT,
        FieldKind.VECTOR,
        None,
        "Default L2-normalized embedding (width = dim x precision).",
    ),
    Field(
        "confidence_bucket",
        "uint8",
        FieldLocation.ROW_DEFAULT,
        FieldKind.SCALAR,
        1,
        "Reproducible evidence-based confidence (high/medium/low).",
    ),
    Field(
        "feature_availability",
        "uint8",
        FieldLocation.ROW_DEFAULT,
        FieldKind.SCALAR,
        1,
        "Bitmask of which feature groups were available for this node.",
    ),
    Field(
        "in_degree", "uint32", FieldLocation.ROW_DEFAULT, FieldKind.SCALAR, 4, "In-links in release"
    ),
    Field(
        "out_degree",
        "uint32",
        FieldLocation.ROW_DEFAULT,
        FieldKind.SCALAR,
        4,
        "Out-links in release.",
    ),
    Field(
        "is_dangling",
        "uint8",
        FieldLocation.ROW_DEFAULT,
        FieldKind.SCALAR,
        1,
        "1 if the node has no out-links.",
    ),
    Field(
        "component_bucket",
        "uint8",
        FieldLocation.ROW_DEFAULT,
        FieldKind.SCALAR,
        1,
        "Giant-SCC membership / coarse SCC-size bucket.",
    ),
    Field(
        "pagerank",
        "float32",
        FieldLocation.ROW_DEFAULT,
        FieldKind.SCALAR,
        4,
        "Directed PageRank (from wgl.features).",
    ),
    Field(
        "harmonic_centrality",
        "float32",
        FieldLocation.ROW_DEFAULT,
        FieldKind.SCALAR,
        4,
        "Harmonic centrality where available.",
    ),
    # Optional sidecars (kept out of the default download).
    Field(
        "reverse_pagerank",
        "float32",
        FieldLocation.ROW_SIDECAR,
        FieldKind.SCALAR,
        4,
        "Reverse (in-link) PageRank.",
    ),
    Field(
        "source_role_embedding",
        "float32",
        FieldLocation.ROW_SIDECAR,
        FieldKind.VECTOR,
        None,
        "Out-role/source component vector (when published separately).",
    ),
    Field(
        "target_role_embedding",
        "float32",
        FieldLocation.ROW_SIDECAR,
        FieldKind.VECTOR,
        None,
        "In-role/target component vector (when published separately).",
    ),
    Field(
        "scc_id",
        "uint32",
        FieldLocation.ROW_SIDECAR,
        FieldKind.SCALAR,
        4,
        "Full strongly-connected-component id (diagnostic).",
    ),
)


def fields_at(location: FieldLocation) -> tuple[Field, ...]:
    """Return the schema fields at the given location."""
    return tuple(f for f in SCHEMA if f.location is location)


def default_row_fields() -> tuple[Field, ...]:
    """Per-node fields shipped in the default artifact."""
    return fields_at(FieldLocation.ROW_DEFAULT)


def sidecar_row_fields() -> tuple[Field, ...]:
    """Per-node fields kept in optional sidecars."""
    return fields_at(FieldLocation.ROW_SIDECAR)


def header_fields() -> tuple[Field, ...]:
    """Release-level header fields."""
    return fields_at(FieldLocation.HEADER)


def fixed_metadata_bytes_per_row() -> int:
    """Fixed-width on-disk bytes per default row, excluding the variable key and the embedding.

    This is the metadata overhead the size model adds on top of the embedding bytes (the key is
    sized separately because it is variable-length).
    """
    return sum(
        f.nbytes
        for f in default_row_fields()
        if f.kind is FieldKind.SCALAR and f.nbytes is not None
    )
