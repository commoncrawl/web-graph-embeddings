"""Per-release manifest and the cross-release identity index.

A **release manifest** records, for one monthly artifact, every host vertex's stable key together
with the *ephemeral* per-release local node id and the graph evidence used for confidence/cohort
metadata. The **cross-release index** joins consecutive manifests on the stable host key to produce
a durable identity table with lifecycle states (``active``/``new``/``missing``/``reappeared``) — the
mechanism that lets users compare consecutive releases even though Common Crawl reassigns
integer node ids every crawl.

This is the Phase 0.1 contract object. At full web scale the per-release host table is sharded
(Phase 3.3); here it serializes to a small JSON header + a TSV of host rows, which is enough for the
synthetic-release gate and for downsampled CC slices.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from wgl.data.graph import DirectedGraph
from wgl.release.identity import (
    confidence_bucket,
    domain_key,
    evidence_cohort,
    host_key,
    is_dangling,
)

# Bumped when the manifest/host-row schema changes in a backward-incompatible way.
SCHEMA_VERSION = "0.1.0"

_HOST_COLUMNS = (
    "host_key",
    "domain_key",
    "local_node_id",
    "in_degree",
    "out_degree",
    "is_dangling",
    "cohort",
    "confidence",
)


class HostState(StrEnum):
    """Lifecycle state of a host key *as of* a given release, relative to prior releases."""

    NEW = "new"  # present now, never seen in any earlier release
    ACTIVE = "active"  # present now and in the immediately-preceding release (persistent)
    REAPPEARED = "reappeared"  # present now, absent in the previous release, seen in an earlier one
    MISSING = "missing"  # absent now, but present in some earlier release


@dataclass(frozen=True)
class ReleaseRecord:
    """One host's row within a single release manifest.

    Attributes:
        host_key: Canonical stable host key (:func:`wgl.release.identity.host_key`).
        domain_key: Registrable domain / PLD key.
        local_node_id: The host's integer id *in this release's graph* (ephemeral across releases).
        in_degree: In-links in this release.
        out_degree: Out-links in this release.
        is_dangling: Whether the host has no out-links.
        cohort: Structural evidence cohort value.
        confidence: Reproducible confidence-bucket value.
    """

    host_key: str
    domain_key: str
    local_node_id: int
    in_degree: int
    out_degree: int
    is_dangling: bool
    cohort: str
    confidence: str

    @classmethod
    def from_evidence(
        cls, key: str, dom: str, local_node_id: int, in_degree: int, out_degree: int
    ) -> ReleaseRecord:
        """Build a record, deriving the cohort/confidence/dangling fields from degrees."""
        return cls(
            host_key=key,
            domain_key=dom,
            local_node_id=int(local_node_id),
            in_degree=int(in_degree),
            out_degree=int(out_degree),
            is_dangling=is_dangling(out_degree),
            cohort=evidence_cohort(in_degree, out_degree).value,
            confidence=confidence_bucket(in_degree, out_degree).value,
        )


@dataclass
class ReleaseManifest:
    """A single release's host manifest: stable keys -> per-release evidence.

    Attributes:
        release_id: Stable release identifier (e.g. ``cc-main-2026-mar-apr-may``).
        source_crawls: The constituent monthly crawl ids (1-3 under the rolling-window policy).
        records: Host rows keyed by ``host_key`` (insertion order = local node id order).
        schema_version: Manifest schema version.
        model_space_version: Embedding model-space tag; ``None`` until embeddings are trained.
    """

    release_id: str
    source_crawls: list[str]
    records: dict[str, ReleaseRecord] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    model_space_version: str | None = None

    @property
    def host_keys(self) -> set[str]:
        """The set of host keys present in this release."""
        return set(self.records)

    def __len__(self) -> int:
        """Number of host records in the manifest."""
        return len(self.records)

    @classmethod
    def from_graph(
        cls,
        graph: DirectedGraph,
        release_id: str,
        source_crawls: Sequence[str],
        reversed_host: bool = True,
        model_space_version: str | None = None,
    ) -> ReleaseManifest:
        """Build a manifest from a prepared :class:`DirectedGraph`.

        Args:
            graph: The release graph (its ``node_names`` provide the host strings).
            release_id: Stable release identifier.
            source_crawls: Constituent crawl ids for this release.
            reversed_host: Whether ``node_names`` use Common Crawl reversed notation.
            model_space_version: Optional model-space tag.

        Returns:
            The populated :class:`ReleaseManifest`.

        Raises:
            ValueError: If the graph has no node names, or two nodes collide on the same host key.
        """
        if graph.node_names is None:
            msg = "ReleaseManifest.from_graph requires graph.node_names"
            raise ValueError(msg)
        in_deg = graph.in_degree()
        out_deg = graph.out_degree()
        records: dict[str, ReleaseRecord] = {}
        for nid, name in enumerate(graph.node_names):
            key = host_key(name, reversed_host=reversed_host)
            if key in records:
                msg = f"duplicate host key {key!r} (nodes {records[key].local_node_id} and {nid})"
                raise ValueError(msg)
            records[key] = ReleaseRecord.from_evidence(
                key, domain_key(name, reversed_host=reversed_host), nid, in_deg[nid], out_deg[nid]
            )
        return cls(
            release_id=release_id,
            source_crawls=list(source_crawls),
            records=records,
            model_space_version=model_space_version,
        )

    def save(self, directory: str | Path) -> None:
        """Persist the manifest to ``directory`` (``manifest.json`` header + ``hosts.tsv`` rows)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        header = {
            "release_id": self.release_id,
            "source_crawls": self.source_crawls,
            "schema_version": self.schema_version,
            "model_space_version": self.model_space_version,
            "num_hosts": len(self.records),
        }
        (directory / "manifest.json").write_text(json.dumps(header, indent=2))
        with (directory / "hosts.tsv").open("w", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(_HOST_COLUMNS)
            for rec in self.records.values():
                writer.writerow(
                    [
                        rec.host_key,
                        rec.domain_key,
                        rec.local_node_id,
                        rec.in_degree,
                        rec.out_degree,
                        int(rec.is_dangling),
                        rec.cohort,
                        rec.confidence,
                    ]
                )

    @classmethod
    def load(cls, directory: str | Path) -> ReleaseManifest:
        """Load a manifest previously written by :meth:`save`."""
        directory = Path(directory)
        header = json.loads((directory / "manifest.json").read_text())
        records: dict[str, ReleaseRecord] = {}
        with (directory / "hosts.tsv").open(newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                records[row["host_key"]] = ReleaseRecord(
                    host_key=row["host_key"],
                    domain_key=row["domain_key"],
                    local_node_id=int(row["local_node_id"]),
                    in_degree=int(row["in_degree"]),
                    out_degree=int(row["out_degree"]),
                    is_dangling=bool(int(row["is_dangling"])),
                    cohort=row["cohort"],
                    confidence=row["confidence"],
                )
        return cls(
            release_id=header["release_id"],
            source_crawls=list(header["source_crawls"]),
            records=records,
            schema_version=header.get("schema_version", SCHEMA_VERSION),
            model_space_version=header.get("model_space_version"),
        )


@dataclass(frozen=True)
class CrossReleaseEntry:
    """A host key's durable cross-release identity row.

    Attributes:
        host_key: The stable host key.
        first_release: Release id where the key was first observed.
        last_seen_release: Most recent release where the key was present.
        state: Lifecycle state as of the latest release in the index.
        local_node_ids: Map ``release_id -> local node id`` for releases where the key was present.
    """

    host_key: str
    first_release: str
    last_seen_release: str
    state: HostState
    local_node_ids: dict[str, int]


@dataclass
class CrossReleaseIndex:
    """Stable identity table joining an ordered sequence of release manifests on the host key.

    The lifecycle ``state`` of each key is computed *relative to the latest release* in the
    sequence: a key present in the latest release is ``new`` (never seen before), ``active`` (also
    in the immediately-preceding release), or ``reappeared`` (absent in the previous release but
    seen earlier); a key absent from the latest release but seen in any earlier one is ``missing``.
    """

    release_ids: list[str]
    entries: dict[str, CrossReleaseEntry]

    @classmethod
    def build(cls, manifests: Iterable[ReleaseManifest]) -> CrossReleaseIndex:
        """Join an *ordered* (oldest-to-newest) sequence of manifests into a cross-release index.

        Args:
            manifests: Release manifests in chronological order. At least one is required.

        Returns:
            The :class:`CrossReleaseIndex`.

        Raises:
            ValueError: If no manifests are given, or two share a release id.
        """
        mans = list(manifests)
        if not mans:
            msg = "CrossReleaseIndex.build requires at least one manifest"
            raise ValueError(msg)
        release_ids = [m.release_id for m in mans]
        if len(set(release_ids)) != len(release_ids):
            msg = f"duplicate release ids: {release_ids}"
            raise ValueError(msg)

        latest_id = release_ids[-1]
        prev_id = release_ids[-2] if len(mans) >= 2 else None

        entries: dict[str, CrossReleaseEntry] = {}
        all_keys = set().union(*(m.host_keys for m in mans))
        for key in all_keys:
            present = [m for m in mans if key in m.records]
            local_ids = {m.release_id: m.records[key].local_node_id for m in present}
            entries[key] = CrossReleaseEntry(
                host_key=key,
                first_release=present[0].release_id,
                last_seen_release=present[-1].release_id,
                state=_state_for(set(local_ids), latest_id, prev_id),
                local_node_ids=local_ids,
            )
        return cls(release_ids=release_ids, entries=entries)

    def state_counts(self) -> dict[str, int]:
        """Return a ``{state_value: count}`` summary over all entries."""
        return dict(Counter(e.state.value for e in self.entries.values()))

    def keys_in_state(self, state: HostState) -> set[str]:
        """Return the host keys currently in the given lifecycle ``state``."""
        return {k for k, e in self.entries.items() if e.state is state}


def _state_for(present_releases: set[str], latest_id: str, prev_id: str | None) -> HostState:
    """Lifecycle state of a key from the set of release ids it appears in.

    Args:
        present_releases: Release ids in which the key is present (non-empty — every key reaches
            this function from the union of all manifests, so ``MISSING`` is the residual case).
        latest_id: The newest release id in the index.
        prev_id: The immediately-preceding release id, or ``None`` when only one release exists.

    Returns:
        ``MISSING`` if absent from the latest release; otherwise ``ACTIVE`` (also in the previous
        release), ``REAPPEARED`` (absent in the previous release but seen in an earlier one), or
        ``NEW`` (present only in the latest release).
    """
    if latest_id not in present_releases:
        return HostState.MISSING
    if prev_id is not None and prev_id in present_releases:
        return HostState.ACTIVE
    # Present now but not in the immediately-previous release: seen earlier => reappeared, else new.
    seen_before_latest = present_releases - {latest_id}
    return HostState.REAPPEARED if seen_before_latest else HostState.NEW
