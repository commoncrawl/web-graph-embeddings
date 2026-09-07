"""V1 release contract: canonical identity keys and cross-release manifests.

This package implements the *artifact contract* frozen in
[`docs/release-contract.md`](../../../docs/release-contract.md) (runbook-v2 Phase 0):

* :mod:`wgl.release.identity` — canonical, normalized host and domain/PLD keys (the stable
  identifiers that survive across monthly releases, independent of Common Crawl's per-release
  integer node ids);
* :mod:`wgl.release.manifest` — a per-release manifest (stable key -> per-release local node id +
  graph evidence) and the cross-release index that joins consecutive releases into a stable identity
  table with ``active``/``new``/``missing``/``reappeared`` lifecycle states.

These are deliberately model-agnostic: they describe *what is released and how it is joined* before
any embedding is trained.
"""

from __future__ import annotations

from wgl.release.identity import (
    ConfidenceBucket,
    EvidenceCohort,
    confidence_bucket,
    domain_key,
    evidence_cohort,
    forward_host,
    host_key,
    is_dangling,
)
from wgl.release.manifest import (
    SCHEMA_VERSION,
    CrossReleaseEntry,
    CrossReleaseIndex,
    HostState,
    ReleaseManifest,
    ReleaseRecord,
)
from wgl.release.schema import (
    SCHEMA,
    Field,
    FieldLocation,
    default_row_fields,
    fixed_metadata_bytes_per_row,
    sidecar_row_fields,
)
from wgl.release.sizing import (
    CC_DOMAIN_NODES_ESTIMATE,
    CC_HOST_NODES,
    SizeEstimate,
    estimate_size,
    format_table,
    size_table,
)

__all__ = [
    "CC_DOMAIN_NODES_ESTIMATE",
    "CC_HOST_NODES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "ConfidenceBucket",
    "CrossReleaseEntry",
    "CrossReleaseIndex",
    "EvidenceCohort",
    "Field",
    "FieldLocation",
    "HostState",
    "ReleaseManifest",
    "ReleaseRecord",
    "SizeEstimate",
    "confidence_bucket",
    "default_row_fields",
    "domain_key",
    "estimate_size",
    "evidence_cohort",
    "fixed_metadata_bytes_per_row",
    "format_table",
    "forward_host",
    "host_key",
    "is_dangling",
    "sidecar_row_fields",
    "size_table",
]
