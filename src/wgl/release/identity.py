"""Canonical identity keys for the v1 release artifact.

Common Crawl distributes the host graph with **reversed** host names (``com.example.www``) and
**per-release, 0-based contiguous integer node ids**. The integer ids are *not* stable across
monthly releases — a host gets a fresh id every crawl — so they cannot be the artifact's identity.
The stable identifiers published in the artifact are *string keys* derived deterministically from
the host name:

* :func:`host_key`   — the canonical normalized host (``www.example.com``);
* :func:`domain_key` — the registrable domain / pay-level domain, a.k.a. PLD (``example.com``).

Both are pure functions of the input string (no network, no per-release state) so the same host maps
to the same key in every release. The registrable domain is computed with the offline ``tldextract``
snapshot (no network fetch), matching :mod:`wgl.tasks.tld`.

This module also provides two small, *evidence-based* (not model-based) classifiers used in the
release metadata: :func:`evidence_cohort` (the structural cohort from the runbook's coverage table)
and :func:`confidence_bucket` (a reproducible confidence label derived only from observable graph
evidence). See [`docs/release-contract.md`](../../../docs/release-contract.md).
"""

from __future__ import annotations

from enum import StrEnum

import tldextract

# Offline extractor: rely on the snapshot bundled with tldextract (no network fetch), exactly as
# wgl.tasks.tld does, so host_to_tld and domain_key agree on the public-suffix boundary.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())


def forward_host(name: str, reversed_host: bool = True) -> str:
    """Return the forward (human) host name for a node name.

    Args:
        name: The host name, possibly in Common Crawl reversed notation (``com.example.www``).
        reversed_host: Whether ``name`` is reversed. When False, ``name`` is returned normalized
            but with its label order unchanged.

    Returns:
        The forward host (``www.example.com``), lowercased with surrounding/trailing dots stripped.
    """
    text = name.strip().strip(".").lower()
    if not text:
        return ""
    if reversed_host:
        return ".".join(reversed(text.split(".")))
    return text


def host_key(name: str, reversed_host: bool = True) -> str:
    """Return the canonical, normalized **host key** for a node name.

    The host key is the stable cross-release identifier for a host vertex. It is the forward host
    name, lowercased, with whitespace and surrounding dots stripped. We deliberately keep the name
    as published (ASCII/punycode labels are left as-is rather than IDNA-decoded) so the key is a
    pure, reversible function of the source string — see the release contract for the rationale.

    Args:
        name: The (possibly reversed) Common Crawl host name.
        reversed_host: Whether ``name`` is in reversed notation.

    Returns:
        The canonical host key (``www.example.com``); empty string for an empty input.
    """
    return forward_host(name, reversed_host=reversed_host)


def domain_key(name: str, reversed_host: bool = True) -> str:
    """Return the **domain/PLD key** (registrable domain) for a node name.

    The registrable domain (eTLD+1, Common Crawl's "pay-level domain") groups every host under its
    owning domain — both an independently useful artifact and the fallback representation for sparse
    hosts. Computed from the public-suffix list via the offline ``tldextract`` snapshot.

    Args:
        name: The (possibly reversed) Common Crawl host name.
        reversed_host: Whether ``name`` is in reversed notation.

    Returns:
        The registrable domain (``example.com``). When the host has no registrable domain above
        its public suffix (e.g. the name *is* a public suffix, or an IP/onion address), falls back
        to the canonical host key so every vertex still maps to a non-empty domain key.
    """
    host = forward_host(name, reversed_host=reversed_host)
    if not host:
        return ""
    ext = _EXTRACT(host)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    # No registrable domain above the public suffix (bare suffix, IP, etc.): host is its own domain.
    return host


class EvidenceCohort(StrEnum):
    """Structural cohort of a host from its directed graph evidence (runbook-v2 coverage table)."""

    BIDIRECTIONAL = "bidirectional"  # has both in-links and out-links
    IN_ONLY = "in_only"  # has in-links, no out-links (dangling sink)
    OUT_ONLY = "out_only"  # has out-links, no in-links (dangling source)
    ISOLATED = "isolated"  # graph-isolated after preprocessing (no edges)


class ConfidenceBucket(StrEnum):
    """Reproducible confidence label from graph evidence only (not model-certainty claims)."""

    HIGH = "high"  # well-observed: links in both directions
    MEDIUM = "medium"  # one-directional evidence
    LOW = "low"  # graph-isolated; relies on metadata / domain fallback


def evidence_cohort(in_degree: int, out_degree: int) -> EvidenceCohort:
    """Classify a host into its structural evidence cohort.

    Args:
        in_degree: Number of in-links (predecessors) in the release graph.
        out_degree: Number of out-links (successors) in the release graph.

    Returns:
        The :class:`EvidenceCohort`.
    """
    has_in = in_degree > 0
    has_out = out_degree > 0
    if has_in and has_out:
        return EvidenceCohort.BIDIRECTIONAL
    if has_in:
        return EvidenceCohort.IN_ONLY
    if has_out:
        return EvidenceCohort.OUT_ONLY
    return EvidenceCohort.ISOLATED


def confidence_bucket(in_degree: int, out_degree: int) -> ConfidenceBucket:
    """Map graph evidence to a reproducible confidence bucket.

    The bucket reflects *how much directed evidence* supports the host's representation, not any
    model-certainty claim — so it is deterministic and reproducible from the published metadata.

    Args:
        in_degree: Number of in-links in the release graph.
        out_degree: Number of out-links in the release graph.

    Returns:
        The :class:`ConfidenceBucket`.
    """
    cohort = evidence_cohort(in_degree, out_degree)
    if cohort is EvidenceCohort.BIDIRECTIONAL:
        return ConfidenceBucket.HIGH
    if cohort is EvidenceCohort.ISOLATED:
        return ConfidenceBucket.LOW
    return ConfidenceBucket.MEDIUM


def is_dangling(out_degree: int) -> bool:
    """Whether a host is *dangling* (no out-links) — the dominant CC host-graph cohort (~75%)."""
    return out_degree == 0
