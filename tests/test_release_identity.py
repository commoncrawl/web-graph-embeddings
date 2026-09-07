"""Tests for the canonical release identity keys (host + domain/PLD)."""

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


def test_forward_host_unreverses_and_normalizes():
    assert forward_host("com.example.www") == "www.example.com"
    assert forward_host("  COM.Example.WWW. ") == "www.example.com"
    assert forward_host("www.example.com", reversed_host=False) == "www.example.com"
    assert forward_host("") == ""


def test_host_key_is_stable_canonical_form():
    # Same host in reversed and forward notation -> identical key.
    assert host_key("uk.co.bbc.www") == host_key("www.bbc.co.uk", reversed_host=False)
    assert host_key("uk.co.bbc.www") == "www.bbc.co.uk"


def test_domain_key_registrable_domain():
    assert domain_key("com.example.www") == "example.com"
    assert domain_key("com.example.blog.shop") == "example.com"
    assert domain_key("uk.co.bbc.www") == "bbc.co.uk"  # multi-label public suffix


def test_domain_key_falls_back_to_host_when_no_registrable_domain():
    # A bare public suffix has no eTLD+1 above it -> domain key falls back to the host key itself.
    assert domain_key("com") == "com"
    assert domain_key("uk.co") == "co.uk"


def test_hosts_under_same_domain_share_domain_key():
    a = domain_key("com.example.www")
    b = domain_key("com.example.mail")
    c = domain_key("com.example.blog.shop")
    assert a == b == c == "example.com"


def test_evidence_cohort():
    assert evidence_cohort(3, 2) is EvidenceCohort.BIDIRECTIONAL
    assert evidence_cohort(3, 0) is EvidenceCohort.IN_ONLY
    assert evidence_cohort(0, 2) is EvidenceCohort.OUT_ONLY
    assert evidence_cohort(0, 0) is EvidenceCohort.ISOLATED


def test_confidence_bucket_tracks_evidence():
    assert confidence_bucket(3, 2) is ConfidenceBucket.HIGH
    assert confidence_bucket(3, 0) is ConfidenceBucket.MEDIUM
    assert confidence_bucket(0, 2) is ConfidenceBucket.MEDIUM
    assert confidence_bucket(0, 0) is ConfidenceBucket.LOW


def test_is_dangling():
    assert is_dangling(0)
    assert not is_dangling(1)
