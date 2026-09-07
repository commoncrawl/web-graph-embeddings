"""Tests for the declarative v1 artifact schema."""

from wgl.release.schema import (
    FieldKind,
    FieldLocation,
    default_row_fields,
    fixed_metadata_bytes_per_row,
    header_fields,
    sidecar_row_fields,
)


def test_default_and_sidecar_fields_disjoint_and_nonempty():
    default = {f.name for f in default_row_fields()}
    sidecar = {f.name for f in sidecar_row_fields()}
    assert default and sidecar
    assert default.isdisjoint(sidecar)


def test_required_default_fields_present():
    default = {f.name for f in default_row_fields()}
    for required in (
        "host_or_domain_key",
        "release_local_node_id",
        "embedding",
        "confidence_bucket",
        "in_degree",
        "out_degree",
        "is_dangling",
    ):
        assert required in default


def test_embedding_is_a_vector_with_no_fixed_width():
    (emb,) = [f for f in default_row_fields() if f.name == "embedding"]
    assert emb.kind is FieldKind.VECTOR
    assert emb.nbytes is None  # width = dim x precision, handled by the size model


def test_key_field_is_variable_width():
    (key,) = [f for f in default_row_fields() if f.name == "host_or_domain_key"]
    assert key.kind is FieldKind.KEY
    assert key.nbytes is None


def test_header_holds_release_level_fields():
    header = {f.name for f in header_fields()}
    assert {"release_id", "source_crawls", "model_space_version"} <= header
    # Header fields are never per-row.
    for f in header_fields():
        assert f.location is FieldLocation.HEADER


def test_fixed_metadata_bytes_per_row():
    # uint32 local_id(4) + confidence(1) + feature_avail(1) + in_deg(4) + out_deg(4)
    # + is_dangling(1) + component_bucket(1) + pagerank(4) + harmonic(4) = 24
    assert fixed_metadata_bytes_per_row() == 24
