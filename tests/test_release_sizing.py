"""Tests for the artifact size model."""

import pytest

from wgl.release.sizing import (
    CC_HOST_NODES,
    PRECISION_BYTES,
    estimate_size,
    size_table,
)


def test_vector_bytes_scale_with_precision():
    n, dim = CC_HOST_NODES, 128
    fp32 = estimate_size(n, dim, "fp32").vector_bytes
    fp16 = estimate_size(n, dim, "fp16").vector_bytes
    int8 = estimate_size(n, dim, "int8").vector_bytes
    assert fp32 == n * dim * 4
    assert fp16 * 2 == fp32  # half
    assert int8 * 4 == fp32  # quarter


def test_vector_bytes_scale_with_dim():
    n = 1_000_000
    s64 = estimate_size(n, 64, "fp32").vector_bytes
    s128 = estimate_size(n, 128, "fp32").vector_bytes
    s256 = estimate_size(n, 256, "fp32").vector_bytes
    assert s128 == 2 * s64
    assert s256 == 2 * s128


def test_metadata_is_dimension_independent():
    n = 1_000_000
    a = estimate_size(n, 64, "fp32").metadata_bytes
    b = estimate_size(n, 256, "int8").metadata_bytes
    assert a == b  # metadata cost depends on rows, not on embedding dim/precision


def test_total_is_vectors_plus_metadata():
    e = estimate_size(CC_HOST_NODES, 128, "fp32")
    assert e.total_uncompressed == e.vector_bytes + e.metadata_bytes
    assert e.total_compressed <= e.total_uncompressed


def test_host_128_fp32_matches_expected_gib():
    e = estimate_size(CC_HOST_NODES, 128, "fp32")
    # 262,351,908 * 128 * 4 bytes ~= 125 GiB of vectors.
    assert 124.0 < e.vector_gib < 126.0


def test_size_table_covers_grid():
    table = size_table(1_000_000, dims=(64, 128), precisions=("fp32", "fp16", "int8"))
    assert len(table) == 2 * 3
    assert {e.precision for e in table} == set(PRECISION_BYTES)
    assert {e.dim for e in table} == {64, 128}


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="precision"):
        estimate_size(1000, 128, "bf16")
    with pytest.raises(ValueError, match="positive"):
        estimate_size(1000, 0, "fp32")
    with pytest.raises(ValueError, match="positive"):
        estimate_size(0, 128, "fp32")
