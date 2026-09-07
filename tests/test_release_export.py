"""Tests for the Parquet shard-writer (`wgl release shard-export`) that publishes the CC table."""

import hashlib
import json

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from wgl.release.export import export_parquet_shards as export_shards  # noqa: E402 (importorskip)

_NAMES = [
    "com.example.www",  # -> www.example.com / example.com
    "uk.co.bbc.www",  # -> www.bbc.co.uk / bbc.co.uk
    "org.wikipedia.en",  # -> en.wikipedia.org / wikipedia.org
    "com.example.blog",  # -> blog.example.com / example.com
    "net.cloudflare",  # -> cloudflare.net / cloudflare.net
]


def _write_inputs(tmp_path, n, dim=4):
    rng = np.random.default_rng(0)
    emb = (rng.standard_normal((n, dim)) * 3.0).astype(np.float32)  # non-unit magnitudes
    emb_path = tmp_path / "emb.npy"
    np.save(emb_path, emb)
    names_path = tmp_path / "node_names.txt"
    # Last line intentionally has NO trailing newline (mirrors the real node_names.txt).
    names_path.write_text("\n".join(_NAMES[:n]), encoding="utf-8")
    return emb, emb_path, names_path


def test_shards_cover_all_rows_and_align(tmp_path):
    emb, emb_path, names_path = _write_inputs(tmp_path, n=5, dim=4)
    out = tmp_path / "hf"
    export_shards(
        emb_path,
        names_path,
        out,
        rows_per_shard=2,  # -> 3 shards (2, 2, 1)
        normalize=True,
        precision="both",
        release_id="cc-main-test",
        source_crawls=["CC-A", "CC-B"],
        model_space_version="test_v1",
        meta={"encoder": "shallow", "test_mrr": 0.95},
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["num_rows"] == 5
    assert manifest["num_shards"] == 3
    assert manifest["schema_version"]  # populated from the contract constant
    assert manifest["normalized"] is True
    assert manifest["metrics"]["test_mrr"] == 0.95
    assert manifest["canonical_precision"] == "float32"
    assert manifest["precisions"] == {"vectors": "float32", "vectors_fp16": "float16"}

    # Reassemble the fp32 shards in row order and check identity + normalized vectors.
    host_keys, dom_keys, rows, vecs = [], [], [], []
    for shard in manifest["shards"]["vectors"]:
        t = pq.read_table(out / shard["filename"]).to_pydict()
        host_keys += t["host_key"]
        dom_keys += t["domain_key"]
        rows += t["row_id"]
        vecs += [np.asarray(v, dtype=np.float32) for v in t["embedding"]]
    assert rows == list(range(5))  # contiguous, in order
    assert host_keys[0] == "www.example.com"
    assert dom_keys[1] == "bbc.co.uk"
    assert dom_keys[4] == "cloudflare.net"  # bare-suffix host falls back to itself

    mat = np.vstack(vecs)
    np.testing.assert_allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-6)  # L2-normalized
    # Direction preserved vs the raw input (cosine unchanged by normalization).
    raw_unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    np.testing.assert_allclose(mat, raw_unit, atol=1e-6)


def test_manifest_sha256_matches_files(tmp_path):
    _, emb_path, names_path = _write_inputs(tmp_path, n=5, dim=4)
    out = tmp_path / "hf"
    export_shards(
        emb_path,
        names_path,
        out,
        rows_per_shard=2,
        normalize=True,
        precision="fp32",
        release_id="cc-main-test",
        source_crawls=["CC-A"],
        model_space_version="test_v1",
        meta={},
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert "vectors_fp16" not in manifest["shards"]
    for shard in manifest["shards"]["vectors"]:
        digest = hashlib.sha256((out / shard["filename"]).read_bytes()).hexdigest()
        assert digest == shard["sha256"]
        assert shard["n_rows"] == shard["row_end"] - shard["row_start"]


def test_no_normalize_keeps_magnitude(tmp_path):
    emb, emb_path, names_path = _write_inputs(tmp_path, n=3, dim=4)
    out = tmp_path / "hf"
    export_shards(
        emb_path,
        names_path,
        out,
        rows_per_shard=4,
        normalize=False,
        precision="fp32",
        release_id="cc-main-test",
        source_crawls=["CC-A"],
        model_space_version="test_v1",
        meta={},
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["normalized"] is False
    t = pq.read_table(out / manifest["shards"]["vectors"][0]["filename"]).to_pydict()
    vecs = np.vstack([np.asarray(v, dtype=np.float32) for v in t["embedding"]])
    np.testing.assert_allclose(vecs, emb, atol=1e-6)


def test_more_names_than_rows_is_rejected(tmp_path):
    # 5 embedding rows but 6 names -> misalignment must be caught.
    _, emb_path, names_path = _write_inputs(tmp_path, n=5, dim=4)
    names_path.write_text("\n".join([*_NAMES, "com.extra"]), encoding="utf-8")
    with pytest.raises(ValueError, match="misaligned"):
        export_shards(
            emb_path,
            names_path,
            tmp_path / "hf",
            rows_per_shard=2,
            normalize=True,
            precision="fp32",
            release_id="cc-main-test",
            source_crawls=["CC-A"],
            model_space_version="test_v1",
            meta={},
        )


def test_fp16_only_is_the_primary_tier(tmp_path):
    emb, emb_path, names_path = _write_inputs(tmp_path, n=5, dim=4)
    out = tmp_path / "hf"
    export_shards(
        emb_path,
        names_path,
        out,
        rows_per_shard=2,
        normalize=True,
        precision="fp16",
        release_id="cc-main-test",
        source_crawls=["CC-A"],
        model_space_version="test_v1",
        meta={},
    )
    manifest = json.loads((out / "manifest.json").read_text())
    # fp16 is written to the primary "vectors/" dir (no separate fp32 set, no sidecar).
    assert list(manifest["shards"].keys()) == ["vectors"]
    assert manifest["canonical_precision"] == "float16"
    assert manifest["precisions"] == {"vectors": "float16"}

    vecs = []
    for shard in manifest["shards"]["vectors"]:
        t = pq.read_table(out / shard["filename"])
        assert t.schema.field("embedding").type.value_type == pa.float16()
        vecs += [np.asarray(v, dtype=np.float32) for v in t.column("embedding").to_pylist()]
    mat = np.vstack(vecs)
    # fp16 round-trip stays within half-precision tolerance of the L2-normalized direction.
    raw_unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    np.testing.assert_allclose(mat, raw_unit, atol=1e-3)
