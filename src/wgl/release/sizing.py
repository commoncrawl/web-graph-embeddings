"""Artifact size model for selecting the v1 embedding dimension and precision (Phase 0.2).

Estimates the uncompressed and (approximate) compressed footprint of the host and domain artifacts
across candidate dimensions (64/128/256) and precisions (fp32/fp16/int8). The dominant term is the
embedding matrix (``num_nodes x dim x bytes_per_value``); the schema's fixed-width metadata
(:func:`wgl.release.schema.fixed_metadata_bytes_per_row`) plus the variable key add a smaller,
dimension-independent overhead.

Compression ratios are deliberate, documented estimates: L2-normalized float vectors are
high-entropy and compress poorly, so the ratios are near 1.0 for floats and lower only for int8 and
for the repetitive metadata column block. They are decision aids, not guarantees — the real ratio is
measured at export time (Phase 3.3). Node counts come from the real cc-main-2026-mar-apr-may host
stats; the domain count is an estimate until the domain graph is extracted.
"""

from __future__ import annotations

from dataclasses import dataclass

from wgl.release.schema import fixed_metadata_bytes_per_row

# Real host-graph node count (cc-main-2026-mar-apr-may host.stats: nodes=262351908).
CC_HOST_NODES = 262_351_908
# Domain/PLD node count is not yet extracted on the cluster (host graph only). Registrable domains
# typically number ~40-55% of hosts for these crawls; use ~135M as a planning estimate, replaced by
# the measured count when the domain graph is prepared.
CC_DOMAIN_NODES_ESTIMATE = 135_000_000

# On-disk bytes per embedding value by precision label.
PRECISION_BYTES = {"fp32": 4, "fp16": 2, "int8": 1}

# Approximate compression ratios (compressed / uncompressed). Floats barely compress; int8 quantized
# vectors and the small-int metadata block compress more. Estimates — verified at export time.
_VECTOR_COMPRESSION = {"fp32": 0.95, "fp16": 0.95, "int8": 0.70}
_METADATA_COMPRESSION = 0.45

# Average bytes for a stable key string (forward host / domain). Estimate; tune from real keys.
DEFAULT_AVG_KEY_BYTES = 24

_GIB = 1024**3


@dataclass(frozen=True)
class SizeEstimate:
    """Estimated footprint of one artifact at a given dimension and precision.

    Attributes:
        num_nodes: Number of rows (hosts or domains).
        dim: Embedding dimension.
        precision: Precision label (``fp32``/``fp16``/``int8``).
        vector_bytes: Uncompressed embedding-matrix bytes.
        metadata_bytes: Uncompressed default-row metadata bytes (key + fixed fields).
        total_uncompressed: ``vector_bytes + metadata_bytes``.
        total_compressed: Approximate compressed total.
    """

    num_nodes: int
    dim: int
    precision: str
    vector_bytes: int
    metadata_bytes: int
    total_uncompressed: int
    total_compressed: int

    @property
    def vector_gib(self) -> float:
        """Uncompressed embedding-matrix size in GiB."""
        return self.vector_bytes / _GIB

    @property
    def total_uncompressed_gib(self) -> float:
        """Uncompressed total in GiB."""
        return self.total_uncompressed / _GIB

    @property
    def total_compressed_gib(self) -> float:
        """Approximate compressed total in GiB."""
        return self.total_compressed / _GIB


def estimate_size(
    num_nodes: int,
    dim: int,
    precision: str = "fp32",
    avg_key_bytes: int = DEFAULT_AVG_KEY_BYTES,
) -> SizeEstimate:
    """Estimate the footprint of one artifact.

    Args:
        num_nodes: Number of rows (hosts or domains).
        dim: Embedding dimension (the *published* default vector width, not any internal
            ComplEx real+imag doubling).
        precision: One of ``fp32``/``fp16``/``int8``.
        avg_key_bytes: Average bytes for a stable key string.

    Returns:
        The :class:`SizeEstimate`.

    Raises:
        ValueError: On an unknown precision or non-positive dim/nodes.
    """
    if precision not in PRECISION_BYTES:
        msg = f"unknown precision {precision!r}; expected one of {sorted(PRECISION_BYTES)}"
        raise ValueError(msg)
    if dim <= 0 or num_nodes <= 0:
        msg = f"dim and num_nodes must be positive (got dim={dim}, num_nodes={num_nodes})"
        raise ValueError(msg)

    vector_bytes = num_nodes * dim * PRECISION_BYTES[precision]
    metadata_bytes = num_nodes * (avg_key_bytes + fixed_metadata_bytes_per_row())
    total_uncompressed = vector_bytes + metadata_bytes
    total_compressed = round(
        vector_bytes * _VECTOR_COMPRESSION[precision] + metadata_bytes * _METADATA_COMPRESSION
    )
    return SizeEstimate(
        num_nodes=num_nodes,
        dim=dim,
        precision=precision,
        vector_bytes=vector_bytes,
        metadata_bytes=metadata_bytes,
        total_uncompressed=total_uncompressed,
        total_compressed=total_compressed,
    )


def size_table(
    num_nodes: int,
    dims: tuple[int, ...] = (64, 128, 256),
    precisions: tuple[str, ...] = ("fp32", "fp16", "int8"),
    avg_key_bytes: int = DEFAULT_AVG_KEY_BYTES,
) -> list[SizeEstimate]:
    """Estimate sizes across the candidate (dim, precision) grid for one artifact."""
    return [
        estimate_size(num_nodes, dim, precision, avg_key_bytes)
        for dim in dims
        for precision in precisions
    ]


def format_table(estimates: list[SizeEstimate], label: str) -> str:
    """Render a human-readable size table (GiB) for a list of estimates."""
    lines = [
        f"{label} ({estimates[0].num_nodes:,} nodes)" if estimates else f"{label} (no rows)",
        f"  {'dim':>4}  {'prec':>5}  {'vectors':>10}  {'uncompressed':>13}  {'compressed':>11}",
    ]
    lines.extend(
        f"  {e.dim:>4}  {e.precision:>5}  {e.vector_gib:>8.1f}GiB  "
        f"{e.total_uncompressed_gib:>11.1f}GiB  {e.total_compressed_gib:>9.1f}GiB"
        for e in estimates
    )
    return "\n".join(lines)
