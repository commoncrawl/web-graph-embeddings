# CPU image (docker/cpu/Dockerfile) for cluster CPU partitions and local-parity runs,
# imported into enroot as a .sqsh (see ../enroot.md). The GPU sibling is docker/gpu/Dockerfile.
#
# Uses the CPU build of torch (small, no CUDA wheels) — matching the project's local dev
# environment, where pyproject pins the pytorch-cpu index. torch is installed explicitly from
# that index first, then the project + extras resolve against the already-present torch.
FROM python:3.12-slim

# Rootless podman (--isolation=chroot on the cluster builder) has no subordinate gids, so apt's
# privilege-dropping sandbox fails with "Could not switch group"; run apt as root inside the build.
RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/sandbox-disable

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Build toolchain for any sdist-only deps; uv for fast installs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
# Ignore the repo's [tool.uv] config inside the build: its `exclude-newer` drops every wheel from
# the PyTorch index (no upload timestamps there) and made the torch install unresolvable.
ENV UV_NO_CONFIG=1

WORKDIR /opt/web-graph-learning
COPY pyproject.toml README.md ./
COPY src ./src

# 1) CPU torch from the PyTorch CPU index (explicit, so we never pull CUDA wheels).
# 2) hatchling — the build backend, needed in-env because we install with
#    --no-build-isolation (which skips uv's isolated build env that would otherwise
#    fetch it). Without it the local wheel build fails: "No module named 'hatchling'".
# 3) The project + the faiss, cc & topic extras; torch is already satisfied.
#    (cc -> duckdb/pyarrow/boto3 for CC columnar-index + Athena/S3 label extraction;
#     topic -> huggingface_hub/zstandard for the WebOrganizer topic jobs in slurm/topic_cpu.sbatch.)
RUN uv pip install --system "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --system hatchling \
    && uv pip install --system --no-build-isolation ".[faiss,cc,topic]" \
    && python -c "import wgl, torch, boto3, duckdb, huggingface_hub, zstandard; print('wgl ok, torch', torch.__version__, 'boto3', boto3.__version__)"

# Vendored OSS (PyTorch-BigGraph / MariusGNN / Cleora / DiskGNN) is added under vendor/ and
# installed here as needed — see vendor/README.md.

ENTRYPOINT ["wgl"]
CMD ["info"]
