# cuGraph-PyG image (docker/cugraph-pyg/Dockerfile) for the Phase 3.1 GPU-sampling backend (Option B).
#
# Based on NVIDIA's NGC PyG container, which is their recommended, version-matched vehicle for
# cuGraph-PyG + WholeGraph. Verified on-cluster (job 3131036, driver 595, GPU node): the base ships
# torch 2.12/nv + torch-geometric 2.8 + CUDA 13.x + RAPIDS, and
#   from cugraph_pyg.loader import LinkNeighborLoader          # GPU link-prediction sampling
#   from cugraph_pyg.data import GraphStore, FeatureStore      # GPU graph/feature stores
#   from pylibwholegraph.torch.embedding import WholeMemoryEmbeddingModule  # sharded trainable emb
# all import and torch.cuda.is_available() == True. This makes the whole "graph on GPU, sampling on
# GPU, embedding table sharded across GPUs (single->multi-GPU, same PyG API)" path turnkey — see
# docs/experiments/2026-06-30-phase3-backend-scoping.md (§3.1 REVISED). Requires a CUDA-13-capable
# driver (cluster has 595.71.05).
#
# We add only our thin glue on top: graph conversion (pyarrow; numpy/pandas already present), the
# profiling harness (tensorboard/psutil/nvidia-ml-py, reused from the GraphStorm work), and wandb for
# optional experiment tracking (cupyg_lp.py --wandb-project; W&B's monitor captures GPU/VRAM/CPU).
#
# Build on the cluster image pipeline (the 12 GB NGC base is cached from the inspection pull):
#   slurm_submit.sh -C <proj> build_image.sbatch --export=ALL,VARIANT=cupyg,IMAGE_VERSION=v1
#
# bf16 image bump DEFERRED (2026-07-10). We wanted `pyg:26.06-py3` for RAPIDS/wholegraph 26.06 (which
# has cugraph-gnn PR #451, "support embedding training with bf16 and fp16" — lifts the WholeMemory
# bf16/fp16 *table* crash, docs/phase4-full-cc-plan.md §5). But NGC never published that tag: the
# newest PyG container is `26.05-py3` (2026-06-01) = wholegraph **26.04.00** (probe job 3159390),
# pre-#451. NGC ships PyG ~every 2 months (26.01/26.03/26.05), so the next is likely `26.07-py3`
# (~late July) — bump to it (and confirm `pylibwholegraph.__version__ >= 26.06` in the inspection job)
# when it lands. Until then bf16 is unavailable and the table stays fp32 (§2/§5).

FROM nvcr.io/nvidia/pyg:26.05-py3

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

RUN pip install pyarrow tensorboard psutil nvidia-ml-py wandb \
    && python -c "import torch; print('cupyg base ok; torch', torch.__version__)"

# NB: do NOT import cugraph_pyg/pylibwholegraph at build (CUDA extensions dlopen the driver, present
# only at runtime on a GPU node) — verified separately in the inspection job.
CMD ["python", "-c", "import torch; print('cugraph-pyg image ready; torch', torch.__version__)"]
