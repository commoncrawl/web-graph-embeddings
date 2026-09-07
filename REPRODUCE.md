# Reproducing the embeddings

Command-by-command path from raw Common Crawl host-graph shards to the published Parquet shard set.
Six stages; the middle four are CPU-only, stage 5 needs 4 GPUs.

> **Resource warning.** Stage 5 as published used **4×H100 (80 GB), ~900 GB host RAM, ~1 h 05 m**
> wall. Stages 2–4 need ~1 h 15 m, ~7 min and ~45 min of CPU respectively, and ~160 GB of scratch.
> To validate the pipeline without that allocation, raise `--min-degree` (16, 32, …) or point stage 1
> at a single shard — every stage below is threshold-agnostic.

## Containers

Two images, both built from this repo:

```bash
docker build -f docker/cpu/Dockerfile          -t wge-cpu:latest .   # stages 2-4, 6
docker build -f docker/cugraph-pyg/Dockerfile  -t wge-gpu:latest .   # stage 5
```

The GPU image is NVIDIA's NGC PyG container (`nvcr.io/nvidia/pyg:26.05-py3`) plus a thin layer; it
ships cuGraph-PyG + WholeGraph and **needs a CUDA-13-capable driver**. It deliberately does *not*
install this project — the trainer is a standalone script tree mounted into it.

Conventions used below (adapt to your own paths):

```bash
export CCRAW=/data/commoncrawl/cc-main-2025-26-nov-dec-jan/host  # stage 1 output
export WORK=/data/wge                                            # working data
export RUNS=/data/wge-runs                                       # checkpoints + exports
```

Each stage shows a plain `docker run`; on a Slurm cluster with pyxis/enroot the equivalent is
`srun --container-image=… --container-mounts=…` with the same mounts and command.

---

## 1. Download the raw host graph

Fetches the Common Crawl **host-level** webgraph release into `<out>/vertices/` + `<out>/edges/`.
The published run used `cc-main-2025-26-nov-dec-jan` — 279,356,058 hosts / ~13.4 B directed edges,
roughly 105 GB of gzipped TSV.

```bash
scripts/download_cc_hostgraph.sh cc-main-2025-26-nov-dec-jan "$CCRAW"
```

No container needed (curl + gzip). Releases are listed at <https://commoncrawl.org/web-graphs>.

## 2. Choose the degree cut, then build the training slice

`degree-distribution` streams the raw shards once and reports node/edge retention per threshold. Run
it before assuming 8 is still the right cut for a newer crawl:

```bash
docker run --rm -v "$CCRAW":/ccraw:ro -v "$WORK":/data wge-cpu \
  wgl data degree-distribution --edges /ccraw/edges --num-nodes 279356058 \
    --out /data/degree_distribution.json --edge-coverage
```

Then stream the shards into a compact induced subgraph of hosts with total degree ≥ 8. This keeps
**52,913,544 hosts (19%) and 13.07 B edges**; the degree-1 sea is dropped. `--streaming` bounds RAM
via a disk memmap — the induced edge list does not fit host memory. Writes `graph.npz`,
`edge_index.npy`, `node_names.txt` and `orig_ids.npy` (compact id → global CC id). ~1 h 15 m.

```bash
docker run --rm -v "$CCRAW":/ccraw:ro -v "$WORK":/data wge-cpu \
  wgl data min-degree-slice --streaming --min-degree 8 --num-nodes 279356058 \
    --vertices /ccraw/vertices --edges /ccraw/edges --out /data/cc_deg8
```

## 3. Extract a memmap-able `edge_index.npy`

Only needed for slices written before the writer emitted it (stage 2 now does). `np.load` ignores
`mmap_mode` for an `.npz` member, which would materialize ~105 GB per rank. ~7 min.

```bash
docker run --rm -v "$WORK":/data wge-cpu \
  wgl data extract-edge-index --graph /data/cc_deg8
```

## 4. Precompute the split cache

Computes the deterministic 80/10/10 edge split **once** into `split_cache/`, so each training rank
memmap-slices its contiguous shard at launch instead of re-hashing the whole edge list on every
(re)start. `--eval-edges` subsamples val/test; stage 5's `--eval-edges` must be ≤ this. ~45 min.

```bash
docker run --rm -v "$WORK":/data wge-cpu \
  wgl data stream-split-cache --graph /data/cc_deg8 --eval-edges 2000000
```

## 5. Train, evaluate and export (4 GPUs)

One job does all three. `--sampler-free` drops the graph from VRAM so the table and its Adam state
shard to ~24 GB/rank, and `--wg-location cuda` gathers over NVLink (~9.7 M edges/s, ~17 min/epoch).
`--export-emb` writes the frozen `[52,913,544, 128]` fp32 table plus a JSON sidecar.

```bash
docker run --rm --gpus all --ipc=host \
  -v "$WORK":/data -v "$RUNS":/runs -v "$PWD/scripts":/gs-scripts \
  wge-gpu bash /gs-scripts/cupyg_lp.sh /runs/cupyg cc_deg8_shallow_full \
    --graph /data/cc_deg8 --split-cache-dir /data/cc_deg8/split_cache \
    --encoder shallow --dim 128 --epochs 3 \
    --stream-split --wg-embed --wg-location cuda --sampler-free \
    --neg-mode inbatch --train-neg 50 --batch-size 32768 \
    --lr 0.003 --lr-schedule cosine --patience 5 --eval-edges 1000000 \
    --ckpt-every-iters 20000 --ckpt-dir /runs/cupyg/ckpt/cc_deg8_shallow_full \
    --export-emb /runs/cupyg/export/cc_deg8_shallow_full_emb.npy
```

`cupyg_lp.sh` wraps `torchrun` (one process per visible GPU), starts the `gs_sysmon.py` metrics
sidecar, and exports `WG_USE_POSIX_SHM=1` — that last one is **load-bearing**: WholeMemory's default
SysV shared-memory path collides across jobs via a hardcoded `ftok` project id and reproducibly
kills any run with ≥ 4 ranks. Set it if you invoke `torchrun` yourself.

**If the job is preempted or times out, resubmit the identical command.** `--ckpt-dir` restores the
table, optimizer and epoch (at most `--ckpt-every-iters` iterations replayed).

Expected outcome: `test_mrr ≈ 0.957`, and

```
/runs/cupyg/export/cc_deg8_shallow_full_emb.npy    # [52,913,544, 128] fp32, 27.09 GB
/runs/cupyg/export/cc_deg8_shallow_full_emb.json   # config + metrics sidecar
```

Row *i* aligns to line *i* of `cc_deg8/node_names.txt` (a reversed host name, e.g.
`com.example.www`). `orig_ids.npy` maps back to the global Common Crawl host id.

## 6. Shard for the hub

Turns the table into the published Parquet shard set plus a checksummed manifest: an fp32 tier
(canonical) and an fp16 tier, vectors L2-normalized, keyed by stable host and PLD keys.

```bash
docker run --rm -v "$WORK":/data -v "$RUNS":/runs wge-cpu \
  wgl release shard-export \
    --emb /runs/cupyg/export/cc_deg8_shallow_full_emb.npy \
    --node-names /data/cc_deg8/node_names.txt \
    --out /runs/release/cc_deg8 \
    --release-id cc-main-2025-26-nov-dec-jan \
    --source-crawls cc-main-2025-26-nov-dec-jan \
    --model-space-version v1 \
    --meta /runs/cupyg/export/cc_deg8_shallow_full_emb.json
```

Add `--limit 100000` first for a smoke test. `wgl release manifest` emits the separate host-identity
manifest (`manifest.json` + `hosts.tsv`) if you need the key/evidence table on its own, and
`wgl release sizing` prints artifact size estimates across dimensions and precisions.
