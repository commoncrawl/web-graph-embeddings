"""cuGraph-PyG featureless link-prediction trainer (Phase 3.1 prototype -> Phase 4 full-CC train).

Started as the single-GPU Option-B prototype and grew into the full-CC training script. The
GraphStorm/DistDGL backend tops out at ~7% GPU utilization because neighbour sampling runs on CPU
(see docs/experiments/2026-07-01-phase3.1-graphstorm-r0.md); cuGraph-PyG keeps sampling **on the
GPU** (``LinkNeighborLoader`` over a GPU-resident ``GraphStore``) and uses the *same* PyG API from
single-GPU up to WholeGraph-sharded multi-GPU (the path to the full ~279M-node CC graph).

The scale-up rungs are flag-gated, so the validated slice path is untouched when they are off:
``--wg-embed`` shards the embedding table across the WholeGraph communicator; ``--stream-split``
memory-bounds the 80/10/10 split + per-rank edge shard for the ~13B-edge cut (reading a memmapped
standalone ``edge_index.npy``, never the whole array); ``--eval-edges`` bounds the periodic MRR;
and ``--ckpt-dir`` / ``--ckpt-every-iters`` give resumable epoch- and mid-epoch checkpoints. Shallow
dot or GraphSAGE encoder; parity target vs 100 negatives = in-house dot 0.780. See
docs/phase4-full-cc-plan.md §4f for the full-CC launch recipe.

Featureless: link prediction learns a per-node embedding table (``nn.Embedding``); there are no
input node features. Homogeneous: one node type ``node`` and one edge relation ``[node, links,
node]`` (our directed COO). Leakage-safe: only *train* edges go into the ``GraphStore`` used for
sampling; val/test edges are held out and scored as ``edge_label_index``.

The stages live in the sibling ``cupyg/`` package (model, split, loaders, losses, eval,
checkpointing); this file is the entry point: argument parsing and ``main``. The alternative
model path -- ``cupyg/sage.py`` + ``cupyg/loader_sampled.py`` -- is reached only through
:func:`_sampled` and can be dropped wholesale (see ``docs/public-release.md``).

Runs in the ``cupyg`` image (NGC PyG base). Even single-GPU needs the cugraph/WholeGraph
communicator initialised, so launch under ``torchrun`` (world size 1 is fine):

    torchrun --standalone --nproc-per-node 1 /gs-scripts/cupyg_lp.py \
        --graph /data/cc_ndj_e005 --encoder shallow --epochs 10 --dim 128

Systems metrics (GPU util/VRAM/CPU) come from the ``gs_sysmon.py`` sidecar started by
``scripts/cupyg_lp.sh``; this script prints loss/MRR + per-epoch wall time and logs scalars to
TensorBoard (``--tb-logdir``). ``--wandb-project`` adds an optional, independent Weights & Biases
sink (rank 0; W&B's own monitor captures GPU/VRAM/CPU) — best-effort, so it never crashes the run.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cupyg.checkpoint import _any_rank_stop, _save_cupyg_ckpt, _SignalCatcher
from cupyg.evaluate import eval_mrr, eval_mrr_gathered, export_table_chunked
from cupyg.loader_free import _SamplerFreeEdgeLoader
from cupyg.model import LPModel, _base
from cupyg.runtime import _log_vram, init_worker
from cupyg.split import _subsample_eval, load_split, load_split_cache, stream_split_and_shard
from cupyg.train import train_epoch


def _sampled():  # noqa: ANN202 - returns the module object
    """Import the cuGraph neighbour-sampling path, or fail with a pointed message.

    ``--encoder sage`` and the non-``--sampler-free`` loader live in ``cupyg/loader_sampled.py``,
    which a minimal reproduction checkout does not ship: the published full-CC artifact is trained
    with ``--encoder shallow --sampler-free``.

    Returns:
        The :mod:`cupyg.loader_sampled` module.

    Raises:
        SystemExit: If the module was not shipped in this checkout.
    """
    try:
        from cupyg import loader_sampled
    except ImportError as exc:  # pragma: no cover - depends on which files the checkout ships
        msg = (
            "the cuGraph-sampling path (cupyg/loader_sampled.py) is not present in this checkout; "
            "the published artifact is trained with --encoder shallow --sampler-free"
        )
        raise SystemExit(msg) from exc
    return loader_sampled


def main() -> None:  # noqa: D103
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--encoder", choices=("shallow", "sage"), default="shallow")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--fanout", default="10,5", help="Comma sep neighbour fanout per hop.")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument(
        "--lr-schedule",
        choices=("none", "cosine"),
        default="none",
        help="cosine: CosineAnnealingLR over --epochs (eta_min=lr/100), to settle the early peak.",
    )
    ap.add_argument("--train-neg", type=int, default=1)
    ap.add_argument(
        "--neg-mode",
        choices=("binary", "inbatch"),
        default="binary",
        help="binary: 1 on-GPU loader negative; inbatch: --train-neg manual negatives each.",
    )
    ap.add_argument("--eval-neg", type=int, default=100)
    ap.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Eval every N epochs (SAGE full-node inference is costly); final epoch always evals.",
    )
    ap.add_argument(
        "--tb-logdir",
        default="",
        help="If set, log train/loss + val|test MRR/Hits scalars to this TensorBoard dir.",
    )
    ap.add_argument(
        "--wandb-project",
        default="",
        help="If set, also log config + train/val/test scalars to this W&B project (rank 0); "
        "W&B's system monitor captures GPU/VRAM/CPU natively. Off by default (TB + gs_sysmon stay "
        "independent). wandb is in the cupyg image; needs WANDB_API_KEY in the env.",
    )
    ap.add_argument("--wandb-run", default="", help="W&B run name (default: auto-generated).")
    ap.add_argument(
        "--wandb-mode",
        default="online",
        choices=("online", "offline", "disabled"),
        help="W&B mode; 'offline' logs locally to sync later when the node can't reach wandb.ai.",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stop after N evals without val_mrr improvement (0 = off). Best model always "
        "restored before the final test.",
    )
    ap.add_argument("--max-iter", type=int, default=0, help="Cap train iters/epoch (0 = full).")
    ap.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Intra-epoch loss print + TB/W&B scalar cadence in iters (rank 0). At full-CC epoch "
        "lengths this is the only sub-12h signal; 0 disables intra-epoch logging.",
    )
    ap.add_argument(
        "--ckpt-every-iters",
        type=int,
        default=0,
        help="Mid-epoch collective checkpoint every N iters (0 = only at eval boundaries). Bounds "
        "preemption loss to N iters instead of a whole ~12 h epoch (needs --ckpt-dir).",
    )
    ap.add_argument("--max-edges", type=int, default=0, help="Subsample first N edges (0 = all).")
    ap.add_argument(
        "--eval-edges",
        type=int,
        default=0,
        help="Cap val/test to a fixed random subsample of N edges for the periodic MRR (0 = all). "
        "Required at full-CC scale (val/test are ~1.3B edges each); validated slice runs use 0.",
    )
    ap.add_argument("--induced", action="store_true", help="Edge-induce the subsample.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--stream-split",
        action="store_true",
        help="Full-CC path (§3.1/§3.2): memory-bounded, permutation-free split streamed off the "
        "mmapped edge_index, with each rank ingesting only its own train shard (no rank holds all "
        "13.4B edges). Off = the validated in-memory load_split (permutes/materializes; fine for a "
        "slice). Assumes an already-compact graph (min-degree-slice output); ignores --induced.",
    )
    ap.add_argument(
        "--split-cache-dir",
        default="",
        help="Load a precomputed split (`wgl data stream-split-cache`) instead of "
        "streaming/hashing edges at startup: each rank memmap-slices its contiguous 1/world_size "
        "train shard (no re-read/re-hash, no thrash). Needs --stream-split; cuts full-CC startup "
        "from ~1h to a fast per-rank read each (re)launch. --eval-edges must be <= the cache's.",
    )
    ap.add_argument(
        "--wg-embed",
        action="store_true",
        help="Shard the node-embedding table across the WholeGraph communicator (multi-GPU path, "
        "own WholeMemory sparse Adam). Off = single-GPU nn.Embedding (the validated path).",
    )
    ap.add_argument(
        "--sampler-free",
        action="store_true",
        help="Shallow only: skip the cuGraph GraphStore + neighbour sampler; iterate seed-edge "
        "batches directly (gather by id from the table, in-batch negatives). Required at full-CC "
        "scale — the MG sampler 32-bit-overflows beyond ~2 B edges/GPU (13 B/4 ranks). Also drops "
        "the graph from VRAM (~63 GB/rank). Ignored for --encoder sage (needs the sampled graph).",
    )
    ap.add_argument(
        "--wg-location",
        choices=("cuda", "cpu"),
        default="cuda",
        help="WholeMemory shard location: cuda (NVLink, fits the slice) or cpu (host, full-CC).",
    )
    ap.add_argument(
        "--wg-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="WholeMemory table dtype (Phase-4 §5 precision experiment). bfloat16 halves table + "
        "Adam-state GPU memory (143->72 GB at full CC); gather upcasts to fp32 so the export stays "
        "fp32. Default float32 (the validated path).",
    )
    ap.add_argument(
        "--export-emb",
        default="",
        help="If set, save the final frozen node embeddings as [num_nodes, dim] float32 .npy here "
        "(+ a .json metadata sidecar) for downstream/repeatable eval. Rank 0 only.",
    )
    ap.add_argument(
        "--ckpt-dir",
        default="",
        help="Stable dir for a RESUMABLE checkpoint (survives resubmit/requeue): saved every eval "
        "and on SIGTERM/SIGUSR1, so a job killed by the wall limit or a transient node error "
        "continues instead of restarting. Covers dense (SAGE) + the sharded --wg-embed table. "
        "Empty = off. For preemption-safe saves also set sbatch --signal=USR1@180.",
    )
    args = ap.parse_args()

    global_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # Bind this rank to its own GPU *before* NCCL init — otherwise every rank default-binds to
    # cuda:0 and rank>0's first op on its real device fails with cudaErrorDevicesUnavailable
    # (world_size 1 never hit this). Must precede init_process_group + the cugraph-id broadcast.
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")

    from pylibcugraph.comms import cugraph_comms_create_unique_id

    cugraph_id = [cugraph_comms_create_unique_id() if global_rank == 0 else None]
    torch.distributed.broadcast_object_list(cugraph_id, src=0, device=torch.device(local_rank))
    init_worker(global_rank, local_rank, world_size, cugraph_id[0])

    is_main = global_rank == 0
    fanout = [int(x) for x in args.fanout.split(",")]
    # Sampler-free is shallow-only (sage needs the sampled subgraph): skip the GraphStore + sampler.
    sampler_free = args.sampler_free and args.encoder == "shallow"
    if args.stream_split:
        # Full-CC: each rank ingests only its own train shard as its graph partition
        # (build_stores partitioned=True); the shard is both the partition and the label edges.
        if args.split_cache_dir:
            # Precomputed split (fast, re-hash-free) -- memmap-slice this rank's contiguous shard.
            num_nodes, train_e_local, val_e, test_e = load_split_cache(
                Path(args.split_cache_dir), world_size, global_rank, eval_cap=args.eval_edges
            )
        else:
            # Stream + hash the mmapped edges in-line (no rank holds all 13.4B edges).
            num_nodes, train_e_local, val_e, test_e = stream_split_and_shard(
                Path(args.graph),
                args.seed,
                world_size,
                global_rank,
                args.max_edges,
                eval_cap=args.eval_edges,
            )
        # Sampler-free skips the 13 B-edge GraphStore entirely (frees ~63 GB/rank; no sampler).
        stores = (
            None
            if sampler_free
            else _sampled().build_stores(num_nodes, train_e_local, global_rank, world_size, partitioned=True)
        )
    else:
        num_nodes, train_e, val_e, test_e = load_split(
            Path(args.graph), args.seed, args.max_edges, args.induced
        )
        stores = None if sampler_free else _sampled().build_stores(num_nodes, train_e, global_rank, world_size)
        # Each rank trains on a disjoint stripe of the label edges (all ranks sample from the same
        # distributed graph); single-GPU keeps the full edge list.
        train_e_local = train_e[:, global_rank::world_size] if world_size > 1 else train_e

    _log_vram("after build_stores (graph ingested to GPU)", local_rank)

    # Bound the periodic MRR to a fixed subsample (all ranks pick the same seeded set, so eval stays
    # replicated + comparable across epochs). Off by default -> validated slice evals every edge.
    if args.eval_edges > 0:
        val_e = _subsample_eval(val_e, args.eval_edges, args.seed)
        test_e = _subsample_eval(test_e, args.eval_edges, args.seed + 1)
        if is_main:
            print(f"[eval-edges] val={val_e.shape[1]:,} test={test_e.shape[1]:,}", flush=True)

    # Sharded (WholeGraph) node-embedding table: created across the communicator, trained by its own
    # WholeMemory sparse Adam. Single-GPU (default) keeps the plain nn.Embedding inside LPModel.
    wg_emb = emb_module = wm_opt = None
    if args.wg_embed:
        import pylibwholegraph.torch as wgth

        comm = wgth.get_global_communicator()
        wg_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[args.wg_dtype]
        _log_vram(f"before create_embedding (wg_location={args.wg_location})", local_rank)
        wg_emb = wgth.create_embedding(
            comm,
            "distributed",
            args.wg_location,
            wg_dtype,
            [num_nodes, args.dim],
            cache_policy=None,
            random_init=True,
        )
        _log_vram("after create_embedding (table allocated)", local_rank)
        emb_module = wgth.embedding.WholeMemoryEmbeddingModule(wg_emb)
        wm_opt = wgth.create_wholememory_optimizer([wg_emb], "adam", {})
        _log_vram("after create_wholememory_optimizer (Adam states)", local_rank)

    model = LPModel(num_nodes, args.dim, args.encoder, args.num_layers, emb_module).cuda()
    _log_vram("after model.cuda()", local_rank)
    # Dense Adam covers only torch params (the sage convs); the sharded shallow model has none, so
    # its optimizer is None and all learning flows through the WholeMemory sparse optimizer.
    dense_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(dense_params, lr=args.lr) if dense_params else None
    if world_size > 1 and dense_params:
        from torch.nn.parallel import DistributedDataParallel

        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    sched = None
    if args.lr_schedule == "cosine" and opt is not None:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr / 100
        )

    def cur_lr(epoch: int) -> float:
        """LR for the WholeMemory optimizer this epoch (mirrors the cosine schedule if enabled)."""
        if args.lr_schedule != "cosine":
            return args.lr
        eta = args.lr / 100
        return eta + 0.5 * (args.lr - eta) * (1 + math.cos(math.pi * (epoch - 1) / args.epochs))

    # inbatch corrupts negatives manually (--train-neg of them); the loader keeps its 1 on-GPU
    # negative (discarded) since amount>1 crashes the cugraph sampler.
    loader_neg = 1 if args.neg_mode == "inbatch" else args.train_neg
    if sampler_free:
        train_loader = _SamplerFreeEdgeLoader(train_e_local, args.batch_size, True, seed=args.seed)
    else:
        train_loader = _sampled().make_loader(stores, train_e_local, fanout, loader_neg, args.batch_size, True)
    _log_vram(f"after make_loader (sampler_free={sampler_free} bs={args.batch_size})", local_rank)
    eval_gen = torch.Generator(device="cuda").manual_seed(args.seed)

    n_dense = sum(p.numel() for p in dense_params)
    if is_main:
        wg = f"wg-embed[{args.wg_location}] " if args.wg_embed else ""
        print(
            f"{wg}world_size={world_size} encoder={args.encoder} dim={args.dim} fanout={fanout} "
            f"dense_params={n_dense:,} neg_mode={args.neg_mode} train_neg={args.train_neg}",
            flush=True,
        )

    # Optional TensorBoard scalar logging (rank 0 only), alongside the gs_sysmon system panel.
    writer = None
    if args.tb_logdir and is_main:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(args.tb_logdir)
        print(f"tensorboard scalars -> {args.tb_logdir}", flush=True)

    # Optional Weights & Biases (rank 0), independent of TB/gs_sysmon. A logging backend must never
    # crash a multi-hour run, so import + init are best-effort: on failure we warn and fall back to
    # TB + gs_sysmon. W&B's own system monitor captures GPU/VRAM/CPU natively.
    run = None
    if args.wandb_project and is_main:
        try:
            import wandb

            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run or None,
                id=args.wandb_run or None,  # stable id -> resubmits resume ONE run, not fragments
                resume="allow",
                mode=args.wandb_mode,
                config=vars(args),
            )
            print(f"wandb: project={args.wandb_project} run={run.name}", flush=True)
        except Exception as exc:  # logging must never kill training
            print(f"wandb: disabled ({type(exc).__name__}: {exc})", flush=True)
            run = None

    # Keep-best on BOTH paths: the dense path snapshots the torch state_dict (best_state); the
    # sharded --wg-embed table lives outside it, so it is saved to wm_best/ on val improvement and
    # reloaded before the final test/export (see the eval block below). Eval is identical across
    # ranks (same gathered table + same eval_gen seed), so patience/early-stop stays in lockstep
    # without extra comms; only the preemption stop-flag is all-reduced (see _any_rank_stop).
    best_val = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0

    # Resume from a checkpoint if one exists (survives resubmit/requeue). WholeMemory table restored
    # collectively; the rest (convs, optimizer, sched, epoch, best-bookkeeping, RNG) from meta.pt.
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None
    start_epoch = 1
    if ckpt_dir is not None and (ckpt_dir / "meta.pt").exists():
        if wg_emb is not None:
            wg_emb.load(str(ckpt_dir / "wm_emb"))
        ck = torch.load(ckpt_dir / "meta.pt", map_location="cuda", weights_only=False)
        with contextlib.suppress(Exception):
            _base(model).load_state_dict(ck["model"])
        if opt is not None and ck.get("opt"):
            opt.load_state_dict(ck["opt"])
        if sched is not None and ck.get("sched"):
            sched.load_state_dict(ck["sched"])
        best_val, best_epoch, no_improve = ck["best_val"], ck["best_epoch"], ck["no_improve"]
        best_state = ck.get("best_state")
        if ck.get("torch_rng") is not None:
            torch.set_rng_state(ck["torch_rng"].to("cpu", torch.uint8))
        if ck.get("cuda_rng") is not None:
            with contextlib.suppress(RuntimeError):
                torch.cuda.set_rng_state_all([t.to("cpu", torch.uint8) for t in ck["cuda_rng"]])
        if ck.get("gen_rng") is not None:
            eval_gen.set_state(ck["gen_rng"].to("cpu", torch.uint8))
        start_epoch = int(ck["epoch"]) + 1
        if is_main:
            print(
                f"[resume] from checkpoint at epoch {ck['epoch']} -> continuing at {start_epoch} "
                f"(best ep{best_epoch}={best_val:.4f})",
                flush=True,
            )

    # Cap every rank to the SAME iteration count. Train shards differ by up to 1 edge and, with
    # drop_last, the batch COUNT can differ across ranks when n % batch_size straddles the boundary;
    # a per-batch collective (WM/cuGraph op, or the mid-epoch save below) would then hang at
    # end-of-epoch on the ranks with more batches. Reuse --max-iter as the shared cap.
    # Batch count: sampler-free = plain n//bs drop_last; the sampler path matches loader chunking.
    local_batches = (
        train_e_local.shape[1] // args.batch_size
        if sampler_free
        else _sampled()._num_batches(train_e_local.shape[1], args.batch_size)
    )
    epoch_cap = local_batches
    if world_size > 1 and torch.distributed.is_initialized():
        t = torch.tensor([local_batches], device="cuda")
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
        epoch_cap = int(t.item())
    if args.max_iter:
        epoch_cap = min(epoch_cap, args.max_iter)
    if is_main:
        print(f"[epoch-cap] iters/epoch={epoch_cap:,} (min across {world_size} ranks)", flush=True)
    _log_vram("before training loop (steady-state setup)", local_rank)

    wm_best_dir = (ckpt_dir / "wm_best") if ckpt_dir is not None else None
    _sig = _SignalCatcher().install()
    # A monotone global iteration counter drives all dashboard steps (a 1-3 epoch full-CC run would
    # have only 1-3 points on an epoch x-axis); continue past the resumed epoch to stay monotone.
    global_step = (start_epoch - 1) * epoch_cap

    def _mid_save(ep: int) -> None:  # collective mid-epoch ckpt; store ep-1 so resume redoes ep
        _save_cupyg_ckpt(
            ckpt_dir,
            epoch=ep - 1,
            best_val=best_val,
            best_epoch=best_epoch,
            no_improve=no_improve,
            best_state=best_state,
            base_model=_base(model),
            opt=opt,
            sched=sched,
            wg_emb=wg_emb,
            gen=eval_gen,
            is_main=is_main,
        )

    def _stop() -> bool:
        return _any_rank_stop(_sig, world_size)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        loss, iters_run, stopped = train_epoch(
            model,
            train_loader,
            opt,
            wm_opt,
            cur_lr(epoch),
            epoch_cap,
            args.neg_mode,
            args.train_neg,
            epoch=epoch,
            global_step=global_step,
            is_main=is_main,
            log_every=args.log_every,
            writer=writer,
            run=run,
            ckpt_every=args.ckpt_every_iters,
            save_cb=(_mid_save if ckpt_dir is not None else None),
            stop_cb=_stop,
        )
        global_step += iters_run
        if stopped:
            if is_main:
                print("[ckpt] mid-epoch stop — ckpt written, exiting for --requeue.", flush=True)
            sys.exit(0)
        if writer:
            writer.add_scalar("train/loss", loss, global_step)
            writer.add_scalar("train/lr", cur_lr(epoch), global_step)
        if run is not None:
            run.log(
                {"train/loss": loss, "train/lr": cur_lr(epoch), "epoch": epoch}, step=global_step
            )
        if sched is not None:
            sched.step()
        if epoch % args.eval_every and epoch != args.epochs:
            if is_main:
                dt = time.time() - t0
                print(
                    f"epoch {epoch}/{args.epochs} loss={loss:.4f} (no-eval, {dt:.0f}s)", flush=True
                )
            continue
        # Shallow gathers only the rows each eval chunk needs (never the ~27 GB full-CC table);
        # sage's message-passed reps require the full-node pass. Both are collective across ranks.
        if args.encoder == "shallow":
            val = eval_mrr_gathered(model, val_e, num_nodes, args.eval_neg, (1, 10), eval_gen)
        else:
            emb = _sampled().node_embeddings(model, stores, num_nodes, fanout, args.dim)
            val = eval_mrr(emb, val_e, args.eval_neg, (1, 10), eval_gen)
        if writer:
            for key, value in val.items():
                writer.add_scalar(f"val/{key}", value, global_step)
        if run is not None:
            run.log({f"val/{key}": value for key, value in val.items()}, step=global_step)
        if is_main:
            print(
                f"epoch {epoch}/{args.epochs} loss={loss:.4f} val_mrr={val['mrr']:.4f} "
                f"val_hits@10={val['hits@10']:.4f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
        # Best-checkpoint tracking: these runs peak early then overtrain, so snapshot the best model
        # and restore it before the final test/export, not the last (overtrained) epoch. Dense path
        # snapshots the state_dict; the sharded --wg-embed table lives outside it, so we save the WM
        # table to wm_best/ on improvement (collective; val is identical on all ranks) and reload it
        # before export -- otherwise the export would ship whatever the table holds at loop exit.
        if val["mrr"] > best_val:
            best_val, best_epoch, no_improve = val["mrr"], epoch, 0
            if not args.wg_embed:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elif wg_emb is not None and wm_best_dir is not None:
                wm_best_dir.mkdir(parents=True, exist_ok=True)
                wg_emb.save(str(wm_best_dir / "emb"))  # collective across the communicator
                if is_main:
                    print(f"[best] saved wm_best (ep {epoch}, val_mrr={best_val:.4f})", flush=True)
        else:
            no_improve += 1
            if args.patience and no_improve >= args.patience:
                if is_main:
                    print(
                        f"early stop at epoch {epoch}: no val_mrr gain in {no_improve} evals "
                        f"(best ep{best_epoch}={best_val:.4f})",
                        flush=True,
                    )
                break
        # Resumable checkpoint at each eval boundary (+ exit cleanly if a stop signal arrived).
        if ckpt_dir is not None:
            _save_cupyg_ckpt(
                ckpt_dir,
                epoch=epoch,
                best_val=best_val,
                best_epoch=best_epoch,
                no_improve=no_improve,
                best_state=best_state,
                base_model=_base(model),
                opt=opt,
                sched=sched,
                wg_emb=wg_emb,
                gen=eval_gen,
                is_main=is_main,
            )
        if _any_rank_stop(_sig, world_size):  # all-reduce so ranks exit together (no hang)
            if is_main:
                print(
                    "[ckpt] stop signal — checkpoint written, exiting cleanly for --requeue.",
                    flush=True,
                )
            sys.exit(0)

    if best_state is not None:  # dense path: restore the best state_dict
        _base(model).load_state_dict({k: v.cuda() for k, v in best_state.items()})
        if is_main:
            print(
                f"restored best model from epoch {best_epoch} (val_mrr={best_val:.4f})", flush=True
            )
    elif (
        args.wg_embed
        and wg_emb is not None
        and wm_best_dir is not None
        and wm_best_dir.exists()
        and any(wm_best_dir.iterdir())
    ):
        wg_emb.load(str(wm_best_dir / "emb"))  # sharded path: reload the best table before export
        if is_main:
            print(f"restored best wm table (ep {best_epoch}, val_mrr={best_val:.4f})", flush=True)
    if args.encoder == "shallow":  # gather-per-chunk eval (no full-table materialization)
        test = eval_mrr_gathered(model, test_e, num_nodes, args.eval_neg, (1, 10), eval_gen)
    else:
        emb = _sampled().node_embeddings(model, stores, num_nodes, fanout, args.dim)
        test = eval_mrr(emb, test_e, args.eval_neg, (1, 10), eval_gen)
    if writer:
        for key, value in test.items():
            writer.add_scalar(f"test/{key}", value, global_step)
        writer.close()
    if run is not None:
        run.log({f"test/{key}": value for key, value in test.items()}, step=global_step)
        run.summary["best_val_mrr"] = best_val
        run.summary["best_epoch"] = best_epoch
        run.finish()
    if is_main:
        print(
            f"[result] encoder={args.encoder} wg_embed={args.wg_embed} world_size={world_size} "
            f"test_mrr={test['mrr']:.4f} test_hits@1={test['hits@1']:.4f} "
            f"test_hits@10={test['hits@10']:.4f} "
            f"(best_epoch={best_epoch} best_val_mrr={best_val:.4f}; in-house dot reference=0.780)",
            flush=True,
        )
    # Persist the frozen [num_nodes, dim] embeddings (row i = graph node i, aligned to the same
    # node ids DirectedGraph/labels use) + a self-describing sidecar, so the downstream eval suite
    # (language/spam/topic) can run on THESE vectors now and be re-run/repeated later without
    # retraining. Shallow gathers the table in chunks (COLLECTIVE — all ranks in lockstep, rank 0
    # keeps it) to bound the full-CC GPU peak; sage's rank-0 `emb` is already the whole table.
    if args.export_emb:
        if args.encoder == "shallow":
            emb_np = export_table_chunked(model, num_nodes, args.dim, keep=is_main)
        else:
            emb_np = emb.detach().cpu().numpy().astype(np.float32) if is_main else None
    if args.export_emb and is_main:
        out = Path(args.export_emb)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, emb_np)
        meta = {
            "graph": args.graph,
            "num_nodes": num_nodes,
            "dim": args.dim,
            "encoder": args.encoder,
            "wg_embed": args.wg_embed,
            "wg_dtype": args.wg_dtype,
            "world_size": world_size,
            "epochs": args.epochs,
            "neg_mode": args.neg_mode,
            "train_neg": args.train_neg,
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_val_mrr": best_val,
            "test_mrr": test["mrr"],
            "test_hits@1": test["hits@1"],
            "test_hits@10": test["hits@10"],
        }
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"[export] embeddings {emb_np.shape} -> {out} (+ .json sidecar)", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
