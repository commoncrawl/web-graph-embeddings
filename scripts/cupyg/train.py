"""The per-epoch training loop.
"""

from __future__ import annotations

import torch

from cupyg.losses import _binary_loss, _inbatch_loss
from cupyg.runtime import _log_vram


def train_epoch(
    model: object,
    loader: object,
    opt: torch.optim.Optimizer | None,
    wm_opt: object,
    lr: float,
    max_iter: int,
    neg_mode: str,
    train_neg: int,
    *,
    epoch: int = 0,
    global_step: int = 0,
    is_main: bool = True,
    log_every: int = 50,
    writer: object = None,
    run: object = None,
    ckpt_every: int = 0,
    save_cb: object = None,
    stop_cb: object = None,
) -> tuple[float, int, bool]:
    """Run one epoch of binary-NLL LP training; return ``(mean_loss, iters_run, stopped)``.

    ``binary`` uses the loader's single on-GPU negative per positive; ``inbatch`` discards it and
    scores ``train_neg`` manual negatives per positive (see :func:`_inbatch_loss`). ``opt`` is the
    dense Adam over torch params (``None`` for the sharded shallow model, which has none);
    ``wm_opt`` is the WholeMemory sparse optimizer for the sharded table, stepped with ``lr``.

    Loss accumulates **on-device** (one ``.item()`` per ``log_every`` window, not per iter -> no
    per-step device sync). Intra-epoch loss is logged (rank 0) at ``global_step + i`` -- a monotone
    global iteration counter, so a 1-3 epoch full-CC run still gets a dense dashboard curve, and the
    kill-criterion is visible within the first hour instead of after a ~12 h epoch. With
    ``ckpt_every > 0`` a collective mid-epoch checkpoint (``save_cb(epoch)``) runs every
    ``ckpt_every`` iters and ``stop_cb()`` (an all-ranks-agree preemption check) can end the epoch
    early -> at most ``ckpt_every`` iters lost on preemption. These boundaries are collective, so
    the caller must run all ranks for the same capped iteration count (see the ``epoch_cap``).
    """
    model.train()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    loss_sum = torch.zeros((), device=dev)  # on-device accumulator; synced only at log boundaries
    n_sum = 0
    done = 0
    stopped = False
    for i, raw in enumerate(loader):
        if max_iter and i >= max_iter:
            break
        batch = raw.cuda()
        if opt is not None:
            opt.zero_grad()
        x = model(batch)
        if neg_mode == "inbatch":
            loss, n = _inbatch_loss(x, batch, train_neg)
        else:
            loss, n = _binary_loss(x, batch)
        loss.backward()
        if opt is not None:
            opt.step()
        if wm_opt is not None:
            wm_opt.step(lr)
        loss_sum += loss.detach() * n
        n_sum += n
        done += 1
        if i == 0:
            _log_vram("train iter 0 (first batch+MFG+backward on GPU)", torch.cuda.current_device())
        if log_every and i % log_every == 0 and is_main:
            lv = loss.item()  # one device sync per window, not per iteration
            # Compact VRAM readout to watch for growth/leaks over the epoch (rank 0's device).
            _free, _tot = torch.cuda.mem_get_info()
            _mem = (
                f"reserved={torch.cuda.memory_reserved() / 1e9:.1f} free={_free / 1e9:.1f}/"
                f"{_tot / 1e9:.0f}GB"
            )
            print(f"  ep{epoch} iter {i} loss {lv:.4f} {_mem}", flush=True)
            step = global_step + i
            if writer is not None:
                writer.add_scalar("train/loss_iter", lv, step)
            if run is not None:
                run.log({"train/loss_iter": lv, "epoch": epoch}, step=step)
        # Mid-epoch collective checkpoint + cooperative stop. All ranks reach these boundaries in
        # lockstep (the loop is capped to a shared iter count), so the collective save can't hang.
        if ckpt_every and i and i % ckpt_every == 0:
            end = bool(stop_cb()) if stop_cb is not None else False
            if save_cb is not None:
                save_cb(epoch)
            if end:
                stopped = True
                break
    mean = (loss_sum / max(n_sum, 1)).item()
    return mean, done, stopped


# --- Resumable checkpointing (ported from wgl.train.checkpoint; wgl is not in the cupyg image) ----
# Covers BOTH training paths: the dense nn.Embedding path (SAGE / non-wg shallow) via torch
# state_dicts, and the WholeGraph-sharded --wg-embed path via the WholeMemory table's own collective
# save/load (`WholeMemoryEmbedding.save/load(prefix)`). Saved at every eval boundary, optionally
# mid-epoch (`--ckpt-every-iters`, so a ~12h full-CC epoch isn't lost on preemption), and on a
# SIGTERM/USR1 stop signal (acted on at the next such boundary).

