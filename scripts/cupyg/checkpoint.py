"""Resumable epoch- and mid-epoch checkpointing (ported from ``wgl.train.checkpoint``).

The cupyg image has no ``wgl``, so this is a deliberate port rather than an import. Restores the
table, the optimizer and the epoch so an identical resubmit continues a preempted 24h run.
"""

from __future__ import annotations

import contextlib
import signal
from pathlib import Path
from types import FrameType

import torch

from cupyg.model import LPModel


class _SignalCatcher:
    """Trap SIGTERM/SIGUSR1 and expose ``.stop`` so the loop checkpoints + exits for --requeue."""

    def __init__(self) -> None:
        self.stop = False
        self._prev: dict = {}

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        self.stop = True
        print(f"[ckpt] caught signal {signum}; will checkpoint + stop at next bound.", flush=True)

    def install(self) -> _SignalCatcher:
        """Install the stop-flag handlers (no-op off the main thread); periodic save is the net."""
        for s in (signal.SIGUSR1, signal.SIGTERM):
            with contextlib.suppress(ValueError, OSError):
                signal.signal(s, self._handle)
        return self



def _any_rank_stop(sig: _SignalCatcher | None, world_size: int) -> bool:
    """True if ANY rank caught a stop signal (all-reduce MAX over the flag).

    Collective saves/exits must be unanimous: if one rank exits on its local flag while the others
    proceed to the next collective (WM save / gather), they deadlock. All-reducing the flag first
    keeps every rank on the same branch.
    """
    local = 1 if (sig is not None and sig.stop) else 0
    if world_size > 1 and torch.distributed.is_initialized():
        t = torch.tensor([local], device="cuda")
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        return bool(t.item())
    return bool(local)



def _save_cupyg_ckpt(
    ckpt_dir: Path,
    *,
    epoch: int,
    best_val: float,
    best_epoch: int,
    no_improve: int,
    best_state: dict | None,
    base_model: LPModel,
    opt: object,
    sched: object,
    wg_emb: object,
    gen: object,
    is_main: bool,
) -> None:
    """Atomically checkpoint training state. WholeMemory table saved collectively by all ranks."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if wg_emb is not None:  # collective across the communicator (every rank writes its shard)
        wg_emb.save(str(ckpt_dir / "wm_emb"))
    if not is_main:
        return
    meta = {
        "epoch": int(epoch),
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "no_improve": int(no_improve),
        "best_state": best_state,  # dense best-restore snapshot (None on the wg path)
        "model": base_model.state_dict(),  # sage convs (+ nn.Embedding table on the dense path)
        "opt": None if opt is None else opt.state_dict(),
        "sched": None if sched is None else sched.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "gen_rng": gen.get_state() if gen is not None else None,
        "wg": wg_emb is not None,
    }
    tmp = ckpt_dir / "meta.pt.tmp"
    torch.save(meta, tmp)
    tmp.replace(ckpt_dir / "meta.pt")
    print(f"[ckpt] wrote resumable checkpoint (epoch {epoch}) -> {ckpt_dir}/meta.pt", flush=True)

