"""RMM / WholeGraph / cugraph communicator bring-up and VRAM logging.

Required even at world size 1: the cugraph structures need the communicator to exist first.
"""

from __future__ import annotations

import torch


# One node type "node", one relation -> a homogeneous graph in cuGraph-PyG's hetero API.
EDGE_TYPE = ("node", "links", "node")

def init_worker(global_rank: int, local_rank: int, world_size: int, cugraph_id: object) -> None:
    """Initialise RMM + WholeGraph + cugraph comms for this rank (mirrors the cuGraph-PyG examples).

    Required even for world_size 1: ``GraphStore`` holds the graph in cugraph's distributed
    structure, so the communicator must exist before any edges are added.
    """
    import rmm

    rmm.reinitialize(devices=local_rank, managed_memory=True, pool_allocator=False)

    from pylibwholegraph.torch.initialize import init as wm_init

    wm_init(global_rank, world_size, local_rank, torch.cuda.device_count())

    import cupy
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.Device(local_rank).use()
    cupy.cuda.set_allocator(rmm_cupy_allocator)

    from pylibcugraph.comms import cugraph_comms_init

    cugraph_comms_init(rank=global_rank, world_size=world_size, uid=cugraph_id, device=local_rank)
    torch.cuda.set_device(local_rank)


# ── model ────────────────────────────────────────────────────────────────────────────────────



def _log_vram(tag: str, local_rank: int) -> None:
    """Per-rank VRAM snapshot for OOM/alloc debugging: allocated/reserved/peak + device free/total.

    Printed from every rank (each owns a device) so a rank imbalance is visible; cheap, flushed.
    """
    if not torch.cuda.is_available():
        return
    try:
        free, total = torch.cuda.mem_get_info()
        print(
            f"[vram] rank{local_rank} {tag}: "
            f"alloc={torch.cuda.memory_allocated() / 1e9:.1f} "
            f"reserved={torch.cuda.memory_reserved() / 1e9:.1f} "
            f"peak={torch.cuda.max_memory_allocated() / 1e9:.1f} "
            f"free={free / 1e9:.1f}/{total / 1e9:.1f} GB",
            flush=True,
        )
    except Exception as exc:  # debug logging must never kill the run
        print(f"[vram] rank{local_rank} {tag}: (mem_get_info failed: {exc})", flush=True)

