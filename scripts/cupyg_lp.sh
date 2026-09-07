#!/bin/bash
# Launch the cuGraph-PyG LP trainer (scripts/cupyg_lp.py) under torchrun with the gs_sysmon.py
# system-metrics sidecar (GPU util/VRAM/CPU -> TensorBoard + a stdout summary, readable straight from
# the job log). Auto-scales torchrun to the visible GPU count for the WholeGraph-sharded multi-GPU /
# full-CC run (sbatch --gpus-per-node=N + the trainer's --wg-embed). Everything after <out_dir>
# <tag> is forwarded to cupyg_lp.py (encoder, --stream-split, --wg-embed, --wandb-project, ...); this
# wrapper only injects --tb-logdir and runs the sidecar. W&B is opt-in via --wandb-project (its own
# system monitor also captures GPU/VRAM/CPU); TB + gs_sysmon stay independent.
#
#   slurm_submit.sh -C <proj> slurm/cupyg_gpu.sbatch -- bash /gs-scripts/cupyg_lp.sh \
#       /runs/cupyg cc_deg8_shallow --graph /data/cc_deg8 --encoder shallow --wg-embed --stream-split
set -uo pipefail

# Resolve sibling scripts relative to this file, not a hardcoded mount point: the cluster mounts
# scripts/ at /gs-scripts, but a plain `docker run`/`torchrun` elsewhere may not.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUT="${1:?usage: cupyg_lp.sh <out_dir> <tag> [cupyg_lp.py args...]}"
TAG="${2:?need a run tag}"
shift 2

TB="$OUT/tb/$TAG"
mkdir -p "$TB"

echo "=== [$(date +%T)] cupyg $TAG | args: $* ==="
python "$HERE/gs_sysmon.py" --logdir "$TB/system" --interval 3 &
SYS=$!
# Reap the sidecar on any exit path (error, or the wrapper itself being signaled), belt-and-braces
# with the explicit kill below; srun's cgroup cleanup is the final backstop.
trap 'kill -TERM "$SYS" 2>/dev/null' EXIT

# Even single-GPU cuGraph-PyG needs the cugraph/WholeGraph communicator -> launch under torchrun.
# NPROC = one process per GPU (the multi-GPU WholeGraph path: sbatch --gpus-per-node=N + the
# trainer's --wg-embed). Default to the count of GPUs visible in the container so it auto-scales with
# the allocation (no env-propagation dance); override with NPROC=... if needed. Default the
# training-scalar TB dir to the same tag dir as the system panel; a caller-supplied --tb-logdir in
# "$@" takes precedence (argparse keeps the last value).
NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"
NPROC="${NPROC:-1}"
[[ "$NPROC" -ge 1 ]] 2>/dev/null || NPROC=1

# WholeMemory host shared-memory: use POSIX shm (/dev/shm), NOT the buggy default SysV path.
# WholeMemory's host-shm coordination (create_and_map_shared_host_memory) defaults to SysV shmget
# with a ftok key = (comm_id starting at 0) + first-rank PID + a HARDCODED proj-id 0xE601EEEE shared
# by every wholegraph user. SysV segments survive process death and are only IPC_RMID'd on a clean
# exit, so a crashed prior run — or another user's wholegraph job colliding via ftok's 8-bit inode
# truncation — leaves a segment that makes rank 0's IPC_CREAT|IPC_EXCL shmget fail: "Create host
# shared memory from IPC key … Reason=File exists" (dirty node) or a wrong-size attach -> SIGSEGV
# (shared node). This reproducibly killed every >=4-rank run at create_embedding (memory_handle.cpp,
# verified on branch-24.10/main; our wholegraph is 26.04). WG_USE_POSIX_SHM=1 switches to shm_open
# with O_CREAT (no O_EXCL, so a stale file can't fatal) and rank 0 shm_unlinks immediately after
# attach, so nothing leaks or collides across runs. See docs/phase4-full-cc-plan.md §5.
export WG_USE_POSIX_SHM=1

echo "=== torchrun --nproc-per-node $NPROC (GPUs visible: $(nvidia-smi -L 2>/dev/null | wc -l)) | WG_USE_POSIX_SHM=$WG_USE_POSIX_SHM ==="
torchrun --standalone --nnodes 1 --nproc-per-node "$NPROC" "$HERE/cupyg_lp.py" \
  --tb-logdir "$TB/train" "$@"
RC=$?

kill -TERM "$SYS" 2>/dev/null
wait "$SYS" 2>/dev/null

echo "=== TensorBoard: $TB (system) ==="
exit "$RC"
