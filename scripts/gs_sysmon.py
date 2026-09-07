"""System-metrics sidecar -> TensorBoard (Phase 3.1 throughput / GPU-utilization profiling).

GraphStorm's ``tensorboard_task_tracker`` logs only *training* metrics (loss/MRR). This sidecar adds
the *system* metrics the optimization needs: GPU utilization %, VRAM, power, temperature (via NVML)
and CPU % / RAM (via psutil), sampled every ``--interval`` seconds into a TensorBoard SummaryWriter
under the same parent logdir. It runs until SIGTERM (stopped by the wrapper) and prints a one-line
summary (mean/max GPU util, mean VRAM/CPU) to stdout — so the headline utilization is readable
straight from the job log, no TensorBoard needed.

    python gs_sysmon.py --logdir <tb_dir>/system --interval 3
"""

import argparse
import signal
import time

import psutil
from torch.utils.tensorboard import SummaryWriter

try:
    import pynvml

    pynvml.nvmlInit()
    _H = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML = True
except Exception as exc:
    print(f"[sysmon] NVML unavailable ({exc}); logging CPU/RAM only", flush=True)
    NVML = False


def main() -> None:  # noqa: D103
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()

    writer = SummaryWriter(args.logdir)
    run = [True]

    def _stop(*_a: object) -> None:
        run[0] = False

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop)

    psutil.cpu_percent()  # prime the first (0.0) reading
    gpu_utils: list[float] = []
    mem_used: list[float] = []
    cpu_utils: list[float] = []
    t0 = time.time()
    i = 0
    while run[0]:
        i += 1
        if NVML:
            util = pynvml.nvmlDeviceGetUtilizationRates(_H)
            mem = pynvml.nvmlDeviceGetMemoryInfo(_H)
            writer.add_scalar("gpu/util_pct", util.gpu, i)
            writer.add_scalar("gpu/mem_io_pct", util.memory, i)
            writer.add_scalar("gpu/mem_used_GB", mem.used / 1e9, i)
            writer.add_scalar("gpu/power_W", pynvml.nvmlDeviceGetPowerUsage(_H) / 1000.0, i)
            writer.add_scalar(
                "gpu/temp_C",
                pynvml.nvmlDeviceGetTemperature(_H, pynvml.NVML_TEMPERATURE_GPU),
                i,
            )
            gpu_utils.append(float(util.gpu))
            mem_used.append(mem.used / 1e9)
        cpu = psutil.cpu_percent()
        vm = psutil.virtual_memory()
        writer.add_scalar("cpu/util_pct", cpu, i)
        writer.add_scalar("cpu/ram_used_GB", vm.used / 1e9, i)
        writer.add_scalar("meta/elapsed_s", time.time() - t0, i)
        writer.flush()
        cpu_utils.append(cpu)
        time.sleep(args.interval)

    writer.close()

    def _stat(xs: list[float]) -> str:
        if not xs:
            return "n/a"
        return f"mean={sum(xs) / len(xs):.1f} max={max(xs):.1f}"

    print(
        f"[sysmon] summary over {i} samples ({time.time() - t0:.0f}s): "
        f"GPU util%({_stat(gpu_utils)}) VRAM_GB({_stat(mem_used)}) CPU%({_stat(cpu_utils)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
