"""
Throughput + GPU memory benchmark: all 4 backends across N.

Measures:
  throughput (Mcell/s) — fixed N_STEPS, median of N_BENCH runs
  peak GPU memory (MiB) — nvidia-smi delta after solver creation + warmup

Saves: benchmarks/scaling/plot_throughput.png  and  plot_memory.png

Run from examples/warplabs_fluids/:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/scaling/throughput_memory.py
"""

import gc
import os
import subprocess
import sys
import statistics
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warp as wp

# disable JAX GPU pre-allocation so we measure real footprint
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).parent.parent.parent
OUT  = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from warplabs_fluids import WarpEuler1D, prim_to_cons
from cases.sod import ic as sod_ic
from benchmarks.jax_euler import JaxEuler1D

# ── config ────────────────────────────────────────────────────────────────────
GRID_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
GAMMA      = 1.4
CFL        = 0.4
N_STEPS    = 200
N_BENCH    = 5
MAX_WALL_S = 60.0

# ── GPU memory helpers ────────────────────────────────────────────────────────

def _nvml_mem_mib():
    """Current used GPU-0 memory in MiB via nvidia-smi. Returns 0.0 on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--id=0",
             "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


def _theory_warp_mib(N):
    """Theoretical Warp GPU footprint: Q + Q_stage, float32."""
    return 2 * 3 * (N + 4) * 4 / (1024 ** 2)


def _stable_dt(N):
    dx = 1.0 / N
    return CFL * dx / np.sqrt(GAMMA)     # a_max ~ sqrt(gamma) for Sod left state


# ── per-solver benchmark fns ──────────────────────────────────────────────────

def bench_warp(device, N):
    dt   = _stable_dt(N)
    Q0,_ = sod_ic(N, GAMMA)

    wp.synchronize()
    mem0 = _nvml_mem_mib()

    solver = WarpEuler1D(N, 1.0/N, gamma=GAMMA, bc="outflow", device=device)
    solver.initialize(Q0)

    t0w = time.perf_counter()
    for _ in range(N_STEPS):
        solver.step(dt)
    wp.synchronize()
    if time.perf_counter() - t0w > MAX_WALL_S:
        return None, None

    mem_peak  = _nvml_mem_mib()
    mem_delta = max(0.0, mem_peak - mem0)

    times = []
    for _ in range(N_BENCH):
        solver.initialize(Q0)
        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            solver.step(dt)
        wp.synchronize()
        times.append(time.perf_counter() - t0)

    tp = N * N_STEPS / statistics.median(times) / 1e6
    del solver
    gc.collect()
    return tp, mem_delta


def bench_jax(N, jax_device):
    import jax
    dt   = _stable_dt(N)
    Q0,_ = sod_ic(N, GAMMA)

    mem0 = _nvml_mem_mib()

    with jax.default_device(jax_device):
        solver = JaxEuler1D(N, 1.0/N, gamma=GAMMA, bc="outflow")
        solver.initialize(Q0)

        t0w = time.perf_counter()
        for _ in range(N_STEPS):
            solver.step(dt)
        jax.block_until_ready(solver._Q)
        if time.perf_counter() - t0w > MAX_WALL_S:
            return None, None

        mem_peak  = _nvml_mem_mib()
        mem_delta = max(0.0, mem_peak - mem0)

        times = []
        for _ in range(N_BENCH):
            solver.initialize(Q0)
            t0 = time.perf_counter()
            for _ in range(N_STEPS):
                solver.step(dt)
            jax.block_until_ready(solver._Q)
            times.append(time.perf_counter() - t0)

    tp = N * N_STEPS / statistics.median(times) / 1e6
    del solver
    gc.collect()
    return tp, mem_delta


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import jax

    wp.init()

    jax_cpu = jax.devices("cpu")[0]
    try:
        jax_gpus = jax.devices("gpu")
        jax_gpu  = jax_gpus[0] if jax_gpus else None
    except Exception:
        jax_gpu = None

    if jax_gpu:
        print(f"[info] JAX GPU: {jax_gpu}  —  4-backend run")
    else:
        print("[info] No JAX GPU  —  3-backend run")

    solvers = {
        "JAX CPU":   (lambda N: bench_jax(N, jax_cpu),   "#e07b00", "o--", False),
        "Warp CPU":  (lambda N: bench_warp("cpu",  N),    "#0072b2", "s--", False),
        "Warp CUDA": (lambda N: bench_warp("cuda", N),    "#009e73", "^-",  True),
    }
    if jax_gpu:
        solvers["JAX CUDA"] = (lambda N: bench_jax(N, jax_gpu), "#d55e00", "D-", True)

    ordered = ["JAX CPU", "Warp CPU", "JAX CUDA", "Warp CUDA"]
    solvers = {k: solvers[k] for k in ordered if k in solvers}

    results = {name: {"N": [], "tp": [], "mem": []} for name in solvers}

    for N in GRID_SIZES:
        print(f"\nN = {N:>7}", flush=True)
        for name, (fn, *_) in solvers.items():
            tp, mem = fn(N)
            if tp is None:
                print(f"  {name:<12}  skipped (>{MAX_WALL_S}s warmup)")
            else:
                results[name]["N"].append(N)
                results[name]["tp"].append(tp)
                results[name]["mem"].append(mem)
                mem_str = f"  mem Δ={mem:.1f} MiB" if mem > 0.01 else ""
                print(f"  {name:<12}  {tp:8.2f} Mcell/s{mem_str}")

    # ── table ────────────────────────────────────────────────────────────────
    print("\n-- Throughput (Mcell/s) ----------------------------------------")
    hdr = f"{'N':>8}" + "".join(f"  {n:>12}" for n in solvers)
    print(hdr); print("-" * len(hdr))
    for N in GRID_SIZES:
        row = f"{N:>8}"
        for d in results.values():
            if N in d["N"]:
                row += f"  {d['tp'][d['N'].index(N)]:>12.2f}"
            else:
                row += f"  {'--':>12}"
        print(row)

    all_N = sorted({n for d in results.values() for n in d["N"]})
    subtitle = (
        "1-D Euler  |  WENO3-HLLC-RK2 (fused)  |  200 steps, median of 5 runs\n"
        "Linux WSL2 Ubuntu 22.04  ·  RTX 5000 Ada  ·  Warp 1.12.1  ·  JAX 0.6.2"
    )

    # ── throughput plot ───────────────────────────────────────────────────────
    fig_tp, ax_tp = plt.subplots(figsize=(9, 6))
    fig_tp.suptitle(subtitle, fontsize=9)

    for name, (fn, color, mls, _) in solvers.items():
        d = results[name]
        if not d["N"]:
            continue
        m, ls = mls[0], mls[1:]
        ax_tp.plot(d["N"], d["tp"], marker=m, ls=ls, color=color,
                   lw=1.8, ms=7, label=name)

    ax_tp.set_xscale("log", base=2)
    ax_tp.set_yscale("log", base=2)
    ax_tp.set_xlabel("Grid size  N", fontsize=12)
    ax_tp.set_ylabel("Throughput  (Mcell-updates / s)", fontsize=12)
    ax_tp.set_title("Throughput scaling", fontsize=13)
    ax_tp.legend(fontsize=11)
    ax_tp.grid(True, which="both", lw=0.4, alpha=0.5)
    ax_tp.set_xticks(all_N)
    ax_tp.set_xticklabels([str(n) for n in all_N], rotation=35, ha="right", fontsize=8)

    fig_tp.tight_layout()
    out_tp = OUT / "plot_throughput.png"
    fig_tp.savefig(out_tp, dpi=150, bbox_inches="tight")
    plt.close(fig_tp)
    print(f"\nSaved -> {out_tp}")

    # ── memory plot ───────────────────────────────────────────────────────────
    fig_mem, ax_mem = plt.subplots(figsize=(9, 6))
    fig_mem.suptitle(subtitle, fontsize=9)

    has_mem = False
    for name, (fn, color, mls, show_mem) in solvers.items():
        d = results[name]
        if not show_mem or not d["N"]:
            continue
        m, ls = mls[0], mls[1:]
        mems = d["mem"]
        if any(v > 0.01 for v in mems):
            ax_mem.plot(d["N"], mems, marker=m, ls=ls, color=color,
                        lw=1.8, ms=7, label=f"{name} (measured)")
            has_mem = True

    N_th = np.array(all_N, dtype=float)
    th = [_theory_warp_mib(n) for n in all_N]
    ax_mem.plot(N_th, th, color="0.5", ls=":", lw=1.4, marker=".",
                label="theory  2 arrays × 3 vars × N × 4 B")

    ax_mem.set_xscale("log", base=2)
    ax_mem.set_yscale("log", base=2)
    ax_mem.set_xlabel("Grid size  N", fontsize=12)
    ax_mem.set_ylabel("Peak GPU memory  Δ (MiB)", fontsize=12)
    ax_mem.set_title("GPU memory footprint  (XLA_PYTHON_CLIENT_PREALLOCATE=false)", fontsize=12)
    ax_mem.legend(fontsize=11)
    ax_mem.grid(True, which="both", lw=0.4, alpha=0.5)
    ax_mem.set_xticks(all_N)
    ax_mem.set_xticklabels([str(n) for n in all_N], rotation=35, ha="right", fontsize=8)

    if not has_mem:
        ax_mem.text(0.5, 0.5, "nvidia-smi not available\n(theoretical line only)",
                    transform=ax_mem.transAxes, ha="center", va="center",
                    fontsize=11, color="0.5")

    fig_mem.tight_layout()
    out_mem = OUT / "plot_memory.png"
    fig_mem.savefig(out_mem, dpi=150, bbox_inches="tight")
    plt.close(fig_mem)
    print(f"Saved -> {out_mem}")


if __name__ == "__main__":
    main()
