"""
Shu-Osher — throughput + GPU memory benchmark (all backends, N-sweep).

Measures:
  throughput (Mcell/s) — fixed N_STEPS, median of N_BENCH runs
  peak GPU memory (MiB) — nvidia-smi delta

Saves:
  shu_osher_scaling.png  — throughput vs N (log-log)
  shu_osher_memory.png   — peak GPU memory vs N (log-log)

Run from examples/warplabs_fluids/:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/shu_osher/throughput_memory.py
"""

import gc, os, subprocess, sys, statistics, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warp as wp

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).parent.parent.parent
OUT  = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from warplabs_fluids import WarpEuler1D
from cases.shu_osher import ic as shu_ic, L, GAMMA, RHO_L, U_L, P_L
from benchmarks.jax_euler import JaxEuler1D

GRID_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
CFL        = 0.4
N_STEPS    = 200
N_BENCH    = 5
MAX_WALL_S = 60.0

# Post-shock state sets the max wave speed: a_max = |U_L| + c_L
_c_L   = float(np.sqrt(GAMMA * P_L / RHO_L))
A_MAX  = abs(U_L) + _c_L    # ≈ 4.57 for Mach-3 Shu-Osher


def _nvml_mem_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


def _theory_mib(N):
    return 2 * 3 * (N + 4) * 4 / (1024 ** 2)


def _stable_dt(N):
    return CFL * (L / N) / A_MAX


def bench_warp(device, N):
    dt   = _stable_dt(N)
    Q0,_ = shu_ic(N, GAMMA)
    wp.synchronize()
    mem0   = _nvml_mem_mib()
    solver = WarpEuler1D(N, L/N, gamma=GAMMA, bc="outflow", device=device)
    solver.initialize(Q0)
    t0w = time.perf_counter()
    for _ in range(N_STEPS): solver.step(dt)
    wp.synchronize()
    if time.perf_counter() - t0w > MAX_WALL_S: return None, None
    mem_delta = max(0.0, _nvml_mem_mib() - mem0)
    times = []
    for _ in range(N_BENCH):
        solver.initialize(Q0)
        t0 = time.perf_counter()
        for _ in range(N_STEPS): solver.step(dt)
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    tp = N * N_STEPS / statistics.median(times) / 1e6
    del solver; gc.collect()
    return tp, mem_delta


def bench_jax(N, jax_device):
    import jax
    dt   = _stable_dt(N)
    Q0,_ = shu_ic(N, GAMMA)
    mem0 = _nvml_mem_mib()
    with jax.default_device(jax_device):
        solver = JaxEuler1D(N, L/N, gamma=GAMMA, bc="outflow")
        solver.initialize(Q0)
        t0w = time.perf_counter()
        for _ in range(N_STEPS): solver.step(dt)
        jax.block_until_ready(solver._Q)
        if time.perf_counter() - t0w > MAX_WALL_S: return None, None
        mem_delta = max(0.0, _nvml_mem_mib() - mem0)
        times = []
        for _ in range(N_BENCH):
            solver.initialize(Q0)
            t0 = time.perf_counter()
            for _ in range(N_STEPS): solver.step(dt)
            jax.block_until_ready(solver._Q)
            times.append(time.perf_counter() - t0)
    tp = N * N_STEPS / statistics.median(times) / 1e6
    del solver; gc.collect()
    return tp, mem_delta


def main():
    import jax
    wp.init()
    jax_cpu = jax.devices("cpu")[0]
    try:
        jax_gpu = jax.devices("gpu")[0]
    except Exception:
        jax_gpu = None

    solvers = {
        "JAX CPU":   (lambda N: bench_jax(N, jax_cpu),    "#e07b00", "o--", False),
        "Warp CPU":  (lambda N: bench_warp("cpu",  N),     "#0072b2", "s--", False),
        "Warp CUDA": (lambda N: bench_warp("cuda", N),     "#009e73", "^-",  True),
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
                print(f"  {name:<12}  skipped (>{MAX_WALL_S}s)")
            else:
                results[name]["N"].append(N)
                results[name]["tp"].append(tp)
                results[name]["mem"].append(mem)
                mem_str = f"  mem Δ={mem:.1f} MiB" if mem > 0.01 else ""
                print(f"  {name:<12}  {tp:8.2f} Mcell/s{mem_str}")

    print("\n-- Throughput (Mcell/s) ----------------------------------------")
    hdr = f"{'N':>8}" + "".join(f"  {n:>12}" for n in solvers)
    print(hdr); print("-" * len(hdr))
    for N in GRID_SIZES:
        row = f"{N:>8}"
        for d in results.values():
            if N in d["N"]: row += f"  {d['tp'][d['N'].index(N)]:>12.2f}"
            else:            row += f"  {'--':>12}"
        print(row)

    all_N = sorted({n for d in results.values() for n in d["N"]})
    subtitle = (
        f"1-D Euler  ·  Shu-Osher (Mach-3 shock + sin ρ)  ·  WENO3-HLLC-RK2 (fused)  ·  "
        f"200 steps, median of 5 runs\n"
        f"WSL2 Ubuntu 22.04  ·  RTX 5000 Ada  ·  Warp 1.12.1  ·  JAX 0.6.2"
    )

    # ── throughput plot ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(subtitle, fontsize=8, color="0.4")
    for name, (fn, color, mls, _) in solvers.items():
        d = results[name]
        if not d["N"]: continue
        m, ls = mls[0], mls[1:]
        ax.plot(d["N"], d["tp"], marker=m, ls=ls, color=color, lw=1.8, ms=7, label=name)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel("Throughput  (Mcell-updates / s)", fontsize=12)
    ax.set_title("Shu-Osher  —  throughput vs grid size", fontsize=13, pad=10)
    ax.legend(fontsize=11); ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.set_xticks(all_N); ax.set_xticklabels([f"{n:,}" for n in all_N], rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    out = OUT / "shu_osher_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved -> {out}")

    # ── memory plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(subtitle, fontsize=8, color="0.4")
    has_mem = False
    for name, (fn, color, mls, show_mem) in solvers.items():
        d = results[name]
        if not show_mem or not d["N"]: continue
        m, ls = mls[0], mls[1:]
        if any(v > 0.01 for v in d["mem"]):
            ax.plot(d["N"], d["mem"], marker=m, ls=ls, color=color, lw=1.8, ms=7,
                    label=f"{name} (measured)")
            has_mem = True
    N_th = np.array(all_N, dtype=float)
    ax.plot(N_th, [_theory_mib(n) for n in all_N], color="0.5", ls=":", lw=1.4,
            marker=".", label="theory  2×3×N×4 B")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel("Peak GPU memory  Δ (MiB)", fontsize=12)
    ax.set_title("Shu-Osher  —  GPU memory vs grid size\n(XLA_PYTHON_CLIENT_PREALLOCATE=false)",
                 fontsize=12, pad=10)
    ax.legend(fontsize=11); ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.set_xticks(all_N); ax.set_xticklabels([f"{n:,}" for n in all_N], rotation=35, ha="right", fontsize=8)
    if not has_mem:
        ax.text(0.5, 0.5, "nvidia-smi not available\n(theoretical line only)",
                transform=ax.transAxes, ha="center", va="center", fontsize=11, color="0.5")
    fig.tight_layout()
    out = OUT / "shu_osher_memory.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
