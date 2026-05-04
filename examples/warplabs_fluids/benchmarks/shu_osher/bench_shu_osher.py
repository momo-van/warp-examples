"""
Shu-Osher shock-density interaction: Warp CUDA vs JAX CUDA.

Mach-3 shock propagating into a sinusoidal density field.
No exact solution — compares the two solvers head-to-head.

N = 256, 512, 1024  |  domain [0, 10]  |  t_end = 1.8

Saves:
  shu_osher_profiles.png  — density/velocity/pressure at t=1.8
  shu_osher_throughput.png — throughput comparison (Mcell/s)

Run from examples/warplabs_fluids/:
  python benchmarks/shu_osher/bench_shu_osher.py
"""

import sys, time, statistics, gc
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warp as wp

ROOT = Path(__file__).parent.parent.parent
OUT  = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from warplabs_fluids import WarpEuler1D, cons_to_prim
from cases.shu_osher import ic as shu_ic, L, T_END, GAMMA
from benchmarks.jax_euler import JaxEuler1D

GRID_SIZES = [256, 512, 1024]
CFL        = 0.4
N_BENCH    = 3
N_WARM     = 1


def run_warp(N):
    dx = L / N
    Q0, x = shu_ic(N, GAMMA)
    solver = WarpEuler1D(N, dx, gamma=GAMMA, bc="outflow", device="cuda")
    for _ in range(N_WARM):
        solver.initialize(Q0); solver.run(T_END, CFL)
    wp.synchronize()
    times = []; n_steps = 0
    for _ in range(N_BENCH):
        solver.initialize(Q0)
        t0 = time.perf_counter()
        n_steps = solver.run(T_END, CFL)
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    t_med = statistics.median(times)
    tp = N * n_steps / t_med / 1e6
    rho, u, p = cons_to_prim(solver.state, GAMMA)
    del solver; gc.collect()
    return tp, rho, u, p, x, n_steps, t_med


def run_jax(N):
    import jax
    dx = L / N
    Q0, x = shu_ic(N, GAMMA)
    gpu = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]
    n_steps = 0
    with jax.default_device(gpu):
        solver = JaxEuler1D(N, dx, gamma=GAMMA, bc="outflow")
        for _ in range(N_WARM):
            solver.initialize(Q0); n_steps = solver.run(T_END, CFL)
        jax.block_until_ready(solver._Q)
        times = []
        for _ in range(N_BENCH):
            solver.initialize(Q0)
            t0 = time.perf_counter()
            n_steps = solver.run(T_END, CFL)
            jax.block_until_ready(solver._Q)
            times.append(time.perf_counter() - t0)
    t_med = statistics.median(times)
    tp = N * n_steps / t_med / 1e6
    Q = np.asarray(solver._Q)
    rho, u, p = cons_to_prim(Q, GAMMA)
    del solver; gc.collect()
    return tp, rho, u, p, x, n_steps, t_med


def main():
    wp.init()

    results = {}

    for N in GRID_SIZES:
        print(f"\nN = {N}", flush=True)

        try:
            tp, rho, u, p, x, ns, tm = run_warp(N)
            results.setdefault("Warp CUDA", []).append(
                dict(N=N, tp=tp, rho=rho, u=u, p=p, x=x, steps=ns))
            print(f"  Warp CUDA  {tp:8.2f} Mcell/s  ({ns} steps, {tm:.2f}s)")
        except Exception as e:
            print(f"  Warp CUDA  ERROR: {e}")

        try:
            tp, rho, u, p, x, ns, tm = run_jax(N)
            results.setdefault("JAX CUDA", []).append(
                dict(N=N, tp=tp, rho=rho, u=u, p=p, x=x, steps=ns))
            print(f"  JAX CUDA   {tp:8.2f} Mcell/s  ({ns} steps, {tm:.2f}s)")
        except Exception as e:
            print(f"  JAX CUDA   ERROR: {e}")

    # ── throughput table ──────────────────────────────────────────────────────
    print("\n-- Throughput (Mcell/s) -----------------------------------------")
    print(f"{'N':>6}  {'Warp CUDA':>12}  {'JAX CUDA':>12}  {'Speedup':>8}")
    print("-" * 45)
    for N in GRID_SIZES:
        w = next((r["tp"] for r in results.get("Warp CUDA", []) if r["N"] == N), None)
        j = next((r["tp"] for r in results.get("JAX CUDA",  []) if r["N"] == N), None)
        sp = f"{w/j:.2f}×" if w and j else "--"
        ws = f"{w:.2f}" if w else "--"
        js = f"{j:.2f}" if j else "--"
        print(f"{N:>6}  {ws:>12}  {js:>12}  {sp:>8}")

    # throughput style: markers OK (only 3 points)
    tp_style = {
        "Warp CUDA": dict(color="#009e73", marker="^", ls="-",  lw=1.8, ms=7),
        "JAX CUDA":  dict(color="#d55e00", marker="o", ls="--", lw=1.8, ms=7),
    }
    # profile style: clean solid/dashed lines, no markers (1000+ points per line)
    prof_style = {
        "Warp CUDA": dict(color="#009e73", ls="-",  lw=1.8),
        "JAX CUDA":  dict(color="#d55e00", ls="--", lw=1.5),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, runs in results.items():
        ns_list = [r["N"]  for r in runs]
        tp_list = [r["tp"] for r in runs]
        ax.plot(ns_list, tp_list, **tp_style[name], label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of cells  N", fontsize=12)
    ax.set_ylabel("Throughput  (Mcell-updates / s)", fontsize=12)
    ax.set_title(
        f"Shu-Osher throughput  |  t_end={T_END}  |  WENO3-HLLC-RK2  |  float32",
        fontsize=10)
    ax.legend(fontsize=11); ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.set_xticks(GRID_SIZES)
    ax.set_xticklabels([str(n) for n in GRID_SIZES], fontsize=9)
    plt.tight_layout()
    out = OUT / "shu_osher_throughput.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved -> {out}")

    # ── profiles at N=512 (or largest available) ──────────────────────────────
    N_prof = max(N for N in GRID_SIZES
                 if any(r["N"] == N for r in results.get("Warp CUDA", []))
                 or any(r["N"] == N for r in results.get("JAX CUDA",  [])))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"Shu-Osher shock-density interaction  |  N={N_prof}  |  t={T_END}\n"
        "Mach-3 shock + sinusoidal density  ·  WENO3-HLLC-RK2 (fused, float32)",
        fontsize=10, fontweight="bold")
    field_names = ["density  ρ", "velocity  u", "pressure  p"]
    field_keys  = ["rho", "u", "p"]

    x_ref = None
    for name, runs in results.items():
        r = next((r for r in runs if r["N"] == N_prof), None)
        if r is None:
            continue
        x_ref = r["x"]
        for ax, fname, fkey in zip(axes, field_names, field_keys):
            ax.plot(r["x"], r[fkey], **prof_style[name], label=name, alpha=0.9)

    # overlay IC (density only)
    Q0_ic, x_ic = shu_ic(N_prof, GAMMA)
    rho0, u0, p0 = cons_to_prim(Q0_ic, GAMMA)
    for ax, ic_val in zip(axes, [rho0, u0, p0]):
        ax.plot(x_ic, ic_val, color="0.72", lw=1.0, ls=":", label="t=0 (IC)", zorder=1)

    for ax, fname in zip(axes, field_names):
        ax.set_xlabel("x", fontsize=10); ax.set_title(fname, fontsize=11)
        ax.set_xlim(0, L); ax.legend(fontsize=8); ax.grid(True, lw=0.4, alpha=0.5)

    plt.tight_layout()
    out2 = OUT / "shu_osher_profiles.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out2}")


if __name__ == "__main__":
    main()
