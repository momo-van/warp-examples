"""
Sod shock tube: warplabs-fluids vs JAX reference.

Produces:
  sod_profiles.png   — density / velocity / pressure profiles + exact solution
  sod_benchmark.png  — throughput comparison (Mcell-updates / second)

Run from examples/warplabs_fluids/:
  python benchmarks/sod/compare_sod.py
"""

import sys, time, statistics, contextlib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warp as wp

ROOT   = Path(__file__).parent.parent.parent
OUT    = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from warplabs_fluids import WarpEuler1D, prim_to_cons, cons_to_prim, l1_error, l2_error
from cases.sod import ic as sod_ic, exact as sod_exact
from benchmarks.jax_euler import JaxEuler1D

N       = 512
GAMMA   = 1.4
DX      = 1.0 / N
T_END   = 0.2
CFL     = 0.4
N_BENCH = 5
N_WARM  = 3


def _run_warp(solver, Q0, n_warm, n_bench):
    for _ in range(n_warm):
        solver.initialize(Q0)
        n_steps = solver.run(T_END, CFL)
    wp.synchronize()
    times = []
    for _ in range(n_bench):
        solver.initialize(Q0)
        t0      = time.perf_counter()
        n_steps = solver.run(T_END, CFL)
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return solver.state, n_steps, med, N * n_steps / med / 1e6


def _run_jax(solver, Q0, n_warm, n_bench, jax_device=None):
    import jax
    ctx = jax.default_device(jax_device) if jax_device else contextlib.nullcontext()
    with ctx:
        for _ in range(n_warm):
            solver.initialize(Q0)
            n_steps = solver.run(T_END, CFL)
        jax.block_until_ready(solver._Q)
        times = []
        for _ in range(n_bench):
            solver.initialize(Q0)
            t0      = time.perf_counter()
            n_steps = solver.run(T_END, CFL)
            jax.block_until_ready(solver._Q)
            times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return solver.state, n_steps, med, N * n_steps / med / 1e6


def main():
    import jax
    wp.init()
    Q0, x = sod_ic(N, GAMMA)

    jax_gpu = next((d for d in jax.devices() if d.platform == "gpu"), None)
    results = {}

    print(f"[1] JAX CPU ...", flush=True)
    s = JaxEuler1D(N, DX, gamma=GAMMA, bc="outflow")
    Q, n, t, tp = _run_jax(s, Q0, N_WARM, N_BENCH, jax.devices("cpu")[0])
    results["JAX CPU"] = dict(Q=Q, n=n, t=t, tp=tp, color="#e07b00", ls="--")
    print(f"   {tp:.2f} Mcell/s")

    if jax_gpu:
        print(f"[2] JAX CUDA ...", flush=True)
        s = JaxEuler1D(N, DX, gamma=GAMMA, bc="outflow")
        Q, n, t, tp = _run_jax(s, Q0, N_WARM, N_BENCH, jax_gpu)
        results["JAX CUDA"] = dict(Q=Q, n=n, t=t, tp=tp, color="#d55e00", ls="--")
        print(f"   {tp:.2f} Mcell/s")

    print(f"[3] Warp CPU ...", flush=True)
    s = WarpEuler1D(N, DX, gamma=GAMMA, bc="outflow", device="cpu")
    Q, n, t, tp = _run_warp(s, Q0, N_WARM, N_BENCH)
    results["Warp CPU"] = dict(Q=Q, n=n, t=t, tp=tp, color="#0072b2", ls="-")
    print(f"   {tp:.2f} Mcell/s")

    print(f"[4] Warp CUDA ...", flush=True)
    try:
        s = WarpEuler1D(N, DX, gamma=GAMMA, bc="outflow", device="cuda")
        Q, n, t, tp = _run_warp(s, Q0, N_WARM, N_BENCH)
        results["Warp CUDA"] = dict(Q=Q, n=n, t=t, tp=tp, color="#009e73", ls="-")
        print(f"   {tp:.2f} Mcell/s")
    except Exception as e:
        print(f"   CUDA unavailable: {e}")

    rho_ex, u_ex, p_ex = sod_exact(T_END, x, GAMMA)

    print("\n-- Accuracy vs exact Riemann solution ----------------------")
    print(f"{'Solver':<14}  {'L1(rho)':>10}  {'L1(u)':>10}  {'L1(p)':>10}")
    for name, r in results.items():
        rho, u, p = cons_to_prim(np.asarray(r["Q"]), GAMMA)
        print(f"{name:<14}  {l1_error(rho,rho_ex,DX):>10.3e}  "
              f"{l1_error(u,u_ex,DX):>10.3e}  {l1_error(p,p_ex,DX):>10.3e}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"Sod shock tube  |  N={N}  |  t={T_END}", fontsize=12, fontweight="bold")
    fields, exact_v = ["density","velocity","pressure"], [rho_ex,u_ex,p_ex]
    rho0,u0,p0 = cons_to_prim(Q0.copy(), GAMMA)
    for ax, fname, ev, ic in zip(axes, fields, exact_v, [rho0,u0,p0]):
        ax.plot(x, ic, color="0.75", lw=1.2, ls=":", label="t=0 (IC)")
        ax.plot(x, ev, color="k",    lw=1.6, ls="-",  label="exact", zorder=5)
        for name, r in results.items():
            rho,u,p = cons_to_prim(np.asarray(r["Q"]), GAMMA)
            ax.plot(x, [rho,u,p][fields.index(fname)],
                    color=r["color"], lw=1.4, ls=r["ls"], label=name, alpha=0.9)
        ax.set_xlabel("x"); ax.set_title(fname)
        ax.legend(fontsize=7); ax.grid(True, lw=0.4, alpha=0.5)
    plt.tight_layout()
    out = OUT / "sod_profiles.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
