"""
Shu-Osher self-convergence study — Warp WENO5-Z+HLLC+RK3 (f32).

No exact solution — uses N_ref=4096 as high-resolution reference.
Block-averages reference to coarser grids, computes L1(rho) vs N.

Test grids: N = 256, 512, 1024, 2048
Reference:  N = 4096

Saves (to benchmarks/shu_osher/):
  shu_osher_convergence.png
  convergence.csv

Run from projects/warplabs_fluids/:
  python benchmarks/shu_osher/convergence_study.py
"""

import csv, gc, sys
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

GRID_SIZES = [256, 512, 1024, 2048]
N_REF      = 4096
CFL        = 0.4

COLORS = {256: "#d55e00", 512: "#e07b00", 1024: "#0072b2", 2048: "#009e73", 4096: "0.3"}
LS     = {256: (0, [3, 2]), 512: "--", 1024: "-.", 2048: "-", 4096: "-"}


def run(N, device="cuda"):
    Q0, x = shu_ic(N, GAMMA)
    solver = WarpEuler1D(N, L/N, gamma=GAMMA, bc="outflow", device=device, scheme="weno5z-rk3")
    solver.initialize(Q0)
    solver.run(T_END, CFL)
    if device == "cuda": wp.synchronize()
    rho, u, p = cons_to_prim(solver.state, GAMMA)
    del solver; gc.collect()
    return rho, u, p, x


def block_avg(arr, factor):
    N = len(arr)
    return arr.reshape(N // factor, factor).mean(axis=1)


def main():
    wp.init()
    try:
        wp.get_device("cuda"); device = "cuda"
    except Exception:
        device = "cpu"; print("[info] CUDA not available, using CPU")

    print(f"\nRunning reference solution N={N_REF} ...", flush=True)
    rho_ref, u_ref, p_ref, x_ref = run(N_REF, device)
    print(f"  N={N_REF} done  ({len(rho_ref)} cells)")

    coarse = {}
    for N in GRID_SIZES:
        factor = N_REF // N
        print(f"\nRunning N={N} ...", flush=True)
        rho, u, p, x = run(N, device)
        rho_ref_c = block_avg(rho_ref, factor)
        l1 = float(np.mean(np.abs(rho - rho_ref_c)))
        coarse[N] = dict(rho=rho, u=u, p=p, x=x, l1=l1)
        print(f"  N={N:<5}  L1(rho) = {l1:.3e}")

    print("\n-- Self-convergence (L1 density vs N_ref=4096) --")
    print(f"{'N':>6}  {'L1(rho)':>10}  {'rate':>8}")
    print("-" * 30)
    Ns = sorted(coarse)
    for i, N in enumerate(Ns):
        l1 = coarse[N]["l1"]
        if i > 0:
            rate = np.log2(coarse[Ns[i-1]]["l1"] / l1) / np.log2(N / Ns[i-1])
            print(f"{N:>6}  {l1:>10.3e}  {rate:>8.2f}")
        else:
            print(f"{N:>6}  {l1:>10.3e}  {'--':>8}")

    # ── save CSV ──────────────────────────────────────────────────────────────
    with open(OUT / "convergence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solver", "N", "N_ref", "L1_rho"])
        for N in Ns:
            w.writerow(["Warp WENO5-Z (f32)", N, N_REF, coarse[N]["l1"]])
    print(f"\nSaved -> {OUT/'convergence.csv'}")

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5))
    ax_full = fig.add_subplot(1, 3, 1)
    ax_zoom = fig.add_subplot(1, 3, 2)
    ax_conv = fig.add_subplot(1, 3, 3)
    fig.suptitle(
        f"Shu-Osher self-convergence  ·  N_ref={N_REF}  ·  t={T_END}\n"
        "Mach-3 shock + sin density  ·  Warp WENO5-Z+HLLC+RK3 (fused, float32)",
        fontsize=10, fontweight="bold")

    ax_full.plot(x_ref, rho_ref, color=COLORS[N_REF], ls="-", lw=0.8,
                 label=f"N={N_REF} (ref)", zorder=5)
    ax_zoom.plot(x_ref, rho_ref, color=COLORS[N_REF], ls="-", lw=0.8,
                 label=f"N={N_REF} (ref)", zorder=5)

    l1_vals = []
    for N in sorted(coarse):
        d = coarse[N]; ls = LS[N]
        kw = dict(color=COLORS[N], ls=ls, lw=1.6, label=f"N={N}")
        ax_full.plot(d["x"], d["rho"], **kw)
        ax_zoom.plot(d["x"], d["rho"], **kw)
        l1_vals.append(d["l1"])

    ax_full.set_xlabel("x", fontsize=10); ax_full.set_ylabel("density  ρ", fontsize=10)
    ax_full.set_title(f"Density at t={T_END}", fontsize=11)
    ax_full.set_xlim(0, L); ax_full.legend(fontsize=8, loc="upper left")
    ax_full.grid(True, lw=0.4, alpha=0.5)

    ax_zoom.set_xlabel("x", fontsize=10)
    ax_zoom.set_title("Post-shock region  (zoomed)", fontsize=11)
    ax_zoom.set_xlim(1.5, 6.5); ax_zoom.legend(fontsize=8, loc="upper left")
    ax_zoom.grid(True, lw=0.4, alpha=0.5)

    Ns_arr = np.array(sorted(coarse))
    ax_conv.loglog(Ns_arr, l1_vals, "o-", color="#0072b2", lw=1.8, ms=7, label="L1(rho)")
    x0, y0 = Ns_arr[0], l1_vals[0]
    ax_conv.loglog(Ns_arr, y0 * (Ns_arr / x0)**(-1.0), color="0.6", ls="--", lw=1.0, label="O(N⁻¹)")
    ax_conv.loglog(Ns_arr, y0 * (Ns_arr / x0)**(-2.0), color="0.8", ls=":",  lw=1.0, label="O(N⁻²)")
    ax_conv.set_xlabel("N", fontsize=10); ax_conv.set_ylabel("L1(ρ)  vs N_ref=4096", fontsize=10)
    ax_conv.set_title("Self-convergence", fontsize=11)
    ax_conv.set_xticks(Ns_arr); ax_conv.set_xticklabels([str(n) for n in Ns_arr], fontsize=8)
    ax_conv.legend(fontsize=9); ax_conv.grid(True, which="both", lw=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(OUT / "shu_osher_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT/'shu_osher_convergence.png'}")


if __name__ == "__main__":
    main()
