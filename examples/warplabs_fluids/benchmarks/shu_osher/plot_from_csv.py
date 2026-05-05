"""
Regenerate all Shu-Osher benchmark PNGs from saved CSV files (no GPU required).

Reads:
  bench_jaxfluids_throughput.csv
  bench_jaxfluids_profiles_N512.csv
  throughput_scaling.csv
  memory_scaling.csv
  convergence.csv

Saves:
  jaxfluids_throughput.png
  jaxfluids_profiles.png
  shu_osher_scaling.png
  shu_osher_memory.png
  shu_osher_convergence.png

Run from examples/warplabs_fluids/ on Windows (no GPU needed):
  python benchmarks/shu_osher/plot_from_csv.py
"""

import csv, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
OUT  = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from cases.shu_osher import L, T_END

COLORS = {
    "JaxFluids (WENO5-Z, f64)": "#e07b00",
    "Warp WENO5-Z (f32)":       "#009e73",
    "Warp CPU  (WENO5-Z, f32)": "#0072b2",
    "Warp CUDA (WENO5-Z, f32)": "#009e73",
}
STYLES = {
    "JaxFluids (WENO5-Z, f64)": "D-",
    "Warp WENO5-Z (f32)":       "^-",
    "Warp CPU  (WENO5-Z, f32)": "s--",
    "Warp CUDA (WENO5-Z, f32)": "^-",
}


def _color(name):
    for k, v in COLORS.items():
        if k in name: return v
    return "#555555"


def _style(name):
    for k, v in STYLES.items():
        if k in name: return v
    return "o-"


def read_csv(path):
    if not path.exists():
        return None
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def plot_throughput(csv_path, out_path, title):
    rows = read_csv(csv_path)
    if not rows:
        print(f"[skip] {csv_path.name} not found"); return
    data = defaultdict(lambda: {"N": [], "tp": []})
    for r in rows:
        data[r["solver"]]["N"].append(int(r["N"]))
        data[r["solver"]]["tp"].append(float(r["throughput_Mcells"]))
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, d in data.items():
        m, ls = _style(name)[0], _style(name)[1:]
        ax.plot(d["N"], d["tp"], marker=m, ls=ls, color=_color(name), lw=1.8, ms=7, label=name)
    all_N = sorted({n for d in data.values() for n in d["N"]})
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel("Throughput  (Mcell-updates / s)", fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.set_xticks(all_N); ax.set_xticklabels([f"{n:,}" for n in all_N], rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_memory(csv_path, out_path, title, theory_fn=None):
    rows = read_csv(csv_path)
    if not rows:
        print(f"[skip] {csv_path.name} not found"); return
    data = defaultdict(lambda: {"N": [], "mem": []})
    for r in rows:
        data[r["solver"]]["N"].append(int(r["N"]))
        data[r["solver"]]["mem"].append(float(r["mem_MiB"]))
    fig, ax = plt.subplots(figsize=(9, 6))
    has_data = False
    for name, d in data.items():
        if not any(v > 0.01 for v in d["mem"]): continue
        m, ls = _style(name)[0], _style(name)[1:]
        ax.plot(d["N"], d["mem"], marker=m, ls=ls, color=_color(name), lw=1.8, ms=7,
                label=f"{name} (measured)")
        has_data = True
    all_N = sorted({n for d in data.values() for n in d["N"]})
    if theory_fn and all_N:
        N_th = np.array(all_N, dtype=float)
        ax.plot(N_th, [theory_fn(n) for n in all_N], color="0.5", ls=":", lw=1.4,
                marker=".", label="theory  3×3×(N+6)×4 B  (Warp RK3)")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel("Peak GPU memory  Δ (MiB)", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.set_xticks(all_N); ax.set_xticklabels([f"{n:,}" for n in all_N], rotation=35, ha="right", fontsize=8)
    if not has_data:
        ax.text(0.5, 0.5, "no measured data  (theoretical line only)",
                transform=ax.transAxes, ha="center", va="center", fontsize=11, color="0.5")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_profiles(csv_path, out_path):
    rows = read_csv(csv_path)
    if not rows:
        print(f"[skip] {csv_path.name} not found"); return
    cols = list(rows[0].keys())
    x = np.array([float(r["x"]) for r in rows])
    fields = ["rho", "u", "p"]
    field_labels = ["density  ρ", "velocity  u", "pressure  p"]

    # Load IC for overlay
    try:
        from warplabs_fluids import cons_to_prim as c2p
        from cases.shu_osher import ic as shu_ic, GAMMA
        N = len(x)
        Q0, x_ic = shu_ic(N, GAMMA)
        rho0, u0, p0 = c2p(Q0, GAMMA)
        ic_vals = [rho0, u0, p0]
    except Exception:
        ic_vals = None

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"Shu-Osher shock-density interaction  |  N=512  |  t={T_END}\n"
        "WENO5-Z+HLLC+RK3  |  JaxFluids f64 vs Warp f32",
        fontsize=10, fontweight="bold")

    for ax, fld, flabel, ic_v in zip(axes, fields, field_labels, ic_vals or [None]*3):
        if ic_v is not None:
            ax.plot(x, ic_v, color="0.72", lw=1.0, ls=":", label="t=0 (IC)", zorder=1)
        for col in cols:
            if not col.startswith(fld + "_"): continue
            tag = col[len(fld)+1:]
            vals = np.array([float(r[col]) if r[col] else np.nan for r in rows])
            ax.plot(x, vals, color=_color(tag), lw=1.4, ls="-", label=tag, alpha=0.9)
        ax.set_xlabel("x", fontsize=10); ax.set_title(flabel, fontsize=11)
        ax.set_xlim(0, L); ax.legend(fontsize=7); ax.grid(True, lw=0.4, alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_convergence(csv_path, out_path):
    rows = read_csv(csv_path)
    if not rows:
        print(f"[skip] {csv_path.name} not found"); return

    N_ref_val = int(rows[0].get("N_ref", 4096)) if rows else 4096
    data = defaultdict(lambda: {"N": [], "L1": []})
    for r in rows:
        data[r["solver"]]["N"].append(int(r["N"]))
        data[r["solver"]]["L1"].append(float(r["L1_rho"]))

    # load reference profile
    try:
        from warplabs_fluids import WarpEuler1D, cons_to_prim
        from cases.shu_osher import ic as shu_ic, L, T_END, GAMMA
        import warp as wp
        wp.init()
        N_ref = N_ref_val
        Q0, x_ref = shu_ic(N_ref, GAMMA)
        solver = WarpEuler1D(N_ref, L/N_ref, gamma=GAMMA, bc="outflow", device="cpu", scheme="weno5z-rk3")
        solver.initialize(Q0); solver.run(T_END, 0.4)
        rho_ref, _, _ = cons_to_prim(solver.state, GAMMA)
        del solver
        have_ref = True
    except Exception:
        rho_ref = x_ref = None; have_ref = False

    fig = plt.figure(figsize=(14, 5))
    ax_full = fig.add_subplot(1, 3, 1)
    ax_zoom = fig.add_subplot(1, 3, 2)
    ax_conv = fig.add_subplot(1, 3, 3)
    fig.suptitle(
        f"Shu-Osher self-convergence  ·  N_ref={N_ref_val}  ·  t={T_END}\n"
        "Warp WENO5-Z+HLLC+RK3 (fused, float32)",
        fontsize=10, fontweight="bold")

    COLORS_CONV = {256: "#d55e00", 512: "#e07b00", 1024: "#0072b2", 2048: "#009e73"}

    if have_ref:
        ax_full.plot(x_ref, rho_ref, color="0.3", ls="-", lw=0.8, label=f"N={N_ref_val} (ref)", zorder=5)
        ax_zoom.plot(x_ref, rho_ref, color="0.3", ls="-", lw=0.8, label=f"N={N_ref_val} (ref)", zorder=5)

        for name, d in data.items():
            for N, l1 in zip(d["N"], d["L1"]):
                c = COLORS_CONV.get(N, "#888")
                try:
                    from warplabs_fluids import WarpEuler1D, cons_to_prim
                    from cases.shu_osher import ic as shu_ic
                    Q0, x = shu_ic(N, GAMMA)
                    s = WarpEuler1D(N, L/N, gamma=GAMMA, bc="outflow", device="cpu", scheme="weno5z-rk3")
                    s.initialize(Q0); s.run(T_END, 0.4)
                    rho, _, _ = cons_to_prim(s.state, GAMMA)
                    del s
                    ax_full.plot(x, rho, color=c, lw=1.6, label=f"N={N}")
                    ax_zoom.plot(x, rho, color=c, lw=1.6, label=f"N={N}")
                except Exception:
                    pass

    ax_full.set_xlabel("x", fontsize=10); ax_full.set_ylabel("density  ρ", fontsize=10)
    ax_full.set_title(f"Density at t={T_END}", fontsize=11)
    ax_full.set_xlim(0, L); ax_full.legend(fontsize=8); ax_full.grid(True, lw=0.4, alpha=0.5)
    ax_zoom.set_xlabel("x", fontsize=10); ax_zoom.set_title("Post-shock region  (zoomed)", fontsize=11)
    ax_zoom.set_xlim(1.5, 6.5); ax_zoom.legend(fontsize=8); ax_zoom.grid(True, lw=0.4, alpha=0.5)

    for name, d in data.items():
        Ns_arr = np.array(d["N"]); l1_vals = d["L1"]
        ax_conv.loglog(Ns_arr, l1_vals, "o-", color="#0072b2", lw=1.8, ms=7, label="L1(rho)")
        x0, y0 = Ns_arr[0], l1_vals[0]
        ax_conv.loglog(Ns_arr, [y0 * (n/x0)**(-1.0) for n in Ns_arr], "0.6", ls="--", lw=1.0, label="O(N⁻¹)")
        ax_conv.loglog(Ns_arr, [y0 * (n/x0)**(-2.0) for n in Ns_arr], "0.8", ls=":",  lw=1.0, label="O(N⁻²)")
    ax_conv.set_xlabel("N", fontsize=10); ax_conv.set_ylabel(f"L1(ρ)  vs N_ref={N_ref_val}", fontsize=10)
    ax_conv.set_title("Self-convergence", fontsize=11)
    ax_conv.legend(fontsize=9); ax_conv.grid(True, which="both", lw=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out_path}")


def merge_throughput_csvs(*paths):
    """Merge multiple throughput CSVs, deduplicating on (solver, N) — last write wins."""
    merged = {}
    for path in paths:
        rows = read_csv(Path(path))
        if not rows:
            continue
        for r in rows:
            merged[(r["solver"], int(r["N"]))] = float(r["throughput_Mcells"])
    return merged


def plot_throughput_merged(csv_paths, out_path, title):
    merged = merge_throughput_csvs(*csv_paths)
    if not merged:
        print(f"[skip] no data for {out_path.name}"); return
    data = defaultdict(lambda: {"N": [], "tp": []})
    for (solver, N), tp in sorted(merged.items(), key=lambda x: (x[0][0], x[0][1])):
        data[solver]["N"].append(N)
        data[solver]["tp"].append(tp)
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, d in data.items():
        m, ls = _style(name)[0], _style(name)[1:]
        ax.plot(d["N"], d["tp"], marker=m, ls=ls, color=_color(name), lw=1.8, ms=7, label=name)
    all_N = sorted({n for d in data.values() for n in d["N"]})
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel("Throughput  (Mcell-updates / s)", fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.set_xticks(all_N); ax.set_xticklabels([f"{n:,}" for n in all_N], rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    theory = lambda N: 3 * 3 * (N + 6) * 4 / (1024 ** 2)

    plot_throughput(
        OUT / "bench_jaxfluids_throughput.csv",
        OUT / "jaxfluids_throughput.png",
        title="JaxFluids vs Warp  —  Shu-Osher throughput\nBoth: WENO5-Z+HLLC+RK3  |  JaxFluids f64  |  Warp f32")

    plot_profiles(
        OUT / "bench_jaxfluids_profiles_N512.csv",
        OUT / "jaxfluids_profiles.png")

    # scaling plot: merge Warp N-sweep + JaxFluids head-to-head data
    plot_throughput_merged(
        [OUT / "throughput_scaling.csv", OUT / "bench_jaxfluids_throughput.csv"],
        OUT / "shu_osher_scaling.png",
        title="Shu-Osher  —  throughput vs grid size  (WENO5-Z+HLLC+RK3)")

    plot_memory(
        OUT / "memory_scaling.csv",
        OUT / "shu_osher_memory.png",
        title="Shu-Osher  —  GPU memory vs grid size\n(XLA_PYTHON_CLIENT_PREALLOCATE=false)",
        theory_fn=theory)

    plot_convergence(
        OUT / "convergence.csv",
        OUT / "shu_osher_convergence.png")


if __name__ == "__main__":
    main()
