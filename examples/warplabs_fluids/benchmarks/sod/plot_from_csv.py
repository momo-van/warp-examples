"""
Regenerate all Sod benchmark PNGs from saved CSV files (no GPU required).

Reads:
  bench_jaxfluids_throughput.csv
  bench_jaxfluids_profiles_N512.csv
  bench_jaxfluids_accuracy.csv
  throughput_scaling.csv
  memory_scaling.csv
  convergence.csv

Saves:
  jaxfluids_throughput.png
  jaxfluids_profiles.png
  sod_scaling.png
  sod_memory.png
  sod_convergence.png

Run from examples/warplabs_fluids/ on Windows (no GPU needed):
  python benchmarks/sod/plot_from_csv.py
"""

import csv, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent

COLORS = {
    "JaxFluids (WENO5-Z, f64)": "#e07b00",
    "Warp WENO5-Z (f32)":       "#009e73",
    "Warp CPU  (WENO5-Z, f32)": "#0072b2",
    "Warp CUDA (WENO5-Z, f32)": "#009e73",
    "JaxFluids":                 "#e07b00",
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


def plot_throughput(csv_path, out_path, title, subtitle=None):
    rows = read_csv(csv_path)
    if not rows:
        print(f"[skip] {csv_path.name} not found"); return
    data = defaultdict(lambda: {"N": [], "tp": []})
    for r in rows:
        data[r["solver"]]["N"].append(int(r["N"]))
        data[r["solver"]]["tp"].append(float(r["throughput_Mcells"]))
    fig, ax = plt.subplots(figsize=(9, 6))
    if subtitle: fig.suptitle(subtitle, fontsize=8, color="0.4")
    for name, d in data.items():
        m, ls = _style(name)[0], _style(name)[1:]
        ax.plot(d["N"], d["tp"], marker=m, ls=ls, color=_color(name), lw=1.8, ms=7, label=name)
    all_N = sorted({n for d in data.values() for n in d["N"]})
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel("Throughput  (Mcell-updates / s)", fontsize=12)
    ax.set_title(title, fontsize=12)
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
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Sod shock tube  |  N=512  |  WENO5-Z+HLLC+RK3\nJaxFluids f64 vs Warp f32",
                 fontsize=10, fontweight="bold")
    for ax, fld, flabel in zip(axes, fields, field_labels):
        for col in cols:
            if not col.startswith(fld + "_"): continue
            tag = col[len(fld)+1:]
            vals = np.array([float(r[col]) if r[col] else np.nan for r in rows])
            if tag == "exact":
                ax.plot(x, vals, "k-", lw=2.0, label="exact Riemann", zorder=5)
            else:
                name = tag.replace("_", " ").replace("JaxFluids ", "JaxFluids (").replace("f64 ", "WENO5-Z, f64)").replace("Warp WENO5Z ", "Warp WENO5-Z (").replace("f32 ", "f32)")
                ax.plot(x, vals, color=_color(tag), lw=1.4, ls="-", label=tag, alpha=0.85)
        ax.set_xlabel("x"); ax.set_title(flabel)
        ax.legend(fontsize=7); ax.grid(True, lw=0.4, alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_convergence(csv_path, out_path):
    rows = read_csv(csv_path)
    if not rows:
        print(f"[skip] {csv_path.name} not found"); return
    data = defaultdict(lambda: {"N": [], "global": [], "smooth": []})
    for r in rows:
        name = r["solver"]
        data[name]["N"].append(int(r["N"]))
        data[name]["global"].append(float(r["L1_rho_global"]))
        data[name]["smooth"].append(float(r["L1_rho_smooth"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Convergence — Sod shock tube  |  t=0.2  |  WENO5-Z+HLLC+RK3\n"
                 "JaxFluids f64 vs Warp f32", fontsize=11, fontweight="bold")
    all_N = sorted({n for d in data.values() for n in d["N"]})
    ns = np.array(all_N, float)

    for ax, key, title in [
        (axes[0], "global", "Global L1(ρ)"),
        (axes[1], "smooth", "Smooth-region L1(ρ)  (rarefaction fan)"),
    ]:
        for name, d in data.items():
            if len(d["N"]) < 2: continue
            ns_d = np.array(d["N"]); es_d = np.array(d[key])
            sl = np.polyfit(np.log2(ns_d), np.log2(es_d), 1)[0]
            ax.plot(ns_d, es_d, marker=_style(name)[0], ls=_style(name)[1:],
                    color=_color(name), lw=1.8, ms=7, label=f"{name}  (slope {sl:+.2f})")
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_xlabel("N", fontsize=10); ax.set_ylabel("L1 error  (density)", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(all_N); ax.set_xticklabels(all_N, fontsize=8)
        ax.legend(fontsize=8); ax.grid(True, which="both", lw=0.4, alpha=0.5)

    # ratio panel
    ax = axes[2]
    names = list(data.keys())
    if len(names) >= 2:
        n1, n2 = names[0], names[1]
        Ns = [N for N in data[n1]["N"] if N in data[n2]["N"]]
        if len(Ns) >= 2:
            rg = [data[n1]["global"][data[n1]["N"].index(N)] /
                  data[n2]["global"][data[n2]["N"].index(N)] for N in Ns]
            ax.plot(Ns, rg, "D-", color="#e07b00", lw=1.8, ms=7,
                    label=f"{n1} / {n2}  (mean {np.mean(rg):.2f}×)")
    ax.axhline(1.0, color="black", ls="-", lw=0.6, alpha=0.3, label="parity")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N", fontsize=10); ax.set_ylabel("L1 ratio", fontsize=10)
    ax.set_title("Accuracy ratio", fontsize=10, fontweight="bold")
    ax.set_xticks(all_N); ax.set_xticklabels(all_N, fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, which="both", lw=0.4, alpha=0.5)

    plt.tight_layout()
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
        title="JaxFluids vs Warp  —  Sod throughput\nBoth: WENO5-Z+HLLC+RK3  |  JaxFluids f64  |  Warp f32")

    plot_profiles(
        OUT / "bench_jaxfluids_profiles_N512.csv",
        OUT / "jaxfluids_profiles.png")

    # scaling plot: merge Warp N-sweep + JaxFluids head-to-head data
    plot_throughput_merged(
        [OUT / "throughput_scaling.csv", OUT / "bench_jaxfluids_throughput.csv"],
        OUT / "sod_scaling.png",
        title="Sod  —  throughput vs grid size  (WENO5-Z+HLLC+RK3)")

    plot_memory(
        OUT / "memory_scaling.csv",
        OUT / "sod_memory.png",
        title="Sod  —  GPU memory vs grid size\n(XLA_PYTHON_CLIENT_PREALLOCATE=false)",
        theory_fn=theory)

    plot_convergence(
        OUT / "convergence.csv",
        OUT / "sod_convergence.png")


if __name__ == "__main__":
    main()
