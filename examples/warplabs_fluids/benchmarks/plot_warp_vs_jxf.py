"""
Warp CUDA graph vs JaxFluids — comparison plots for Sod and Shu-Osher.
Generates one figure per case: throughput panel + ratio panel.

Run from examples/warplabs_fluids/:
  python benchmarks/plot_warp_vs_jxf.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCH = Path(__file__).parent

CASES = [
    {
        "name":     "sod",
        "csv":      BENCH / "sod"       / "cuda_graph_benchmark.csv",
        "out":      BENCH / "sod"       / "warp_vs_jxf.png",
        "title":    "Sod shock tube",
    },
    {
        "name":     "shu_osher",
        "csv":      BENCH / "shu_osher" / "cuda_graph_benchmark.csv",
        "out":      BENCH / "shu_osher" / "warp_vs_jxf.png",
        "title":    "Shu-Osher shock-density",
    },
]


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    def col(key):
        return [float(r[key]) if r.get(key) else None for r in rows]
    ns = [int(r["N"]) for r in rows]
    return (
        ns,
        col("warp_f32_graph_Mcells"),
        col("warp_f64_graph_Mcells"),
        col("jxf_f32_fair_Mcells"),
        col("jxf_f64_fair_Mcells"),
    )


def paired(ns, warp, jxf):
    """Return (ns, warp, jxf, ratio) for rows where both are present."""
    n_v, w_v, j_v, r_v = [], [], [], []
    for n, w, j in zip(ns, warp, jxf):
        if w is not None and j is not None:
            n_v.append(n); w_v.append(w); j_v.append(j); r_v.append(w / j)
    return n_v, w_v, j_v, r_v


for case in CASES:
    ns, wf32, wf64, jf32, jf64 = load(case["csv"])

    n32, w32, j32, r32 = paired(ns, wf32, jf32)
    n64, w64, j64, r64 = paired(ns, wf64, jf64)

    fig, (ax_tp, ax_r) = plt.subplots(
        2, 1, figsize=(9, 7),
        gridspec_kw={"height_ratios": [3, 1.4]},
        sharex=False,
    )

    subtitle = (
        f"1-D Euler  ·  {case['title']}  ·  WENO5-Z+HLLC+RK3  ·  RTX 5000 Ada  ·  WSL2\n"
        "Warp: CUDA graph (fused kernel)  ·  JaxFluids: do_integration_step loop, single sync"
    )
    fig.suptitle(subtitle, fontsize=8, color="0.4", y=0.98)

    # ── Throughput panel ──────────────────────────────────────────────────────
    ax_tp.plot(n32, w32, "^-",  color="#009e73", lw=2, ms=7, label="Warp f32 (CUDA graph)")
    ax_tp.plot(n32, j32, "o--", color="#009e73", lw=2, ms=7, alpha=0.75,
               label="JaxFluids f32")
    ax_tp.plot(n64, w64, "s-",  color="#0072b2", lw=2, ms=7, label="Warp f64 (CUDA graph)")
    ax_tp.plot(n64, j64, "o--", color="#0072b2", lw=2, ms=7, alpha=0.75,
               label="JaxFluids f64")

    ax_tp.set_xscale("log", base=2)
    ax_tp.set_yscale("log", base=2)
    ax_tp.set_ylabel("Throughput  (Mcell-updates / s)", fontsize=11)
    ax_tp.set_title(f"{case['title']}  —  Warp CUDA Graph vs JaxFluids", fontsize=12)
    ax_tp.legend(fontsize=9)
    ax_tp.grid(True, which="both", lw=0.4, alpha=0.5)

    all_n = sorted(set(n32) | set(n64))
    ax_tp.set_xticks(all_n)
    ax_tp.set_xticklabels([f"{n:,}" for n in all_n], rotation=35, ha="right", fontsize=8)

    # ── Ratio panel ───────────────────────────────────────────────────────────
    ax_r.plot(n32, r32, "^-", color="#009e73", lw=2, ms=7, label="f32  Warp/JxF")
    ax_r.plot(n64, r64, "s-", color="#0072b2", lw=2, ms=7, label="f64  Warp/JxF")
    ax_r.axhline(1.0, color="0.5", lw=1, ls=":")

    ax_r.set_xscale("log", base=2)
    ax_r.set_ylabel("Warp / JaxFluids", fontsize=10)
    ax_r.set_xlabel("Number of cells", fontsize=11)
    ax_r.legend(fontsize=9)
    ax_r.grid(True, which="both", lw=0.4, alpha=0.5)
    ax_r.set_xticks(all_n)
    ax_r.set_xticklabels([f"{n:,}" for n in all_n], rotation=35, ha="right", fontsize=8)

    # annotate ratio values
    for n, r in zip(n32, r32):
        ax_r.annotate(f"{r:.1f}×", (n, r), textcoords="offset points",
                      xytext=(0, 6), ha="center", fontsize=7, color="#009e73")
    for n, r in zip(n64, r64):
        ax_r.annotate(f"{r:.1f}×", (n, r), textcoords="offset points",
                      xytext=(0, -12), ha="center", fontsize=7, color="#0072b2")

    fig.tight_layout()
    fig.savefig(case["out"], dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {case['out']}")
