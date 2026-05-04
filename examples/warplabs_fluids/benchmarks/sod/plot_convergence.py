"""
Convergence study — plot from hardcoded benchmark results.
Saves sod_convergence.png (3-panel figure).

Run from examples/warplabs_fluids/ (no GPU needed):
  python benchmarks/sod/plot_convergence.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent

NS = [64, 128, 256, 512, 1024]

GLOBAL_L1 = {
    "JaxFluids\n(WENO5-Z+HLLC+RK3, f64)": [6.184e-3, 3.140e-3, 1.563e-3, 8.519e-4, 4.380e-4],
    "JAX CUDA\n(WENO3+HLLC+RK2, f32)":    [1.082e-2, 5.578e-3, 3.009e-3, 1.729e-3, 9.259e-4],
    "Warp CUDA\n(WENO3+HLLC+RK2, f32)":   [1.082e-2, 5.578e-3, 3.009e-3, 1.729e-3, 9.259e-4],
}

SMOOTH_L1 = {
    "JaxFluids\n(WENO5-Z+HLLC+RK3, f64)": [1.487e-3, 6.855e-4, 3.611e-4, 1.878e-4, 1.006e-4],
    "JAX CUDA\n(WENO3+HLLC+RK2, f32)":    [2.762e-3, 1.282e-3, 6.990e-4, 3.627e-4, 1.825e-4],
    "Warp CUDA\n(WENO3+HLLC+RK2, f32)":   [2.762e-3, 1.282e-3, 6.990e-4, 3.627e-4, 1.825e-4],
}

COLORS = {
    "JaxFluids\n(WENO5-Z+HLLC+RK3, f64)": "#e07b00",
    "JAX CUDA\n(WENO3+HLLC+RK2, f32)":    "#d55e00",
    "Warp CUDA\n(WENO3+HLLC+RK2, f32)":   "#009e73",
}
MARKERS = {
    "JaxFluids\n(WENO5-Z+HLLC+RK3, f64)": "D",
    "JAX CUDA\n(WENO3+HLLC+RK2, f32)":    "o",
    "Warp CUDA\n(WENO3+HLLC+RK2, f32)":   "^",
}
LS = {
    "JaxFluids\n(WENO5-Z+HLLC+RK3, f64)": "-",
    "JAX CUDA\n(WENO3+HLLC+RK2, f32)":    "--",
    "Warp CUDA\n(WENO3+HLLC+RK2, f32)":   "-",
}


def fit_slope(ns, errors):
    return np.polyfit(np.log2(ns), np.log2(errors), 1)[0]


def ref_line(ax, ns, e_anchor, n_anchor, order, color="gray", ls=":", label=None):
    n_arr = np.array([ns[0], ns[-1]], float)
    ax.plot(n_arr, e_anchor * (n_arr / n_anchor) ** order, ls=ls, color=color, lw=0.9, label=label)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Convergence study — Sod shock tube  |  t = 0.2  |  slopes = least-squares fit on log₂–log₂",
    fontsize=11, fontweight="bold")

ns = np.array(NS, float)

# Panel 1: Global L1
ax = axes[0]
for name, errs in GLOBAL_L1.items():
    sl = fit_slope(ns, errs)
    ax.plot(ns, errs, marker=MARKERS[name], ls=LS[name], color=COLORS[name], lw=1.8, ms=7,
            label=name.replace("\n", "  ") + f"  (slope {sl:+.2f})")
warp_g = np.array(GLOBAL_L1["Warp CUDA\n(WENO3+HLLC+RK2, f32)"])
ref_line(ax, ns, warp_g[2], ns[2], -1, label="O(N⁻¹)")
ref_line(ax, ns, warp_g[0]*0.3, ns[0], -3, ls="-.", label="O(N⁻³)")
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xlabel("N", fontsize=10); ax.set_ylabel("L1 error  (density)", fontsize=10)
ax.set_title("Global L1(ρ)  —  full domain", fontsize=10, fontweight="bold")
ax.text(0.04, 0.05, "All solvers ~O(N⁻¹): shocks and contact\ndiscontinuity cap convergence rate.",
    transform=ax.transAxes, fontsize=8, color="#444", va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax.set_xticks(NS); ax.set_xticklabels(NS, fontsize=8)
ax.legend(fontsize=7.5, loc="upper right"); ax.grid(True, which="both", lw=0.4, alpha=0.5)

# Panel 2: Smooth-region L1
ax = axes[1]
for name, errs in SMOOTH_L1.items():
    sl = fit_slope(ns, errs)
    ax.plot(ns, errs, marker=MARKERS[name], ls=LS[name], color=COLORS[name], lw=1.8, ms=7,
            label=name.replace("\n", "  ") + f"  (slope {sl:+.2f})")
warp_s = np.array(SMOOTH_L1["Warp CUDA\n(WENO3+HLLC+RK2, f32)"])
ref_line(ax, ns, warp_s[2], ns[2], -1,  label="O(N⁻¹)")
ref_line(ax, ns, warp_s[0]*0.3, ns[0], -3, ls="-.", label="O(N⁻³)")
ref_line(ax, ns, warp_s[0]*0.05, ns[0], -5, ls="--", label="O(N⁻⁵)")
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xlabel("N", fontsize=10); ax.set_ylabel("L1 error  (density)", fontsize=10)
ax.set_title("Smooth-region L1(ρ)  —  rarefaction fan\n(x ∈ [0.27, 0.47])", fontsize=10, fontweight="bold")
ax.text(0.04, 0.05, "Still ~O(N⁻¹): fan head/tail are C⁰\ncharacteristics — smooth region bounded\nby non-smooth features.",
    transform=ax.transAxes, fontsize=8, color="#444", va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax.set_xticks(NS); ax.set_xticklabels(NS, fontsize=8)
ax.legend(fontsize=7.5, loc="upper right"); ax.grid(True, which="both", lw=0.4, alpha=0.5)

# Panel 3: Accuracy ratio
ax = axes[2]
jxf_g  = np.array(GLOBAL_L1["JaxFluids\n(WENO5-Z+HLLC+RK3, f64)"])
jxf_s  = np.array(SMOOTH_L1["JaxFluids\n(WENO5-Z+HLLC+RK3, f64)"])
warp_g = np.array(GLOBAL_L1["Warp CUDA\n(WENO3+HLLC+RK2, f32)"])
warp_s = np.array(SMOOTH_L1["Warp CUDA\n(WENO3+HLLC+RK2, f32)"])
ratio_g, ratio_s = jxf_g / warp_g, jxf_s / warp_s
ax.plot(ns, ratio_g, marker="D", ls="-",  color="#e07b00", lw=1.8, ms=7,
        label=f"Global L1  (mean {ratio_g.mean():.2f}×)")
ax.plot(ns, ratio_s, marker="s", ls="--", color="#5566cc", lw=1.8, ms=7,
        label=f"Smooth L1  (mean {ratio_s.mean():.2f}×)")
ax.axhline(ratio_g.mean(), color="#e07b00", ls=":", lw=1.0, alpha=0.6)
ax.axhline(ratio_s.mean(), color="#5566cc", ls=":", lw=1.0, alpha=0.6)
ax.axhline(1.0, color="black", ls="-", lw=0.6, alpha=0.3, label="parity  (ratio = 1)")
ax.set_xscale("log", base=2); ax.set_ylim(0, 1.0)
ax.set_xlabel("N", fontsize=10); ax.set_ylabel("L1 ratio  (JaxFluids / Warp CUDA)", fontsize=10)
ax.set_title("Accuracy ratio  —  JaxFluids vs Warp CUDA\n(lower = JaxFluids more accurate)", fontsize=10, fontweight="bold")
ax.text(0.04, 0.77,
    "Ratio is FLAT across all N:\nJaxFluids ~2× more accurate,\ngap neither grows nor shrinks.\nWarp CUDA achieves same accuracy\nat 1 resolution doubling lower cost.",
    transform=ax.transAxes, fontsize=8, color="#222", va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax.set_xticks(NS); ax.set_xticklabels(NS, fontsize=8)
ax.legend(fontsize=8, loc="lower left"); ax.grid(True, which="both", lw=0.4, alpha=0.5)

plt.tight_layout()
out = OUT / "sod_convergence.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")
