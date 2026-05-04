"""
Convergence study: JaxFluids vs Warp CUDA vs JAX CUDA — Sod shock tube.

Runs live at N = 64, 128, 256, 512, 1024 and saves sod_convergence.png.
Requires JaxFluids venv on WSL2 — use plot_convergence.py for a Windows-safe
version that plots from hardcoded data.

Run from examples/warplabs_fluids/:
  source /root/venv-jf/bin/activate
  python benchmarks/sod/convergence_study.py
"""

import json, sys, tempfile, gc
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path(__file__).parent.parent.parent
OUT    = Path(__file__).parent
JXF_EX = Path("/root/JAXFLUIDS/examples/examples_1D/02_sod_shock_tube")

sys.path.insert(0, str(ROOT))

from warplabs_fluids import WarpEuler1D, cons_to_prim, l1_error
from cases.sod import ic as sod_ic, exact as sod_exact
from benchmarks.jax_euler import JaxEuler1D

GRID_SIZES  = [64, 128, 256, 512, 1024]
GAMMA       = 1.4
T_END       = 0.2
CFL         = 0.4
X_SMOOTH_LO = 0.27
X_SMOOTH_HI = 0.47

COLORS  = {
    "JaxFluids\n(WENO5-Z+HLLC+RK3, f64)": "#e07b00",
    "JAX CUDA\n(WENO3+HLLC+RK2, f32)":    "#d55e00",
    "Warp CUDA\n(WENO3+HLLC+RK2, f32)":   "#009e73",
}
MARKERS = {"JaxFluids\n(WENO5-Z+HLLC+RK3, f64)": "D",
           "JAX CUDA\n(WENO3+HLLC+RK2, f32)":    "o",
           "Warp CUDA\n(WENO3+HLLC+RK2, f32)":   "^"}
LS      = {"JaxFluids\n(WENO5-Z+HLLC+RK3, f64)": "-",
           "JAX CUDA\n(WENO3+HLLC+RK2, f32)":    "--",
           "Warp CUDA\n(WENO3+HLLC+RK2, f32)":   "-"}


def l1_region(q_num, q_ref, x, dx, x_lo, x_hi):
    mask = (x >= x_lo) & (x <= x_hi)
    return float(np.sum(np.abs(q_num[mask] - q_ref[mask])) * dx)


def _patch_case(case_tmpl, N, run_dir):
    case = json.loads(json.dumps(case_tmpl))
    case["domain"]["x"]["cells"] = N
    case["general"]["save_path"] = str(run_dir)
    case["general"]["save_dt"]   = 999.0
    (run_dir / "case.json").write_text(json.dumps(case))
    return run_dir / "case.json"


def run_jaxfluids(N, case_tmpl, num_path, base_tmp):
    from jaxfluids import InputManager, InitializationManager, SimulationManager
    import glob, h5py
    run_dir = base_tmp / f"jxf_N{N}"; run_dir.mkdir(exist_ok=True)
    case_path = _patch_case(case_tmpl, N, run_dir)
    im  = InputManager(str(case_path), str(num_path))
    buf = InitializationManager(im).initialization()
    SimulationManager(im).simulate(buf)
    h5_files = sorted(glob.glob(str(run_dir / "sod" / "domain" / "data_*.h5")))
    h5_final = max(h5_files, key=lambda p: float(Path(p).stem.replace("data_", "")))
    with h5py.File(h5_final, "r") as f:
        return np.array(f["primitives/density"][0, 0, :])


def run_warp(N):
    import warp as wp
    dx = 1.0 / N; Q0, _ = sod_ic(N, GAMMA)
    solver = WarpEuler1D(N, dx, gamma=GAMMA, bc="outflow", device="cuda")
    solver.initialize(Q0); solver.run(T_END, CFL); wp.synchronize()
    rho, *_ = cons_to_prim(solver.state, GAMMA)
    del solver; gc.collect(); return rho


def run_jax(N):
    import jax
    dx = 1.0 / N; Q0, _ = sod_ic(N, GAMMA)
    gpu = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]
    with jax.default_device(gpu):
        solver = JaxEuler1D(N, dx, gamma=GAMMA, bc="outflow")
        solver.initialize(Q0); solver.run(T_END, CFL); jax.block_until_ready(solver._Q)
        Q = np.asarray(solver._Q)
    rho, *_ = cons_to_prim(Q, GAMMA)
    del solver; gc.collect(); return rho


def fit_slope(ns, errs):
    return np.polyfit(np.log2(np.array(ns, float)), np.log2(np.array(errs, float)), 1)[0]


def ref_line(ax, ns, e_anchor, n_anchor, order, color="gray", ls=":", label=None):
    n_arr = np.array([ns[0], ns[-1]], float)
    ax.plot(n_arr, e_anchor * (n_arr / n_anchor) ** order, ls=ls, color=color, lw=0.9, label=label)


def main():
    import warp as wp
    wp.init()

    try:
        with open(JXF_EX / "sod.json") as f: case_tmpl = json.load(f)
        with open(JXF_EX / "numerical_setup.json") as f: ns_dict = json.load(f)
        jxf_ok = True; print("[info] JaxFluids templates loaded")
    except Exception as e:
        print(f"[warn] JaxFluids not available: {e}"); jxf_ok = False

    base_tmp = Path(tempfile.mkdtemp(prefix="conv_"))
    if jxf_ok:
        (base_tmp / "numerical_setup.json").write_text(json.dumps(ns_dict))
        num_path = base_tmp / "numerical_setup.json"

    SOLVERS = list(COLORS.keys())
    data = {s: {"N": [], "global": [], "smooth": []} for s in SOLVERS}

    for N in GRID_SIZES:
        dx = 1.0 / N
        _, x = sod_ic(N, GAMMA)
        rho_ex, *_ = sod_exact(T_END, x, GAMMA)
        print(f"\nN = {N}", flush=True)

        runners = []
        if jxf_ok:
            try:
                rho = run_jaxfluids(N, case_tmpl, num_path, base_tmp)
                runners.append(("JaxFluids\n(WENO5-Z+HLLC+RK3, f64)", rho))
                print("  JaxFluids done", flush=True)
            except Exception as e: print(f"  JaxFluids ERROR: {e}")
        try:
            runners.append(("JAX CUDA\n(WENO3+HLLC+RK2, f32)", run_jax(N)))
            print("  JAX CUDA  done", flush=True)
        except Exception as e: print(f"  JAX CUDA ERROR: {e}")
        try:
            runners.append(("Warp CUDA\n(WENO3+HLLC+RK2, f32)", run_warp(N)))
            print("  Warp CUDA done", flush=True)
        except Exception as e: print(f"  Warp CUDA ERROR: {e}")

        for name, rho in runners:
            g = l1_error(rho, rho_ex, dx)
            s = l1_region(rho, rho_ex, x, dx, X_SMOOTH_LO, X_SMOOTH_HI)
            data[name]["N"].append(N)
            data[name]["global"].append(g)
            data[name]["smooth"].append(s)
            print(f"    {name.split(chr(10))[0]:<28}  global={g:.3e}  smooth={s:.3e}")

    print("\n── Convergence slopes ──────────────────────────────")
    for name, d in data.items():
        if len(d["N"]) >= 2:
            print(f"  {name.split(chr(10))[0]:<28}  global={fit_slope(d['N'],d['global']):+.2f}  "
                  f"smooth={fit_slope(d['N'],d['smooth']):+.2f}")

    # ── 3-panel convergence figure ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Convergence study — Sod shock tube  |  t = 0.2  |  slopes = least-squares fit on log₂–log₂",
        fontsize=11, fontweight="bold")
    ns = np.array(GRID_SIZES, float)

    for ax, key, title, note, refs in [
        (axes[0], "global", "Global L1(ρ)  —  full domain",
         "All solvers ~O(N⁻¹): shocks and contact\ndiscontinuity cap convergence rate.",
         [(-1,":",ns[2],"O(N⁻¹)"),(-3,"-.",ns[0],"O(N⁻³)")]),
        (axes[1], "smooth", f"Smooth-region L1(ρ)  —  rarefaction fan\n(x ∈ [{X_SMOOTH_LO}, {X_SMOOTH_HI}])",
         "Still ~O(N⁻¹): fan head/tail are C⁰\ncharacteristics.",
         [(-1,":",ns[2],"O(N⁻¹)"),(-3,"-.",ns[0],"O(N⁻³)"),(-5,"--",ns[0],"O(N⁻⁵)")]),
    ]:
        for name, d in data.items():
            if len(d["N"]) < 2: continue
            ns_d = np.array(d["N"]); es_d = np.array(d[key])
            sl = fit_slope(ns_d, es_d)
            ax.plot(ns_d, es_d, marker=MARKERS[name], ls=LS[name],
                    color=COLORS[name], lw=1.8, ms=7,
                    label=name.replace("\n","  ")+f"  (slope {sl:+.2f})")
        warp_d = data["Warp CUDA\n(WENO3+HLLC+RK2, f32)"]
        if warp_d[key]:
            for order, lsty, n_anchor, lbl in refs:
                e_anch = np.interp(n_anchor, warp_d["N"], warp_d[key])
                ref_line(ax, ns, e_anch, n_anchor, order, ls=lsty, label=lbl)
        ax.set_xscale("log",base=2); ax.set_yscale("log",base=2)
        ax.set_xlabel("N",fontsize=10); ax.set_ylabel("L1 error  (density)",fontsize=10)
        ax.set_title(title,fontsize=10,fontweight="bold")
        ax.text(0.04,0.05,note,transform=ax.transAxes,fontsize=8,color="#444",va="bottom",
                bbox=dict(boxstyle="round,pad=0.3",fc="white",alpha=0.8))
        ax.set_xticks(GRID_SIZES); ax.set_xticklabels(GRID_SIZES,fontsize=8)
        ax.legend(fontsize=7.5,loc="upper right"); ax.grid(True,which="both",lw=0.4,alpha=0.5)

    # Panel 3: accuracy ratio
    ax = axes[2]
    jxf_g  = np.array(data["JaxFluids\n(WENO5-Z+HLLC+RK3, f64)"]["global"])
    jxf_s  = np.array(data["JaxFluids\n(WENO5-Z+HLLC+RK3, f64)"]["smooth"])
    warp_g = np.array(data["Warp CUDA\n(WENO3+HLLC+RK2, f32)"]["global"])
    warp_s = np.array(data["Warp CUDA\n(WENO3+HLLC+RK2, f32)"]["smooth"])
    if len(jxf_g) and len(warp_g):
        rg = jxf_g / warp_g; rs = jxf_s / warp_s
        ax.plot(ns[:len(rg)], rg, marker="D", ls="-",  color="#e07b00", lw=1.8, ms=7,
                label=f"Global L1  (mean {rg.mean():.2f}×)")
        ax.plot(ns[:len(rs)], rs, marker="s", ls="--", color="#5566cc", lw=1.8, ms=7,
                label=f"Smooth L1  (mean {rs.mean():.2f}×)")
        ax.axhline(rg.mean(),color="#e07b00",ls=":",lw=1.0,alpha=0.6)
        ax.axhline(rs.mean(),color="#5566cc",ls=":",lw=1.0,alpha=0.6)
    ax.axhline(1.0, color="black", ls="-", lw=0.6, alpha=0.3, label="parity  (ratio = 1)")
    ax.set_xscale("log",base=2); ax.set_ylim(0,1.0)
    ax.set_xlabel("N",fontsize=10); ax.set_ylabel("L1 ratio  (JaxFluids / Warp CUDA)",fontsize=10)
    ax.set_title("Accuracy ratio  —  JaxFluids vs Warp CUDA\n(lower = JaxFluids more accurate)",
                 fontsize=10,fontweight="bold")
    ax.set_xticks(GRID_SIZES); ax.set_xticklabels(GRID_SIZES,fontsize=8)
    ax.legend(fontsize=8,loc="lower left"); ax.grid(True,which="both",lw=0.4,alpha=0.5)

    plt.tight_layout()
    out = OUT / "sod_convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
