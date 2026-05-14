"""
Convergence study — Sod shock tube.
Warp WENO5-Z+HLLC+RK3 (f32)  vs  JaxFluids WENO5-Z+HLLC+RK3 (f64)  vs exact Riemann.

Both solvers now share the same algorithm.  The accuracy gap measures the cost
of float32 vs float64, not the scheme order difference.

Saves (to benchmarks/sod/):
  sod_convergence.png
  convergence.csv

Run from projects/warplabs_fluids/:
  source /root/venv-jf/bin/activate   # or omit for Warp-only
  python benchmarks/sod/convergence_study.py
"""

import csv, json, sys, tempfile, gc
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

GRID_SIZES  = [64, 128, 256, 512, 1024]
GAMMA       = 1.4
T_END       = 0.2
CFL         = 0.4
X_SMOOTH_LO = 0.27
X_SMOOTH_HI = 0.47


def _l1_region(q_num, q_ref, x, dx, lo, hi):
    mask = (x >= lo) & (x <= hi)
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
    cp  = _patch_case(case_tmpl, N, run_dir)
    im  = InputManager(str(cp), str(num_path))
    buf = InitializationManager(im).initialization()
    SimulationManager(im).simulate(buf)
    h5s = sorted(glob.glob(str(run_dir / "sod" / "domain" / "data_*.h5")))
    h5  = max(h5s, key=lambda p: float(Path(p).stem.replace("data_", "")))
    with h5py.File(h5, "r") as f:
        return np.array(f["primitives/density"][0, 0, :])


def run_warp(N):
    import warp as wp
    Q0, _ = sod_ic(N, GAMMA)
    solver = WarpEuler1D(N, 1.0/N, gamma=GAMMA, bc="outflow", device="cuda", scheme="weno5z-rk3")
    solver.initialize(Q0); solver.run(T_END, CFL); wp.synchronize()
    rho, *_ = cons_to_prim(solver.state, GAMMA)
    del solver; gc.collect(); return rho


def fit_slope(ns, errs):
    return np.polyfit(np.log2(np.array(ns, float)), np.log2(np.array(errs, float)), 1)[0]


def main():
    import warp as wp
    wp.init()

    jxf_ok = False
    try:
        case_tmpl = json.load(open(JXF_EX / "sod.json"))
        num_setup = json.load(open(JXF_EX / "numerical_setup.json"))
        base_tmp  = Path(tempfile.mkdtemp(prefix="conv_sod_"))
        num_path  = base_tmp / "numerical_setup.json"
        num_path.write_text(json.dumps(num_setup))
        jxf_ok = True; print("[info] JaxFluids templates loaded")
    except Exception as e:
        print(f"[info] JaxFluids not available ({e})")

    SOLVERS = ["JaxFluids (WENO5-Z, f64)", "Warp WENO5-Z (f32)"]
    data = {s: {"N": [], "global": [], "smooth": []} for s in SOLVERS}

    for N in GRID_SIZES:
        dx = 1.0 / N
        _, x = sod_ic(N, GAMMA)
        rho_ex, *_ = sod_exact(T_END, x, GAMMA)
        print(f"\nN = {N}", flush=True)

        if jxf_ok:
            try:
                rho = run_jaxfluids(N, case_tmpl, num_path, base_tmp)
                g = l1_error(rho, rho_ex, dx)
                s = _l1_region(rho, rho_ex, x, dx, X_SMOOTH_LO, X_SMOOTH_HI)
                data["JaxFluids (WENO5-Z, f64)"]["N"].append(N)
                data["JaxFluids (WENO5-Z, f64)"]["global"].append(g)
                data["JaxFluids (WENO5-Z, f64)"]["smooth"].append(s)
                print(f"  JaxFluids  global={g:.3e}  smooth={s:.3e}")
            except Exception as e:
                print(f"  JaxFluids  ERROR: {e}")

        try:
            rho = run_warp(N)
            g = l1_error(rho, rho_ex, dx)
            s = _l1_region(rho, rho_ex, x, dx, X_SMOOTH_LO, X_SMOOTH_HI)
            data["Warp WENO5-Z (f32)"]["N"].append(N)
            data["Warp WENO5-Z (f32)"]["global"].append(g)
            data["Warp WENO5-Z (f32)"]["smooth"].append(s)
            print(f"  Warp       global={g:.3e}  smooth={s:.3e}")
        except Exception as e:
            print(f"  Warp       ERROR: {e}")

    if jxf_ok:
        import shutil; shutil.rmtree(base_tmp, ignore_errors=True)

    print("\n── Convergence slopes ──────────────────────────────")
    for name, d in data.items():
        if len(d["N"]) >= 2:
            print(f"  {name:<35}  global={fit_slope(d['N'],d['global']):+.2f}  "
                  f"smooth={fit_slope(d['N'],d['smooth']):+.2f}")

    # ── save CSV ──────────────────────────────────────────────────────────────
    with open(OUT / "convergence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solver", "N", "L1_rho_global", "L1_rho_smooth"])
        for name, d in data.items():
            for N, g, s in zip(d["N"], d["global"], d["smooth"]):
                w.writerow([name, N, g, s])
    print(f"\nSaved -> {OUT/'convergence.csv'}")

    # ── figure ────────────────────────────────────────────────────────────────
    COLORS  = {"JaxFluids (WENO5-Z, f64)": "#e07b00", "Warp WENO5-Z (f32)": "#009e73"}
    MARKERS = {"JaxFluids (WENO5-Z, f64)": "D",        "Warp WENO5-Z (f32)": "^"}
    LS      = {"JaxFluids (WENO5-Z, f64)": "-",         "Warp WENO5-Z (f32)": "-"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Convergence — Sod shock tube  |  t=0.2  |  WENO5-Z+HLLC+RK3\n"
        "JaxFluids f64 vs Warp f32  ·  slopes = least-squares on log₂–log₂",
        fontsize=11, fontweight="bold")
    ns = np.array(GRID_SIZES, float)

    for ax, key, title in [
        (axes[0], "global", "Global L1(ρ)  —  full domain"),
        (axes[1], "smooth", f"Smooth-region L1(ρ)  —  rarefaction fan\n(x ∈ [{X_SMOOTH_LO}, {X_SMOOTH_HI}])"),
    ]:
        for name, d in data.items():
            if len(d["N"]) < 2: continue
            ns_d = np.array(d["N"]); es_d = np.array(d[key])
            sl = fit_slope(ns_d, es_d)
            ax.plot(ns_d, es_d, marker=MARKERS[name], ls=LS[name],
                    color=COLORS[name], lw=1.8, ms=7,
                    label=f"{name}  (slope {sl:+.2f})")
        warp_d = data["Warp WENO5-Z (f32)"]
        if warp_d[key]:
            n_anchor = ns[2]; e_anch = np.interp(n_anchor, warp_d["N"], warp_d[key])
            ax.plot(ns, e_anch * (ns / n_anchor)**(-1.0), color="0.6", ls=":", lw=0.9, label="O(N⁻¹)")
            ax.plot(ns, e_anch * (ns / n_anchor)**(-3.0), color="0.8", ls="-.", lw=0.9, label="O(N⁻³)")
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_xlabel("N", fontsize=10); ax.set_ylabel("L1 error  (density)", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(GRID_SIZES); ax.set_xticklabels(GRID_SIZES, fontsize=8)
        ax.legend(fontsize=8, loc="upper right"); ax.grid(True, which="both", lw=0.4, alpha=0.5)

    # Panel 3: accuracy ratio (f32 vs f64 gap)
    ax = axes[2]
    jxf_d  = data["JaxFluids (WENO5-Z, f64)"]
    warp_d = data["Warp WENO5-Z (f32)"]
    Ns_common = [N for N in jxf_d["N"] if N in warp_d["N"]]
    if len(Ns_common) >= 2:
        rg = [jxf_d["global"][jxf_d["N"].index(N)] / warp_d["global"][warp_d["N"].index(N)]
              for N in Ns_common]
        rs = [jxf_d["smooth"][jxf_d["N"].index(N)] / warp_d["smooth"][warp_d["N"].index(N)]
              for N in Ns_common]
        ax.plot(Ns_common, rg, "D-",  color="#e07b00", lw=1.8, ms=7,
                label=f"Global L1  (mean {np.mean(rg):.2f}×)")
        ax.plot(Ns_common, rs, "s--", color="#5566cc", lw=1.8, ms=7,
                label=f"Smooth L1  (mean {np.mean(rs):.2f}×)")
        ax.axhline(np.mean(rg), color="#e07b00", ls=":", lw=1.0, alpha=0.6)
        ax.axhline(np.mean(rs), color="#5566cc", ls=":", lw=1.0, alpha=0.6)
    ax.axhline(1.0, color="black", ls="-", lw=0.6, alpha=0.3, label="parity (ratio=1)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N", fontsize=10)
    ax.set_ylabel("L1 ratio  (JaxFluids f64 / Warp f32)", fontsize=10)
    ax.set_title("Accuracy ratio  —  f64 vs f32 gap\n(same WENO5-Z algorithm)",
                 fontsize=10, fontweight="bold")
    ax.set_xticks(GRID_SIZES); ax.set_xticklabels(GRID_SIZES, fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, which="both", lw=0.4, alpha=0.5)

    plt.tight_layout()
    fig.savefig(OUT / "sod_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT/'sod_convergence.png'}")


if __name__ == "__main__":
    main()
