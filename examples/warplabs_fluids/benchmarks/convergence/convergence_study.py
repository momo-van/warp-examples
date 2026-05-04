"""
Convergence study: JaxFluids vs Warp CUDA vs JAX CUDA — Sod shock tube.

Two metrics at N = 64, 128, 256, 512, 1024:
  1. Global L1  — dominated by discontinuities, all solvers ~O(N^-1)
  2. Smooth-region L1 (rarefaction fan only, x ∈ [0.27, 0.47]) — reveals true
     scheme order: WENO5-Z (f64) vs WENO3 (f32), including float32 precision floor

Run inside the Python 3.11 JaxFluids venv:
  source /root/venv-jf/bin/activate
  cd examples/warplabs_fluids
  python benchmarks/convergence/convergence_study.py
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

GRID_SIZES = [64, 128, 256, 512, 1024]
GAMMA      = 1.4
T_END      = 0.2
CFL        = 0.4

X_SMOOTH_LO = 0.27   # just inside rarefaction head
X_SMOOTH_HI = 0.47   # just inside rarefaction tail


def l1_region(q_num, q_ref, x, dx, x_lo, x_hi):
    mask = (x >= x_lo) & (x <= x_hi)
    return float(np.sum(np.abs(q_num[mask] - q_ref[mask])) * dx)


# ── JaxFluids runner ──────────────────────────────────────────────────────────

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

    run_dir = base_tmp / f"jxf_N{N}"
    run_dir.mkdir(exist_ok=True)
    case_path = _patch_case(case_tmpl, N, run_dir)

    im   = InputManager(str(case_path), str(num_path))
    init = InitializationManager(im)
    sim  = SimulationManager(im)

    buf = init.initialization()
    sim.simulate(buf)   # JaxFluidsBuffers is a NamedTuple; final state in H5

    h5_files = sorted(glob.glob(str(run_dir / "sod" / "domain" / "data_*.h5")))
    h5_final = max(h5_files, key=lambda p: float(Path(p).stem.replace("data_", "")))
    with h5py.File(h5_final, "r") as f:
        rho = np.array(f["primitives/density"][0, 0, :])
        u   = np.array(f["primitives/velocity"][0, 0, :, 0])
        p   = np.array(f["primitives/pressure"][0, 0, :])
    return rho, u, p


# ── Warp CUDA runner ──────────────────────────────────────────────────────────

def run_warp(N):
    import warp as wp
    dx = 1.0 / N
    Q0, _ = sod_ic(N, GAMMA)
    solver = WarpEuler1D(N, dx, gamma=GAMMA, bc="outflow", device="cuda")
    solver.initialize(Q0)
    solver.run(T_END, CFL)
    wp.synchronize()
    rho, u, p = cons_to_prim(solver.state, GAMMA)
    del solver; gc.collect()
    return rho, u, p


# ── JAX CUDA runner ───────────────────────────────────────────────────────────

def run_jax(N):
    import jax
    dx = 1.0 / N
    Q0, _ = sod_ic(N, GAMMA)
    gpu = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]
    with jax.default_device(gpu):
        solver = JaxEuler1D(N, dx, gamma=GAMMA, bc="outflow")
        solver.initialize(Q0)
        solver.run(T_END, CFL)
        jax.block_until_ready(solver._Q)
        Q = np.asarray(solver._Q)
    rho, u, p = cons_to_prim(Q, GAMMA)
    del solver; gc.collect()
    return rho, u, p


def fit_slope(ns, errors):
    log_n = np.log2(np.array(ns, float))
    log_e = np.log2(np.array(errors, float))
    slope, _ = np.polyfit(log_n, log_e, 1)
    return slope


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import warp as wp
    wp.init()

    try:
        with open(JXF_EX / "sod.json")             as f: case_tmpl = json.load(f)
        with open(JXF_EX / "numerical_setup.json")  as f: ns_dict   = json.load(f)
        jxf_ok = True
        print("[info] JaxFluids templates loaded")
    except Exception as e:
        print(f"[warn] JaxFluids not available: {e}")
        jxf_ok = False

    base_tmp = Path(tempfile.mkdtemp(prefix="conv_study_"))
    if jxf_ok:
        num_path = base_tmp / "numerical_setup.json"
        num_path.write_text(json.dumps(ns_dict))

    SOLVERS = [
        "JaxFluids\n(WENO5-Z+HLLC+RK3, f64)",
        "JAX CUDA\n(WENO3+HLLC+RK2, f32)",
        "Warp CUDA\n(WENO3+HLLC+RK2, f32)",
    ]
    data = {s: {"N": [], "global": [], "smooth": []} for s in SOLVERS}

    for N in GRID_SIZES:
        dx = 1.0 / N
        _, x = sod_ic(N, GAMMA)
        rho_ex, u_ex, p_ex = sod_exact(T_END, x, GAMMA)
        print(f"\nN = {N}", flush=True)

        runs = []

        if jxf_ok:
            try:
                rho, u, p = run_jaxfluids(N, case_tmpl, num_path, base_tmp)
                runs.append(("JaxFluids\n(WENO5-Z+HLLC+RK3, f64)", rho, u, p))
                print(f"  JaxFluids  done", flush=True)
            except Exception as e:
                print(f"  JaxFluids  ERROR: {e}")

        try:
            rho, u, p = run_jax(N)
            runs.append(("JAX CUDA\n(WENO3+HLLC+RK2, f32)", rho, u, p))
            print(f"  JAX CUDA   done", flush=True)
        except Exception as e:
            print(f"  JAX CUDA   ERROR: {e}")

        try:
            rho, u, p = run_warp(N)
            runs.append(("Warp CUDA\n(WENO3+HLLC+RK2, f32)", rho, u, p))
            print(f"  Warp CUDA  done", flush=True)
        except Exception as e:
            print(f"  Warp CUDA  ERROR: {e}")

        for name, rho, u, p in runs:
            g = l1_error(rho, rho_ex, dx)
            s = l1_region(rho, rho_ex, x, dx, X_SMOOTH_LO, X_SMOOTH_HI)
            data[name]["N"].append(N)
            data[name]["global"].append(g)
            data[name]["smooth"].append(s)
            print(f"    {name.split(chr(10))[0]:<28}  global={g:.3e}  smooth={s:.3e}")

    for metric, key in [("Global L1(rho)", "global"), ("Smooth-region L1(rho)", "smooth")]:
        print(f"\n── {metric} ──────────────────────────────────")
        print(f"{'N':>6}", end="")
        for s in SOLVERS:
            print(f"  {s.split(chr(10))[0]:>28}", end="")
        print()
        print("-" * 100)
        for N in GRID_SIZES:
            print(f"{N:>6}", end="")
            for s in SOLVERS:
                d = data[s]
                if N in d["N"]:
                    print(f"  {d[key][d['N'].index(N)]:>28.3e}", end="")
                else:
                    print(f"  {'--':>28}", end="")
            print()

    print("\n── Convergence slopes ──────────────────────────────────────────────")
    for name, d in data.items():
        if len(d["N"]) >= 2:
            sg = fit_slope(d["N"], d["global"])
            ss = fit_slope(d["N"], d["smooth"])
            print(f"  {name.split(chr(10))[0]:<28}  global slope={sg:.2f}  smooth slope={ss:.2f}")

    colors  = {
        "JaxFluids\n(WENO5-Z+HLLC+RK3, f64)": "#e07b00",
        "JAX CUDA\n(WENO3+HLLC+RK2, f32)":    "#d55e00",
        "Warp CUDA\n(WENO3+HLLC+RK2, f32)":   "#009e73",
    }
    markers = {
        "JaxFluids\n(WENO5-Z+HLLC+RK3, f64)": "D",
        "JAX CUDA\n(WENO3+HLLC+RK2, f32)":    "o",
        "Warp CUDA\n(WENO3+HLLC+RK2, f32)":   "^",
    }
    ls_map  = {
        "JaxFluids\n(WENO5-Z+HLLC+RK3, f64)": "-",
        "JAX CUDA\n(WENO3+HLLC+RK2, f32)":    "--",
        "Warp CUDA\n(WENO3+HLLC+RK2, f32)":   "-",
    }

    fig, axes = plt.subplots(2, 1, figsize=(9, 10))

    row_cfg = [
        ("global", "Global L1(ρ) — full domain",
         "All solvers converge ~O(N⁻¹): discontinuities cap convergence rate regardless of scheme order.",
         [(-1, ":", "O(N⁻¹)"), (-3, "-.", "O(N⁻³)")]),
        ("smooth", f"Smooth-region L1(ρ) — rarefaction fan only  (x ∈ [{X_SMOOTH_LO}, {X_SMOOTH_HI}])",
         "Still ~O(N⁻¹): fan head/tail are C⁰ characteristics — smooth region bounded by non-smooth features.",
         [(-1, ":", "O(N⁻¹)"), (-3, "-.", "O(N⁻³)"), (-5, "--", "O(N⁻⁵)")]),
    ]

    for ax, (key, title, subtitle, refs) in zip(axes, row_cfg):
        n_ref = np.array([GRID_SIZES[0], GRID_SIZES[-1]], dtype=float)
        mid   = np.sqrt(n_ref[0] * n_ref[-1])

        for name, d in data.items():
            if len(d["N"]) < 2:
                continue
            ns  = np.array(d["N"])
            es  = np.array(d[key])
            sl  = fit_slope(ns, es)
            lbl = name.replace("\n", "  ") + f"  (slope {sl:+.2f})"
            ax.plot(ns, es, marker=markers[name], ls=ls_map[name],
                    color=colors[name], lw=1.8, ms=7, label=lbl)

        warp_key = "Warp CUDA\n(WENO3+HLLC+RK2, f32)"
        if data[warp_key][key]:
            e_mid = np.interp(mid, data[warp_key]["N"], data[warp_key][key])
            for order, lsty, lbl in refs:
                ref = e_mid * (n_ref / mid) ** order
                ax.plot(n_ref, ref, ls=lsty, color="gray", lw=1.0, label=lbl)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("N  (number of cells)", fontsize=10)
        ax.set_ylabel("L1 error  (density)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(0.02, 0.04, subtitle, transform=ax.transAxes,
                fontsize=8, color="#444", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        ax.set_xticks(GRID_SIZES)
        ax.set_xticklabels([str(n) for n in GRID_SIZES], fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", lw=0.4, alpha=0.5)

    fig.suptitle(
        "Convergence study — Sod shock tube  |  t = 0.2\n"
        "Least-squares slopes on log₂–log₂ grid",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out = OUT / "convergence_study.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
