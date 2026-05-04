"""
Convergence study: JaxFluids vs Warp CUDA vs JAX CUDA — Sod shock tube.

Runs N = 64, 128, 256, 512, 1024 and measures L1 error vs exact Riemann.
Plots log-log convergence curves with least-squares slope (order of accuracy).

Run inside the Python 3.11 JaxFluids venv:
  source /root/venv-jf/bin/activate
  cd examples/warplabs_fluids
  python benchmarks/convergence_study.py
"""

import json, sys, tempfile, gc
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path(__file__).parent.parent
JXF_EX = Path("/root/JAXFLUIDS/examples/examples_1D/02_sod_shock_tube")

sys.path.insert(0, str(ROOT))

from warplabs_fluids import WarpEuler1D, cons_to_prim, l1_error
from cases.sod import ic as sod_ic, exact as sod_exact
from benchmarks.jax_euler import JaxEuler1D

GRID_SIZES = [64, 128, 256, 512, 1024]
GAMMA      = 1.4
T_END      = 0.2
CFL        = 0.4


# ── JaxFluids runner ──────────────────────────────────────────────────────────

def _patch_case(case_tmpl, N, run_dir):
    case = json.loads(json.dumps(case_tmpl))
    case["domain"]["x"]["cells"] = N
    case["general"]["save_path"] = str(run_dir)
    case["general"]["save_dt"]   = 999.0
    p = run_dir / "case.json"
    p.write_text(json.dumps(case))
    return p


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
        p_  = np.array(f["primitives/pressure"][0, 0, :])

    return rho, u, p_


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


# ── slope fit ─────────────────────────────────────────────────────────────────

def fit_slope(ns, errors):
    log_n = np.log2(ns)
    log_e = np.log2(errors)
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
        print(f"[info] JaxFluids templates loaded")
    except Exception as e:
        print(f"[warn] JaxFluids not available: {e}")
        jxf_ok = False

    base_tmp = Path(tempfile.mkdtemp(prefix="conv_study_"))
    if jxf_ok:
        num_path = base_tmp / "numerical_setup.json"
        num_path.write_text(json.dumps(ns_dict))

    solvers = {
        "JaxFluids\n(WENO5-Z+HLLC+RK3, f64)": {},
        "JAX CUDA\n(WENO3+HLLC+RK2, f32)":    {},
        "Warp CUDA\n(WENO3+HLLC+RK2, f32)":   {},
    }
    errs = {name: {"N": [], "rho": [], "u": [], "p": []} for name in solvers}

    for N in GRID_SIZES:
        dx = 1.0 / N
        _, x = sod_ic(N, GAMMA)
        rho_ex, u_ex, p_ex = sod_exact(T_END, x, GAMMA)
        print(f"\nN = {N}", flush=True)

        if jxf_ok:
            try:
                rho, u, p = run_jaxfluids(N, case_tmpl, num_path, base_tmp)
                name = "JaxFluids\n(WENO5-Z+HLLC+RK3, f64)"
                errs[name]["N"].append(N)
                errs[name]["rho"].append(l1_error(rho, rho_ex, dx))
                errs[name]["u"].append(l1_error(u,   u_ex,   dx))
                errs[name]["p"].append(l1_error(p,   p_ex,   dx))
                print(f"  JaxFluids  L1(rho)={errs[name]['rho'][-1]:.3e}", flush=True)
            except Exception as e:
                print(f"  JaxFluids  ERROR: {e}")

        try:
            rho, u, p = run_jax(N)
            name = "JAX CUDA\n(WENO3+HLLC+RK2, f32)"
            errs[name]["N"].append(N)
            errs[name]["rho"].append(l1_error(rho, rho_ex, dx))
            errs[name]["u"].append(l1_error(u,   u_ex,   dx))
            errs[name]["p"].append(l1_error(p,   p_ex,   dx))
            print(f"  JAX CUDA   L1(rho)={errs[name]['rho'][-1]:.3e}", flush=True)
        except Exception as e:
            print(f"  JAX CUDA   ERROR: {e}")

        try:
            rho, u, p = run_warp(N)
            name = "Warp CUDA\n(WENO3+HLLC+RK2, f32)"
            errs[name]["N"].append(N)
            errs[name]["rho"].append(l1_error(rho, rho_ex, dx))
            errs[name]["u"].append(l1_error(u,   u_ex,   dx))
            errs[name]["p"].append(l1_error(p,   p_ex,   dx))
            print(f"  Warp CUDA  L1(rho)={errs[name]['rho'][-1]:.3e}", flush=True)
        except Exception as e:
            print(f"  Warp CUDA  ERROR: {e}")

    # ── print table ───────────────────────────────────────────────────────────
    print("\n── L1(rho) convergence ──────────────────────────────────────────────")
    hdr = f"{'N':>6}" + "".join(f"  {n.split(chr(10))[0]:>28}" for n in solvers)
    print(hdr); print("-" * len(hdr))
    for N in GRID_SIZES:
        row = f"{N:>6}"
        for name, d in errs.items():
            if N in d["N"]:
                row += f"  {d['rho'][d['N'].index(N)]:>28.3e}"
            else:
                row += f"  {'--':>28}"
        print(row)

    print("\n── Measured convergence slopes (log2 L1 vs log2 N) ─────────────────")
    for name, d in errs.items():
        if len(d["N"]) >= 2:
            s_rho = fit_slope(d["N"], d["rho"])
            s_u   = fit_slope(d["N"], d["u"])
            s_p   = fit_slope(d["N"], d["p"])
            label = name.split("\n")[0]
            print(f"  {label:<28}  rho={s_rho:.2f}  u={s_u:.2f}  p={s_p:.2f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    colors = {
        "JaxFluids\n(WENO5-Z+HLLC+RK3, f64)": "#e07b00",
        "JAX CUDA\n(WENO3+HLLC+RK2, f32)":    "#d55e00",
        "Warp CUDA\n(WENO3+HLLC+RK2, f32)":   "#009e73",
    }
    markers = {
        "JaxFluids\n(WENO5-Z+HLLC+RK3, f64)": "D",
        "JAX CUDA\n(WENO3+HLLC+RK2, f32)":    "o",
        "Warp CUDA\n(WENO3+HLLC+RK2, f32)":   "^",
    }
    lines = {
        "JaxFluids\n(WENO5-Z+HLLC+RK3, f64)": "-",
        "JAX CUDA\n(WENO3+HLLC+RK2, f32)":    "--",
        "Warp CUDA\n(WENO3+HLLC+RK2, f32)":   "-",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Convergence study — Sod shock tube  |  t = 0.2\n"
        "Slopes measured by least-squares fit on log₂–log₂ grid",
        fontsize=11, fontweight="bold"
    )

    fields = ["rho", "u", "p"]
    titles = ["density  L1(ρ)", "velocity  L1(u)", "pressure  L1(p)"]

    for ax, field, title in zip(axes, fields, titles):
        for name, d in errs.items():
            if len(d["N"]) < 2:
                continue
            ns  = np.array(d["N"])
            es  = np.array(d[field])
            slope = fit_slope(ns, es)
            label = name.replace("\n", "  ") + f"  (slope {slope:.2f})"
            ax.plot(ns, es,
                    marker=markers[name], ls=lines[name],
                    color=colors[name], lw=1.8, ms=7, label=label)

        # reference lines
        n_ref = np.array([GRID_SIZES[0], GRID_SIZES[-1]], dtype=float)
        mid   = np.sqrt(n_ref[0] * n_ref[-1])
        # anchor at midpoint of Warp CUDA curve if available
        warp_d = errs["Warp CUDA\n(WENO3+HLLC+RK2, f32)"]
        if len(warp_d["N"]) >= 2:
            e_mid = np.interp(mid, warp_d["N"], warp_d[field])
            for order, ls, lbl in [(1, ":", "O(N⁻¹)"), (3, "-.", "O(N⁻³)")]:
                ref = e_mid * (n_ref / mid) ** (-order)
                ax.plot(n_ref, ref, ls=ls, color="gray", lw=1.0, label=lbl)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("N  (number of cells)", fontsize=10)
        ax.set_ylabel("L1 error", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(GRID_SIZES)
        ax.set_xticklabels([str(n) for n in GRID_SIZES], fontsize=8)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, which="both", lw=0.4, alpha=0.5)

    plt.tight_layout()
    out = ROOT / "benchmarks" / "convergence_study.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
