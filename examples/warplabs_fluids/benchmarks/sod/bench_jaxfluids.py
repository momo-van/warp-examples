"""
JaxFluids vs Warp CUDA — Sod shock tube comparison.

Runs the Sod problem with:
  - JaxFluids  (WENO5-Z + HLLC + RK3, double precision — production solver)
  - Our JAX    (WENO3  + HLLC + RK2, float32 — hand-written reference)
  - Warp CUDA  (WENO3  + HLLC + RK2, float32 — fused kernel)

Measures throughput (Mcell/s) and L1 accuracy vs exact Riemann solution.

Run from examples/warplabs_fluids/ inside the Python 3.11 venv:
  source /root/venv-jf/bin/activate
  python benchmarks/sod/bench_jaxfluids.py
"""

import json, sys, time, tempfile, shutil, os, gc
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).parent.parent.parent
OUT     = Path(__file__).parent
JXF_EX  = Path("/root/JAXFLUIDS/examples/examples_1D/02_sod_shock_tube")

sys.path.insert(0, str(ROOT))

from warplabs_fluids import WarpEuler1D, cons_to_prim, l1_error
from cases.sod import ic as sod_ic, exact as sod_exact
from benchmarks.jax_euler import JaxEuler1D

# ── config ────────────────────────────────────────────────────────────────────
GRID_SIZES = [256, 512, 1024, 2048, 4096]
GAMMA      = 1.4
T_END      = 0.2
CFL_WARP   = 0.4
N_BENCH    = 1      # timed repetitions (JaxFluids is slow; 1 run each)
N_WARM     = 1

# ── JaxFluids helpers ─────────────────────────────────────────────────────────

def _load_jxf_templates():
    """Load and return the example JSON files as dicts."""
    with open(JXF_EX / "sod.json")              as f: case     = json.load(f)
    with open(JXF_EX / "numerical_setup.json")  as f: num_setup = json.load(f)
    return case, num_setup


def _patch_case(case_tmpl, N, tmp_dir):
    """Return (case_path, num_path) for a fresh copy patched to N cells."""
    case = json.loads(json.dumps(case_tmpl))   # deep copy

    # set N, disable file output (we don't need the HDF5 dumps)
    case["domain"]["x"]["cells"] = N
    case["general"]["save_path"] = str(tmp_dir)
    case["general"]["save_dt"]   = 999.0   # no intermediate saves

    case_path = tmp_dir / "case.json"
    with open(case_path, "w") as f:
        json.dump(case, f)
    return case_path


def bench_jaxfluids(N, case_tmpl, num_path, tmp_dir):
    """Run JaxFluids Sod at N cells, return (mcell_s, rho, u, p, n_steps)."""
    from jaxfluids import InputManager, InitializationManager, SimulationManager
    import jax

    # per-N save directory to avoid cross-contamination of H5 files
    run_dir = tmp_dir / f"run_N{N}"
    run_dir.mkdir(exist_ok=True)
    case_path = _patch_case(case_tmpl, N, run_dir)

    input_mgr = InputManager(str(case_path), str(num_path))
    init_mgr  = InitializationManager(input_mgr)
    sim_mgr   = SimulationManager(input_mgr)

    buffers = init_mgr.initialization()

    # warmup (triggers XLA compilation)
    t0 = time.perf_counter()
    sim_mgr.simulate(buffers)
    jax.block_until_ready(buffers)
    t_warm = time.perf_counter() - t0
    print(f"    JaxFluids warmup: {t_warm:.2f}s", flush=True)

    # timed runs — re-initialize each time
    times = []
    for _ in range(N_BENCH):
        buffers = init_mgr.initialization()
        t0 = time.perf_counter()
        sim_mgr.simulate(buffers)
        jax.block_until_ready(buffers)
        times.append(time.perf_counter() - t0)

    t_med = float(np.median(times))

    # estimate steps from CFL + dt
    dx     = 1.0 / N
    a_max  = np.sqrt(GAMMA * 1.0 / 1.0)
    try:
        with open(num_path) as f: ns = json.load(f)
        cfl_jxf = ns["conservatives"]["time_integration"].get("CFL", 0.5)
    except Exception:
        cfl_jxf = 0.5
    dt_est  = cfl_jxf * dx / a_max
    n_steps = max(1, int(round(T_END / dt_est)))

    mcell_s = N * n_steps / t_med / 1e6

    # read final state from HDF5 output (buffers NamedTuple is not mutated by simulate)
    rho = u = p_ = None
    try:
        import glob, h5py
        case_name = "sod"
        h5_pattern = str(run_dir / case_name / "domain" / "data_*.h5")
        h5_files = sorted(glob.glob(h5_pattern))
        h5_final = max(h5_files, key=lambda p: float(
            Path(p).stem.replace("data_", "")))
        with h5py.File(h5_final, "r") as f:
            rho = np.array(f["primitives/density"][0, 0, :])
            u   = np.array(f["primitives/velocity"][0, 0, :, 0])
            p_  = np.array(f["primitives/pressure"][0, 0, :])
    except Exception as e:
        print(f"    [warn] could not read JaxFluids HDF5 output: {e}")

    return mcell_s, rho, u, p_, n_steps, t_med


# ── our solvers ───────────────────────────────────────────────────────────────

def bench_warp(N):
    import warp as wp
    dx  = 1.0 / N
    Q0, x = sod_ic(N, GAMMA)
    solver = WarpEuler1D(N, dx, gamma=GAMMA, bc="outflow", device="cuda")
    solver.initialize(Q0)
    for _ in range(N_WARM):
        solver.initialize(Q0)
        n = solver.run(T_END, CFL_WARP)
    wp.synchronize()
    times = []
    for _ in range(N_BENCH):
        solver.initialize(Q0)
        t0      = time.perf_counter()
        n  = solver.run(T_END, CFL_WARP)
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    t_med = float(np.median(times))
    mcell_s = N * n / t_med / 1e6
    state = solver.state
    rho, u, p = cons_to_prim(state, GAMMA)
    del solver; gc.collect()
    return mcell_s, rho, u, p, n, t_med, x


def bench_jax_ref(N):
    import jax
    dx  = 1.0 / N
    Q0, x = sod_ic(N, GAMMA)
    gpu = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]
    with jax.default_device(gpu):
        solver = JaxEuler1D(N, dx, gamma=GAMMA, bc="outflow")
        solver.initialize(Q0)
        for _ in range(N_WARM):
            solver.initialize(Q0)
            n = solver.run(T_END, CFL_WARP)
        jax.block_until_ready(solver._Q)
        times = []
        for _ in range(N_BENCH):
            solver.initialize(Q0)
            t0      = time.perf_counter()
            n  = solver.run(T_END, CFL_WARP)
            jax.block_until_ready(solver._Q)
            times.append(time.perf_counter() - t0)
    t_med = float(np.median(times))
    mcell_s = N * n / t_med / 1e6
    Q = np.asarray(solver._Q)
    rho, u, p = cons_to_prim(Q, GAMMA)
    del solver; gc.collect()
    return mcell_s, rho, u, p, n, t_med


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import warp as wp
    wp.init()

    # load JaxFluids templates once
    try:
        case_tmpl, num_setup = _load_jxf_templates()
        jxf_ok = True
        print(f"[info] JaxFluids templates loaded from {JXF_EX}")
    except Exception as e:
        print(f"[warn] JaxFluids templates not found: {e}")
        jxf_ok = False

    tmp_dir = Path(tempfile.mkdtemp(prefix="jxf_sod_"))
    if jxf_ok:
        num_path = tmp_dir / "numerical_setup.json"
        with open(num_path, "w") as f:
            json.dump(num_setup, f)

    results = {k: {"N":[], "tp":[], "n":[], "rho":None, "u":None, "p":None}
               for k in ["JaxFluids", "JAX CUDA (ours)", "Warp CUDA"]}

    for N in GRID_SIZES:
        dx = 1.0 / N
        print(f"\nN = {N:>6}", flush=True)

        if jxf_ok:
            try:
                tp, rho, u, p, ns, tm = bench_jaxfluids(
                    N, case_tmpl, num_path, tmp_dir)
                results["JaxFluids"]["N"].append(N)
                results["JaxFluids"]["tp"].append(tp)
                results["JaxFluids"]["n"].append(ns)
                if N == 512:
                    results["JaxFluids"]["rho"] = rho
                    results["JaxFluids"]["u"]   = u
                    results["JaxFluids"]["p"]   = p
                print(f"  JaxFluids      {tp:8.2f} Mcell/s  ({ns} steps est, {tm:.2f}s)")
            except Exception as e:
                print(f"  JaxFluids      ERROR: {e}")

        try:
            tp, rho, u, p, ns, tm = bench_jax_ref(N)
            results["JAX CUDA (ours)"]["N"].append(N)
            results["JAX CUDA (ours)"]["tp"].append(tp)
            results["JAX CUDA (ours)"]["n"].append(ns)
            if N == 512:
                results["JAX CUDA (ours)"]["rho"] = rho
                results["JAX CUDA (ours)"]["u"]   = u
                results["JAX CUDA (ours)"]["p"]   = p
            print(f"  JAX CUDA (ours){tp:8.2f} Mcell/s  ({ns} steps, {tm:.2f}s)")
        except Exception as e:
            print(f"  JAX CUDA (ours) ERROR: {e}")

        try:
            tp, rho, u, p, ns, tm, x = bench_warp(N)
            results["Warp CUDA"]["N"].append(N)
            results["Warp CUDA"]["tp"].append(tp)
            results["Warp CUDA"]["n"].append(ns)
            if N == 512:
                results["Warp CUDA"]["rho"] = rho
                results["Warp CUDA"]["u"]   = u
                results["Warp CUDA"]["p"]   = p
                x512 = x
            print(f"  Warp CUDA      {tp:8.2f} Mcell/s  ({ns} steps, {tm:.2f}s)")
        except Exception as e:
            print(f"  Warp CUDA       ERROR: {e}")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── accuracy table at N=512 ───────────────────────────────────────────────
    Q0, x512 = sod_ic(512, GAMMA)
    rho_ex, u_ex, p_ex = sod_exact(T_END, x512, GAMMA)

    print("\n-- Accuracy vs exact Riemann (N=512, t=0.2) --------------------")
    print(f"{'Solver':<22}  {'L1(rho)':>10}  {'L1(u)':>10}  {'L1(p)':>10}  {'Scheme'}")
    print("-" * 75)
    schemes = {
        "JaxFluids":      "WENO5-Z + HLLC + RK3 (f64)",
        "JAX CUDA (ours)":"WENO3   + HLLC + RK2 (f32)",
        "Warp CUDA":      "WENO3   + HLLC + RK2 (f32)",
    }
    dx = 1.0 / 512
    for name, r in results.items():
        if r["rho"] is None:
            print(f"{name:<22}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {schemes.get(name,'')}")
            continue
        e_r = l1_error(r["rho"], rho_ex, dx)
        e_u = l1_error(r["u"],   u_ex,   dx)
        e_p = l1_error(r["p"],   p_ex,   dx)
        print(f"{name:<22}  {e_r:>10.3e}  {e_u:>10.3e}  {e_p:>10.3e}  {schemes.get(name,'')}")

    # ── throughput table ──────────────────────────────────────────────────────
    print("\n-- Throughput (Mcell/s) -----------------------------------------")
    hdr = f"{'N':>7}" + "".join(f"  {n:>20}" for n in results)
    print(hdr); print("-" * len(hdr))
    for N in GRID_SIZES:
        row = f"{N:>7}"
        for d in results.values():
            if N in d["N"]:
                row += f"  {d['tp'][d['N'].index(N)]:>20.2f}"
            else:
                row += f"  {'--':>20}"
        print(row)

    # ── plots ─────────────────────────────────────────────────────────────────
    colors = {"JaxFluids": "#e07b00", "JAX CUDA (ours)": "#d55e00", "Warp CUDA": "#009e73"}
    styles = {"JaxFluids": "D-",      "JAX CUDA (ours)": "o--",      "Warp CUDA": "^-"}

    # throughput
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, d in results.items():
        if not d["N"]: continue
        m, ls = styles[name][0], styles[name][1:]
        ax.plot(d["N"], d["tp"], marker=m, ls=ls, color=colors[name], lw=1.8, ms=7, label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel("Throughput  (Mcell-updates / s)", fontsize=12)
    ax.set_title(
        "JaxFluids vs Warp CUDA  —  Sod shock tube throughput\n"
        "JaxFluids: WENO5-Z+HLLC+RK3 (f64)   |   Warp/JAX ours: WENO3+HLLC+RK2 (f32)",
        fontsize=10)
    ax.legend(fontsize=11); ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.set_xticks(GRID_SIZES)
    ax.set_xticklabels([f"{n:,}" for n in GRID_SIZES], rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    out = OUT / "jaxfluids_throughput.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved -> {out}")

    # profiles at N=512
    Q0, x = sod_ic(512, GAMMA)
    rho_ex, u_ex, p_ex = sod_exact(T_END, x, GAMMA)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"Sod shock tube  |  N=512  |  t={T_END}", fontsize=12, fontweight="bold")
    fields = ["density", "velocity", "pressure"]
    exact_v = [rho_ex, u_ex, p_ex]
    for ax, fname, ev in zip(axes, fields, exact_v):
        ax.plot(x, ev, "k-", lw=2.0, label="exact", zorder=5)
        for name, d in results.items():
            if d["rho"] is None: continue
            vals = {"density": d["rho"], "velocity": d["u"], "pressure": d["p"]}
            ax.plot(x, vals[fname], color=colors[name], lw=1.4,
                    ls=styles[name][1:], label=name, alpha=0.85)
        ax.set_xlabel("x"); ax.set_title(fname)
        ax.legend(fontsize=7); ax.grid(True, lw=0.4, alpha=0.5)
    plt.tight_layout()
    out2 = OUT / "jaxfluids_profiles.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out2}")


if __name__ == "__main__":
    main()
