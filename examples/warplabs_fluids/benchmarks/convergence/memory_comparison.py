"""
GPU memory comparison: JaxFluids vs JAX CUDA (ours) vs Warp CUDA.

Measures peak GPU memory delta (nvidia-smi) for each solver at N=512..4096.
Uses XLA_PYTHON_CLIENT_PREALLOCATE=false for honest JAX measurements.

Run inside the Python 3.11 JaxFluids venv on WSL2:
  source /root/venv-jf/bin/activate
  cd examples/warplabs_fluids
  XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/convergence/memory_comparison.py
"""

import json, sys, subprocess, tempfile, gc, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path(__file__).parent.parent.parent
OUT    = Path(__file__).parent
JXF_EX = Path("/root/JAXFLUIDS/examples/examples_1D/02_sod_shock_tube")

sys.path.insert(0, str(ROOT))

from warplabs_fluids import WarpEuler1D, cons_to_prim
from cases.sod import ic as sod_ic
from benchmarks.jax_euler import JaxEuler1D

GRID_SIZES = [512, 1024, 2048, 4096]
GAMMA      = 1.4
T_END      = 0.2
CFL        = 0.4


def _nvml_mem_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--id=0",
         "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        text=True
    )
    return int(out.strip())


def measure_mem(setup_fn):
    """Return peak GPU memory delta in MiB."""
    gc.collect()
    time.sleep(0.5)
    baseline = _nvml_mem_mib()
    setup_fn()
    gc.collect()
    time.sleep(0.5)
    peak = _nvml_mem_mib()
    return max(0, peak - baseline)


def _patch_case(case_tmpl, N, run_dir):
    case = json.loads(json.dumps(case_tmpl))
    case["domain"]["x"]["cells"] = N
    case["general"]["save_path"] = str(run_dir)
    case["general"]["save_dt"]   = 999.0
    (run_dir / "case.json").write_text(json.dumps(case))
    return run_dir / "case.json"


def main():
    import warp as wp
    wp.init()

    try:
        with open(JXF_EX / "sod.json")             as f: case_tmpl = json.load(f)
        with open(JXF_EX / "numerical_setup.json")  as f: ns_dict   = json.load(f)
        jxf_ok = True
    except Exception as e:
        print(f"[warn] JaxFluids not available: {e}")
        jxf_ok = False

    base_tmp = Path(tempfile.mkdtemp(prefix="mem_cmp_"))
    if jxf_ok:
        num_path = base_tmp / "numerical_setup.json"
        num_path.write_text(json.dumps(ns_dict))

    mem = {
        "JaxFluids\n(WENO5-Z, f64)": [],
        "JAX CUDA\n(WENO3, f32)":    [],
        "Warp CUDA\n(WENO3, f32)":   [],
    }

    for N in GRID_SIZES:
        dx = 1.0 / N
        Q0, _ = sod_ic(N, GAMMA)
        print(f"\nN = {N}", flush=True)

        if jxf_ok:
            try:
                from jaxfluids import InputManager, InitializationManager, SimulationManager
                run_dir = base_tmp / f"jxf_N{N}"
                run_dir.mkdir(exist_ok=True)
                case_path = _patch_case(case_tmpl, N, run_dir)

                def _jxf():
                    im   = InputManager(str(case_path), str(num_path))
                    init = InitializationManager(im)
                    sim  = SimulationManager(im)
                    buf  = init.initialization()
                    sim.simulate(buf)

                mib = measure_mem(_jxf)
                mem["JaxFluids\n(WENO5-Z, f64)"].append(mib if mib > 0 else None)
                print(f"  JaxFluids  {mib} MiB", flush=True)
                gc.collect()
                import jax; jax.clear_caches()
            except Exception as e:
                print(f"  JaxFluids  ERROR: {e}")
                mem["JaxFluids\n(WENO5-Z, f64)"].append(None)

        try:
            import jax
            def _jax():
                gpu = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]
                with jax.default_device(gpu):
                    solver = JaxEuler1D(N, dx, gamma=GAMMA, bc="outflow")
                    solver.initialize(Q0)
                    solver.run(T_END, CFL)
                    jax.block_until_ready(solver._Q)

            mib = measure_mem(_jax)
            mem["JAX CUDA\n(WENO3, f32)"].append(mib if mib > 0 else None)
            print(f"  JAX CUDA   {mib} MiB", flush=True)
            gc.collect()
        except Exception as e:
            print(f"  JAX CUDA   ERROR: {e}")
            mem["JAX CUDA\n(WENO3, f32)"].append(None)

        try:
            def _warp():
                solver = WarpEuler1D(N, dx, gamma=GAMMA, bc="outflow", device="cuda")
                solver.initialize(Q0)
                solver.run(T_END, CFL)
                wp.synchronize()

            mib = measure_mem(_warp)
            mem["Warp CUDA\n(WENO3, f32)"].append(mib if mib > 0 else None)
            print(f"  Warp CUDA  {mib} MiB", flush=True)
            gc.collect()
        except Exception as e:
            print(f"  Warp CUDA  ERROR: {e}")
            mem["Warp CUDA\n(WENO3, f32)"].append(None)

    print("\n── GPU memory delta (MiB) ──────────────────────────────────────────")
    print(f"{'N':>6}", end="")
    for name in mem:
        print(f"  {name.split(chr(10))[0]:>22}", end="")
    print(); print("-" * 80)
    for i, N in enumerate(GRID_SIZES):
        print(f"{N:>6}", end="")
        for vals in mem.values():
            v = vals[i] if i < len(vals) else None
            print(f"  {str(v) if v is not None else '<1':>22}", end="")
        print()

    jax_theory = [2 * 3 * N * 4 / 1024**2 for N in GRID_SIZES]
    jxf_theory = [8 * 5 * N * 8 / 1024**2 for N in GRID_SIZES]

    colors = {
        "JaxFluids\n(WENO5-Z, f64)": "#e07b00",
        "JAX CUDA\n(WENO3, f32)":    "#d55e00",
        "Warp CUDA\n(WENO3, f32)":   "#009e73",
    }
    markers = {
        "JaxFluids\n(WENO5-Z, f64)": "D",
        "JAX CUDA\n(WENO3, f32)":    "o",
        "Warp CUDA\n(WENO3, f32)":   "^",
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    for name, vals in mem.items():
        ns  = [GRID_SIZES[i] for i, v in enumerate(vals) if v is not None]
        vs  = [v for v in vals if v is not None]
        tri = [GRID_SIZES[i] for i, v in enumerate(vals) if v is None]
        if ns:
            ax.plot(ns, vs, marker=markers[name], color=colors[name],
                    lw=1.8, ms=7, label=name.replace("\n", "  "))
        if tri:
            ax.scatter(tri, [0.4]*len(tri), marker="v", color=colors[name],
                       s=50, zorder=5)

    ax.plot(GRID_SIZES, jax_theory, ls=":", color="#aaa", lw=1.0,
            label="JAX f32 theory  (2×3×N×4 B)")
    ax.plot(GRID_SIZES, jxf_theory, ls="--", color="#ccc", lw=1.0,
            label="JaxFluids f64 theory  (8×5×N×8 B, est.)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of cells  N", fontsize=11)
    ax.set_ylabel("GPU memory delta  (MiB)", fontsize=11)
    ax.set_title(
        "GPU memory footprint — Sod shock tube\n"
        "nvidia-smi delta (1 MiB resolution); ▼ = below 1 MiB floor",
        fontsize=10
    )
    ax.set_xticks(GRID_SIZES)
    ax.set_xticklabels([f"{n:,}" for n in GRID_SIZES], fontsize=9)
    ax.legend(fontsize=9); ax.grid(True, which="both", lw=0.4, alpha=0.5)
    plt.tight_layout()

    out = OUT / "memory_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
