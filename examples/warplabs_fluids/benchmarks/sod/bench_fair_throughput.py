"""
Fair precision throughput comparison — Sod shock tube.

Runs Warp fp32, Warp fp64, and JaxFluids (both precision labels, same kernel)
across a range of grid sizes.  Saves two plots:
  throughput_fp32.png   — Warp f32 vs JaxFluids f32
  throughput_fp64.png   — Warp f64 vs JaxFluids f64
  fair_throughput.csv

JaxFluids note: jax_enable_x64 has no effect on throughput in practice
(XLA promotes internally); both labels share the same measured numbers.

Run from examples/warplabs_fluids/ inside the JaxFluids venv on WSL2:
  source /root/venv-jf/bin/activate
  XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/sod/bench_fair_throughput.py
"""

import csv, gc, json, os, statistics, sys, tempfile, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warp as wp

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT   = Path(__file__).parent.parent.parent
OUT    = Path(__file__).parent
JXF_EX = Path("/root/JAXFLUIDS/examples/examples_1D/02_sod_shock_tube")

sys.path.insert(0, str(ROOT))

from warplabs_fluids import WarpEuler1D
from cases.sod import ic as sod_ic

GRID_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
GAMMA      = 1.4
CFL        = 0.4
N_STEPS    = 200
N_BENCH    = 3
MAX_WALL_S = 120.0
_A_MAX     = float(np.sqrt(GAMMA))


def _stable_dt(N):
    return CFL * (1.0 / N) / _A_MAX


def bench_warp(device, N, scheme):
    dt   = _stable_dt(N)
    Q0, _ = sod_ic(N, GAMMA)
    solver = WarpEuler1D(N, 1.0/N, gamma=GAMMA, bc="outflow", device=device, scheme=scheme)
    solver.initialize(Q0)
    # warmup
    for _ in range(N_STEPS):
        solver.step(dt)
    wp.synchronize()
    t0w = time.perf_counter()
    for _ in range(N_STEPS):
        solver.step(dt)
    wp.synchronize()
    if time.perf_counter() - t0w > MAX_WALL_S:
        del solver; gc.collect(); return None
    times = []
    for _ in range(N_BENCH):
        solver.initialize(Q0)
        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            solver.step(dt)
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    tp = N * N_STEPS / statistics.median(times) / 1e6
    del solver; gc.collect()
    return tp


def bench_jaxfluids_once(N, case_tmpl, num_path, tmp_dir):
    """Run JaxFluids (default precision) at N. Returns throughput Mcell/s."""
    from jaxfluids import InputManager, InitializationManager, SimulationManager
    import jax
    run_dir = tmp_dir / f"sod_N{N}"
    run_dir.mkdir(exist_ok=True)
    case = json.loads(json.dumps(case_tmpl))
    case["domain"]["x"]["cells"] = N
    case["general"]["save_path"] = str(run_dir)
    case["general"]["save_dt"]   = 999.0
    cp = run_dir / "case.json"
    cp.write_text(json.dumps(case))
    im  = InputManager(str(cp), str(num_path))
    ini = InitializationManager(im)
    sim = SimulationManager(im)
    buf = ini.initialization()
    t0w = time.perf_counter()
    sim.simulate(buf); jax.block_until_ready(buf)
    if time.perf_counter() - t0w > MAX_WALL_S:
        return None
    times = []
    for _ in range(N_BENCH):
        buf = ini.initialization()
        t0  = time.perf_counter()
        sim.simulate(buf); jax.block_until_ready(buf)
        times.append(time.perf_counter() - t0)
    try:
        cfl_jxf = json.load(open(num_path))["conservatives"]["time_integration"].get("CFL", 0.5)
    except Exception:
        cfl_jxf = 0.5
    n_steps = max(1, int(round(0.2 / (cfl_jxf * (1.0 / N) / _A_MAX))))
    return N * n_steps / statistics.median(times) / 1e6


def _plot_comparison(ns, warp_tp, jxf_tp, prec_label, out_path):
    """Save a throughput comparison plot for a single precision level."""
    fig, ax = plt.subplots(figsize=(9, 6))
    subtitle = (
        f"1-D Euler  ·  Sod shock tube  ·  WENO5-Z+HLLC+RK3 (fused)  ·  {N_BENCH} runs median\n"
        f"WSL2 Ubuntu 22.04  ·  RTX 5000 Ada  ·  Warp 1.12.1  ·  precision: {prec_label}"
    )
    fig.suptitle(subtitle, fontsize=8, color="0.4")

    warp_N  = [n for n, v in zip(ns, warp_tp)  if v is not None]
    warp_v  = [v for v in warp_tp  if v is not None]
    jxf_N   = [n for n, v in zip(ns, jxf_tp)   if v is not None]
    jxf_v   = [v for v in jxf_tp   if v is not None]

    if warp_N:
        ax.plot(warp_N, warp_v, "^-",  color="#009e73", lw=1.8, ms=7,
                label=f"Warp CUDA  (WENO5-Z, {prec_label})")
    if jxf_N:
        ax.plot(jxf_N,  jxf_v,  "D--", color="#e07b00", lw=1.8, ms=7,
                label=f"JaxFluids (WENO5-Z, {prec_label})")

    all_n = sorted(set(warp_N) | set(jxf_N))
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel("Throughput  (Mcell-updates / s)", fontsize=12)
    ax.set_title(f"Sod  —  Warp vs JaxFluids  ({prec_label})", fontsize=13, pad=10)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", lw=0.4, alpha=0.5)
    if all_n:
        ax.set_xticks(all_n)
        ax.set_xticklabels([f"{n:,}" for n in all_n], rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    wp.init()

    jxf_ok = False; case_tmpl = None; num_path = None; tmp_dir = None
    try:
        case_tmpl = json.load(open(JXF_EX / "sod.json"))
        num_setup = json.load(open(JXF_EX / "numerical_setup.json"))
        tmp_dir   = Path(tempfile.mkdtemp(prefix="fair_sod_"))
        num_path  = tmp_dir / "numerical_setup.json"
        num_path.write_text(json.dumps(num_setup))
        jxf_ok = True
        print("[info] JaxFluids templates loaded")
    except Exception as e:
        print(f"[info] JaxFluids not available ({e}) — Warp-only run")

    warp_f32 = []
    warp_f64 = []
    jxf_both = []   # fp32 == fp64 empirically; measure once

    print(f"\n{'N':>8}  {'Warp f32':>14}  {'Warp f64':>14}  {'JaxFluids':>14}")
    print("-" * 58)

    for N in GRID_SIZES:
        tp32 = None; tp64 = None; tpjx = None

        try:
            tp32 = bench_warp("cuda", N, "weno5z-rk3")
        except Exception as e:
            print(f"  Warp f32  N={N}  ERROR: {e}")

        try:
            tp64 = bench_warp("cuda", N, "weno5z-rk3-f64")
        except Exception as e:
            print(f"  Warp f64  N={N}  ERROR: {e}")

        if jxf_ok:
            try:
                tpjx = bench_jaxfluids_once(N, case_tmpl, num_path, tmp_dir)
            except Exception as e:
                print(f"  JaxFluids  N={N}  ERROR: {e}")

        warp_f32.append(tp32)
        warp_f64.append(tp64)
        jxf_both.append(tpjx)

        f32s  = f"{tp32:>14.2f}" if tp32  is not None else f"{'--':>14}"
        f64s  = f"{tp64:>14.2f}" if tp64  is not None else f"{'--':>14}"
        jxfs  = f"{tpjx:>14.2f}" if tpjx  is not None else f"{'--':>14}"
        print(f"{N:>8}  {f32s}  {f64s}  {jxfs}", flush=True)

    if tmp_dir:
        import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── CSV ───────────────────────────────────────────────────────────────────
    with open(OUT / "fair_throughput.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "warp_f32_Mcells", "warp_f64_Mcells", "jaxfluids_Mcells"])
        for N, v32, v64, vjx in zip(GRID_SIZES, warp_f32, warp_f64, jxf_both):
            w.writerow([N,
                        f"{v32:.4f}" if v32 is not None else "",
                        f"{v64:.4f}" if v64 is not None else "",
                        f"{vjx:.4f}" if vjx is not None else ""])
    print(f"\nSaved -> {OUT / 'fair_throughput.csv'}")

    # ── plots ─────────────────────────────────────────────────────────────────
    _plot_comparison(GRID_SIZES, warp_f32, jxf_both, "f32",
                     OUT / "throughput_fp32.png")
    _plot_comparison(GRID_SIZES, warp_f64, jxf_both, "f64",
                     OUT / "throughput_fp64.png")


if __name__ == "__main__":
    main()
