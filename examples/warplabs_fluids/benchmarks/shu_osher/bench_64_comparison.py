"""
Warp f64 vs JaxFluids f64 — Shu-Osher shock-density interaction.
Capped at N=4096; JaxFluids allowed 900s per run so it can complete.

Run from examples/warplabs_fluids/ inside the JaxFluids venv on WSL2:
  source /root/venv-jf/bin/activate
  XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/shu_osher/bench_64_comparison.py
"""

import csv, gc, json, os, statistics, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warp as wp

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT   = Path(__file__).parent.parent.parent
OUT    = Path(__file__).parent
JXF_EX = Path("/root/JAXFLUIDS/examples/examples_1D/05_shock_density_interaction")

sys.path.insert(0, str(ROOT))

from warplabs_fluids import WarpEuler1D
from cases.shu_osher import ic as shu_ic, L, T_END, GAMMA, RHO_L, U_L, P_L

GRID_SIZES = [256, 512, 1024, 2048, 4096]
CFL        = 0.4
N_STEPS    = 200
N_BENCH    = 3
MAX_WALL_S = 900.0          # Shu-Osher N=4096 needs ~674s (1 warmup + 3 bench)
_c_L       = float(np.sqrt(GAMMA * P_L / RHO_L))
_A_MAX     = abs(U_L) + _c_L   # ≈ 4.57


def _stable_dt(N):
    return CFL * (L / N) / _A_MAX


def bench_warp_f64(N):
    dt    = _stable_dt(N)
    Q0, _ = shu_ic(N, GAMMA)
    solver = WarpEuler1D(N, L/N, gamma=GAMMA, bc="outflow",
                         device="cuda", scheme="weno5z-rk3-f64")
    solver.initialize(Q0)
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


_JXF_WORKER = """
import json, os, shutil, statistics, sys, tempfile, time
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORMS"] = "cuda"
args  = json.loads(sys.argv[1])
case_tmpl, num_path_s, N, N_BENCH, A_MAX, T_END, L = (
    args["case_tmpl"], args["num_path"], args["N"],
    args["N_BENCH"], args["A_MAX"], args["T_END"], args["L"])
import jax
from jaxfluids import InputManager, InitializationManager, SimulationManager
from pathlib import Path
with tempfile.TemporaryDirectory(prefix=f"jxf_shu64_{N}_") as td:
    td = Path(td)
    case = json.loads(json.dumps(case_tmpl))
    case["domain"]["x"]["cells"] = N
    case["general"]["save_path"] = str(td)
    case["general"]["save_dt"]   = 999.0
    cp = td / "case.json"; cp.write_text(json.dumps(case))
    shutil.copy(num_path_s, td / "numerical_setup.json")
    im  = InputManager(str(cp), str(td / "numerical_setup.json"))
    ini = InitializationManager(im); sim = SimulationManager(im)
    buf = ini.initialization(); sim.simulate(buf); jax.block_until_ready(buf)
    times = []
    for _ in range(N_BENCH):
        buf = ini.initialization(); t0 = time.perf_counter()
        sim.simulate(buf); jax.block_until_ready(buf)
        times.append(time.perf_counter() - t0)
try:
    cfl_jxf = json.load(open(num_path_s))["conservatives"]["time_integration"].get("CFL", 0.5)
except Exception:
    cfl_jxf = 0.5
n_steps = max(1, int(round(T_END / (cfl_jxf * (L / N) / A_MAX))))
print(json.dumps({"tp": N * n_steps / statistics.median(times) / 1e6}))
"""


def bench_jaxfluids_f64(N, case_tmpl, num_path, tmp_dir):
    args = json.dumps({
        "case_tmpl": case_tmpl, "num_path": str(num_path),
        "N": N, "N_BENCH": N_BENCH, "A_MAX": _A_MAX, "T_END": T_END, "L": L,
    })
    worker = tmp_dir / "_jxf_worker64.py"
    worker.write_text(_JXF_WORKER)
    try:
        r = subprocess.run(
            [sys.executable, str(worker), args],
            capture_output=True, text=True,
            timeout=MAX_WALL_S,
            env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                 "JAX_PLATFORMS": "cuda"},
        )
        if r.returncode != 0:
            return None
        return json.loads(r.stdout.strip())["tp"]
    except (subprocess.TimeoutExpired, Exception):
        return None


def main():
    wp.init()

    jxf_ok = False; case_tmpl = None; num_path = None; tmp_dir = None
    try:
        cands = ["shock_density_interaction.json", "shu_osher.json",
                 "shock-density-interaction.json"]
        case_json = None
        for c in cands:
            p = JXF_EX / c
            if p.exists():
                case_json = p; break
        if case_json is None:
            jsons = list(JXF_EX.glob("*.json"))
            case_json = next(f for f in jsons if "numerical" not in f.name)
        case_tmpl = json.load(open(case_json))
        num_setup = json.load(open(JXF_EX / "numerical_setup.json"))
        # fp64 precision, silence logging
        num_setup["precision"] = {
            "is_double_precision_compute": True,
            "is_double_precision_output":  True,
        }
        num_setup.setdefault("output", {}).setdefault("logging", {})["level"] = "NONE"
        tmp_dir  = Path(tempfile.mkdtemp(prefix="shu64_"))
        num_path = tmp_dir / "numerical_setup.json"
        num_path.write_text(json.dumps(num_setup))
        jxf_ok = True
        print(f"[info] JaxFluids loaded from {case_json.name} (fp64, logging=NONE, timeout=900s)")
    except Exception as e:
        print(f"[info] JaxFluids not available ({e}) — Warp-only run")

    warp_f64 = []
    jxf_f64  = []

    print(f"\n{'N':>8}  {'Warp f64':>14}  {'JxF f64':>14}")
    print("-" * 42)

    for N in GRID_SIZES:
        tw = None; tj = None

        try:
            tw = bench_warp_f64(N)
        except Exception as e:
            print(f"  Warp f64 N={N} ERROR: {e}")

        if jxf_ok:
            try:
                tj = bench_jaxfluids_f64(N, case_tmpl, num_path, tmp_dir)
            except Exception as e:
                print(f"  JxF f64  N={N} ERROR: {e}")

        warp_f64.append(tw)
        jxf_f64.append(tj)

        ws = f"{tw:>14.2f}" if tw is not None else f"{'--':>14}"
        js = f"{tj:>14.2f}" if tj is not None else f"{'--':>14}"
        print(f"{N:>8}  {ws}  {js}", flush=True)

    if tmp_dir:
        import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── CSV ───────────────────────────────────────────────────────────────────
    with open(OUT / "comparison_64.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "warp_f64_Mcells", "jxf_f64_Mcells"])
        for N, vw, vj in zip(GRID_SIZES, warp_f64, jxf_f64):
            w.writerow([N,
                        f"{vw:.4f}" if vw is not None else "",
                        f"{vj:.4f}" if vj is not None else ""])
    print(f"\nSaved -> {OUT / 'comparison_64.csv'}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    subtitle = (
        "1-D Euler  ·  Shu-Osher  ·  WENO5-Z+HLLC+RK3  ·  fp64  ·  3 runs median\n"
        "WSL2 Ubuntu 22.04  ·  RTX 5000 Ada  ·  Warp 1.12.1  ·  JAX_PLATFORMS=cuda"
    )
    fig.suptitle(subtitle, fontsize=8, color="0.4")

    wN = [n for n, v in zip(GRID_SIZES, warp_f64) if v is not None]
    wV = [v for v in warp_f64 if v is not None]
    jN = [n for n, v in zip(GRID_SIZES, jxf_f64)  if v is not None]
    jV = [v for v in jxf_f64 if v is not None]

    if wN: ax.plot(wN, wV, "^-",  color="#009e73", lw=1.8, ms=7, label="Warp CUDA (WENO5-Z, f64)")
    if jN: ax.plot(jN, jV, "D--", color="#e07b00", lw=1.8, ms=7, label="JaxFluids (WENO5-Z, f64)")

    all_n = sorted(set(wN) | set(jN))
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel("Throughput  (Mcell-updates / s)", fontsize=12)
    ax.set_title("Shu-Osher  —  Warp f64 vs JaxFluids f64", fontsize=13, pad=10)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", lw=0.4, alpha=0.5)
    if all_n:
        ax.set_xticks(all_n)
        ax.set_xticklabels([f"{n:,}" for n in all_n], rotation=35, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "comparison_64.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT / 'comparison_64.png'}")


if __name__ == "__main__":
    main()
