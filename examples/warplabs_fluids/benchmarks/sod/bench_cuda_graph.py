"""
Warp CUDA-graph vs baseline vs JaxFluids throughput — Sod shock tube.

Measures six configurations across N=256..65536:
  warp_f32          — baseline (Python loop, no graph)
  warp_f32_graph    — CUDA graph (N_STEPS captured, single launch)
  warp_f64          — baseline fp64
  warp_f64_graph    — CUDA graph fp64
  jxf_f32           — JaxFluids fp32  (subprocess, up to N=4096)
  jxf_f64           — JaxFluids fp64  (subprocess, up to N=4096)

Outputs:
  cuda_graph_benchmark.csv   — all six columns per N
  cuda_graph_scaling.png     — log-log throughput vs N (all six)

Run from examples/warplabs_fluids/ inside the JaxFluids venv on WSL2:
  source /root/venv-jf/bin/activate
  XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/sod/bench_cuda_graph.py
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
JXF_EX = Path("/root/JAXFLUIDS/examples/examples_1D/02_sod_shock_tube")

sys.path.insert(0, str(ROOT))

from warplabs_fluids import WarpEuler1D
from cases.sod import ic as sod_ic

# Grid sizes: full range for Warp; JaxFluids capped at JXF_MAX_N
GRID_SIZES  = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
JXF_MAX_N   = 4096        # JaxFluids per-step sync makes larger sizes very slow
GAMMA       = 1.4
CFL         = 0.4
N_STEPS     = 200
N_BENCH     = 3
MAX_WALL_S  = 600.0       # JaxFluids subprocess timeout
_A_MAX      = float(np.sqrt(GAMMA))


def _stable_dt(N):
    return CFL * (1.0 / N) / _A_MAX


# ── Warp baseline (Python loop) ───────────────────────────────────────────────

def bench_warp(N, scheme):
    dt    = _stable_dt(N)
    Q0, _ = sod_ic(N, GAMMA)
    solver = WarpEuler1D(N, 1.0/N, gamma=GAMMA, bc="outflow", device="cuda", scheme=scheme)
    solver.initialize(Q0)
    for _ in range(N_STEPS):
        solver.step(dt)
    wp.synchronize()
    # wall-time guard
    t0w = time.perf_counter()
    for _ in range(N_STEPS):
        solver.step(dt)
    wp.synchronize()
    if time.perf_counter() - t0w > MAX_WALL_S:
        del solver; gc.collect(); return None
    times = []
    for _ in range(N_BENCH):
        solver.reset_state(Q0)
        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            solver.step(dt)
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    tp = N * N_STEPS / statistics.median(times) / 1e6
    del solver; gc.collect()
    return tp


# ── Warp CUDA-graph ───────────────────────────────────────────────────────────

def bench_warp_graph(N, scheme):
    dt    = _stable_dt(N)
    Q0, _ = sod_ic(N, GAMMA)
    solver = WarpEuler1D(N, 1.0/N, gamma=GAMMA, bc="outflow", device="cuda", scheme=scheme)
    solver.initialize(Q0)
    # Warmup pass to JIT-compile kernels before capture
    for _ in range(N_STEPS):
        solver.step(dt)
    wp.synchronize()
    # Capture graph (reset_state preserves buffer pointer)
    solver.reset_state(Q0)
    graph = solver.capture_graph(dt, N_STEPS)
    # Warmup graph replay
    solver.reset_state(Q0)
    wp.capture_launch(graph)
    wp.synchronize()
    # wall-time guard
    t0w = time.perf_counter()
    wp.capture_launch(graph)
    wp.synchronize()
    if time.perf_counter() - t0w > MAX_WALL_S:
        del solver; gc.collect(); return None
    times = []
    for _ in range(N_BENCH):
        solver.reset_state(Q0)
        t0 = time.perf_counter()
        wp.capture_launch(graph)
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    tp = N * N_STEPS / statistics.median(times) / 1e6
    del solver; gc.collect()
    return tp


# ── JaxFluids subprocess ──────────────────────────────────────────────────────

_JXF_WORKER = """
import json, os, shutil, statistics, sys, tempfile, time
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORMS"] = "cuda"
args  = json.loads(sys.argv[1])
case_tmpl, num_path_s, N, N_BENCH, A_MAX, T_END = (
    args["case_tmpl"], args["num_path"], args["N"],
    args["N_BENCH"], args["A_MAX"], args["T_END"])
import jax
from jaxfluids import InputManager, InitializationManager, SimulationManager
from pathlib import Path
with tempfile.TemporaryDirectory(prefix=f"jxf_sodcg_{N}_") as td:
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
n_steps = max(1, int(round(T_END / (cfl_jxf * (1.0 / N) / A_MAX))))
print(json.dumps({"tp": N * n_steps / statistics.median(times) / 1e6}))
"""


def bench_jaxfluids(N, case_tmpl, num_path, tmp_dir):
    args = json.dumps({
        "case_tmpl": case_tmpl, "num_path": str(num_path),
        "N": N, "N_BENCH": N_BENCH, "A_MAX": _A_MAX, "T_END": 0.2,
    })
    worker = tmp_dir / "_jxf_cg_worker.py"
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


# ── JaxFluids fair benchmark (do_integration_step loop, single sync) ──────────

_JXF_WORKER_FAIR = """
import json, os, shutil, statistics, sys, tempfile, time
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORMS"] = "cuda"
args  = json.loads(sys.argv[1])
case_tmpl, num_path_s, N, N_BENCH, N_STEPS = (
    args["case_tmpl"], args["num_path"], args["N"],
    args["N_BENCH"], args["N_STEPS"])
import jax
from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids.data_types.ml_buffers import ParametersSetup, CallablesSetup
from pathlib import Path
with tempfile.TemporaryDirectory(prefix=f"jxf_fair_{N}_") as td:
    td = Path(td)
    case = json.loads(json.dumps(case_tmpl))
    case["domain"]["x"]["cells"] = N
    case["general"]["save_path"] = str(td)
    case["general"]["save_dt"]   = 999.0
    cp = td / "case.json"; cp.write_text(json.dumps(case))
    shutil.copy(num_path_s, td / "numerical_setup.json")
    im  = InputManager(str(cp), str(td / "numerical_setup.json"))
    ini = InitializationManager(im)
    sim = SimulationManager(im)
    ml_params    = ParametersSetup()
    ml_callables = CallablesSetup()
    buf = ini.initialization()
    cfp = sim.compute_control_flow_params(buf.time_control_variables, buf.step_information)
    buf, _ = sim.do_integration_step(buf, cfp, ml_params, ml_callables)
    jax.block_until_ready(buf.simulation_buffers.material_fields.primitives)
    times = []
    for _ in range(N_BENCH):
        buf = ini.initialization()
        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            buf, _ = sim.do_integration_step(buf, cfp, ml_params, ml_callables)
        jax.block_until_ready(buf.simulation_buffers.material_fields.primitives)
        times.append(time.perf_counter() - t0)
print(json.dumps({"tp": N * N_STEPS / statistics.median(times) / 1e6}))
"""


def bench_jaxfluids_fair(N, case_tmpl, num_path, tmp_dir):
    args = json.dumps({
        "case_tmpl": case_tmpl, "num_path": str(num_path),
        "N": N, "N_BENCH": N_BENCH, "N_STEPS": N_STEPS,
    })
    worker = tmp_dir / "_jxf_fair_worker.py"
    worker.write_text(_JXF_WORKER_FAIR)
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    wp.init()

    # JaxFluids setup (optional)
    jxf_ok = False; case_tmpl = None; tmp_dir = None
    num_path_f32 = None; num_path_f64 = None
    try:
        case_tmpl  = json.load(open(JXF_EX / "sod.json"))
        base_setup = json.load(open(JXF_EX / "numerical_setup.json"))
        base_setup.setdefault("output", {}).setdefault("logging", {})["level"] = "NONE"
        tmp_dir = Path(tempfile.mkdtemp(prefix="cg_sod_"))

        ns32 = json.loads(json.dumps(base_setup))
        ns32["precision"] = {"is_double_precision_compute": False,
                             "is_double_precision_output":  False}
        num_path_f32 = tmp_dir / "ns_f32.json"; num_path_f32.write_text(json.dumps(ns32))

        ns64 = json.loads(json.dumps(base_setup))
        ns64["precision"] = {"is_double_precision_compute": True,
                             "is_double_precision_output":  True}
        num_path_f64 = tmp_dir / "ns_f64.json"; num_path_f64.write_text(json.dumps(ns64))

        jxf_ok = True
        print("[info] JaxFluids loaded (fp32+fp64, logging=NONE)")
    except Exception as e:
        print(f"[info] JaxFluids not available ({e}) — Warp-only run")

    results = {k: [] for k in [
        "warp_f32", "warp_f32_graph",
        "warp_f64", "warp_f64_graph",
        "jxf_f32",  "jxf_f64",
        "jxf_f32_fair", "jxf_f64_fair",
    ]}

    hdr = (f"{'N':>8}  {'w_f32':>12}  {'w_f32_gr':>12}  {'w_f64':>12}  {'w_f64_gr':>12}"
           f"  {'jxf_f32':>10}  {'jxf_f64':>10}  {'jxf_f32_fair':>13}  {'jxf_f64_fair':>13}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for N in GRID_SIZES:
        row = {}

        for key, scheme, use_graph in [
            ("warp_f32",       "weno5z-rk3",     False),
            ("warp_f32_graph", "weno5z-rk3",     True),
            ("warp_f64",       "weno5z-rk3-f64", False),
            ("warp_f64_graph", "weno5z-rk3-f64", True),
        ]:
            try:
                fn = bench_warp_graph if use_graph else bench_warp
                row[key] = fn(N, scheme)
            except Exception as ex:
                print(f"  {key} N={N} ERROR: {ex}")
                row[key] = None

        for key, num_path, fair in [
            ("jxf_f32",      num_path_f32, False),
            ("jxf_f64",      num_path_f64, False),
            ("jxf_f32_fair", num_path_f32, True),
            ("jxf_f64_fair", num_path_f64, True),
        ]:
            if jxf_ok and N <= JXF_MAX_N:
                try:
                    fn = bench_jaxfluids_fair if fair else bench_jaxfluids
                    row[key] = fn(N, case_tmpl, num_path, tmp_dir)
                except Exception as ex:
                    print(f"  {key} N={N} ERROR: {ex}")
                    row[key] = None
            else:
                row[key] = None

        for k in results:
            results[k].append(row.get(k))

        def _f(v): return f"{v:>12.2f}" if v is not None else f"{'--':>12}"
        def _g(v): return f"{v:>10.2f}" if v is not None else f"{'--':>10}"
        def _h(v): return f"{v:>13.2f}" if v is not None else f"{'--':>13}"
        print(f"{N:>8}  {_f(row['warp_f32'])}  {_f(row['warp_f32_graph'])}  "
              f"{_f(row['warp_f64'])}  {_f(row['warp_f64_graph'])}  "
              f"{_g(row['jxf_f32'])}  {_g(row['jxf_f64'])}  "
              f"{_h(row['jxf_f32_fair'])}  {_h(row['jxf_f64_fair'])}", flush=True)

    if tmp_dir:
        import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = OUT / "cuda_graph_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "warp_f32_Mcells", "warp_f32_graph_Mcells",
                    "warp_f64_Mcells", "warp_f64_graph_Mcells",
                    "jxf_f32_Mcells",  "jxf_f64_Mcells",
                    "jxf_f32_fair_Mcells", "jxf_f64_fair_Mcells"])
        for i, N in enumerate(GRID_SIZES):
            def _v(k): return f"{results[k][i]:.4f}" if results[k][i] is not None else ""
            w.writerow([N, _v("warp_f32"), _v("warp_f32_graph"),
                           _v("warp_f64"), _v("warp_f64_graph"),
                           _v("jxf_f32"),  _v("jxf_f64"),
                           _v("jxf_f32_fair"), _v("jxf_f64_fair")])
    print(f"\nSaved -> {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    subtitle = (
        "1-D Euler  ·  Sod shock tube  ·  WENO5-Z+HLLC+RK3  ·  3 runs median\n"
        "WSL2 Ubuntu 22.04  ·  RTX 5000 Ada  ·  Warp 1.12.1  ·  JAX_PLATFORMS=cuda"
    )
    fig.suptitle(subtitle, fontsize=8, color="0.4")

    styles = {
        "warp_f32":       ("^-",  "#009e73", "Warp f32 (baseline)"),
        "warp_f32_graph": ("^--", "#00c896", "Warp f32 (CUDA graph)"),
        "warp_f64":       ("s-",  "#0072b2", "Warp f64 (baseline)"),
        "warp_f64_graph": ("s--", "#56b4e9", "Warp f64 (CUDA graph)"),
        "jxf_f32":        ("D-",  "#e07b00", "JaxFluids f32 (as-shipped)"),
        "jxf_f64":        ("D--", "#cc3311", "JaxFluids f64 (as-shipped)"),
        "jxf_f32_fair":   ("o-",  "#e07b00", "JaxFluids f32 (no per-step sync)"),
        "jxf_f64_fair":   ("o--", "#cc3311", "JaxFluids f64 (no per-step sync)"),
    }

    all_n = set()
    for key, (marker, color, label) in styles.items():
        ns = [n for n, v in zip(GRID_SIZES, results[key]) if v is not None]
        vs = [v for v in results[key] if v is not None]
        if ns:
            lw = 1.8; alpha = 1.0
            if "fair" not in key and key.startswith("jxf"):
                lw = 1.2; alpha = 0.5
            ax.plot(ns, vs, marker, color=color, lw=lw, ms=7, label=label, alpha=alpha)
            all_n.update(ns)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel("Throughput  (Mcell-updates / s)", fontsize=12)
    ax.set_title("Sod  —  Warp CUDA Graph vs Baseline vs JaxFluids", fontsize=13, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", lw=0.4, alpha=0.5)
    if all_n:
        ticks = sorted(all_n)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{n:,}" for n in ticks], rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    png_path = OUT / "cuda_graph_scaling.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {png_path}")


if __name__ == "__main__":
    main()
