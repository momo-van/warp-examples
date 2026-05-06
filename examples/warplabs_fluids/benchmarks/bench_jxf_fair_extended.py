"""
Extended JaxFluids fair benchmark — N=4096, 8192, 16384 for Sod and Shu-Osher.

Runs only the fair harness (do_integration_step loop, single sync) at f32+f64.
Reads existing cuda_graph_benchmark.csv files, fills in the jxf_*_fair columns
for the extended grid sizes, and writes updated CSVs + PNGs.

Run from examples/warplabs_fluids/ inside the JaxFluids venv on WSL2:
  source /root/venv-jf/bin/activate
  XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/bench_jxf_fair_extended.py
"""

import csv, json, os, statistics, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT       = Path(__file__).parent.parent
BENCH_ROOT = Path(__file__).parent
JXF_SOD    = Path("/root/JAXFLUIDS/examples/examples_1D/02_sod_shock_tube")
JXF_SHU    = Path("/root/JAXFLUIDS/examples/examples_1D/05_shock_density_interaction")

sys.path.insert(0, str(ROOT))

GRID_SIZES = [4096, 8192, 16384]
N_STEPS    = 200
N_BENCH    = 3
MAX_WALL_S = 900.0


# ── Shared fair worker ────────────────────────────────────────────────────────

_JXF_WORKER_FAIR = """
import json, os, shutil, statistics, sys, tempfile, time
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORMS"] = "cuda"
args = json.loads(sys.argv[1])
case_tmpl, num_path_s, N, N_BENCH, N_STEPS = (
    args["case_tmpl"], args["num_path"], args["N"],
    args["N_BENCH"], args["N_STEPS"])
import jax
from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids.data_types.ml_buffers import ParametersSetup, CallablesSetup
from pathlib import Path
with tempfile.TemporaryDirectory(prefix=f"jxf_fairext_{N}_") as td:
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


def bench_fair(N, case_tmpl, num_path, tmp_dir):
    args = json.dumps({
        "case_tmpl": case_tmpl, "num_path": str(num_path),
        "N": N, "N_BENCH": N_BENCH, "N_STEPS": N_STEPS,
    })
    worker = tmp_dir / f"_jxf_fairext_{N}.py"
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
            print(f"    [stderr] {r.stderr[-200:]}")
            return None
        return json.loads(r.stdout.strip())["tp"]
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"    [error] {e}")
        return None


# ── Case setup ────────────────────────────────────────────────────────────────

def load_case(jxf_ex: Path, case_filename: str):
    case_tmpl  = json.load(open(jxf_ex / case_filename))
    base_setup = json.load(open(jxf_ex / "numerical_setup.json"))
    base_setup.setdefault("output", {}).setdefault("logging", {})["level"] = "NONE"
    return case_tmpl, base_setup


def write_num_paths(base_setup, tmp_dir):
    ns32 = json.loads(json.dumps(base_setup))
    ns32["precision"] = {"is_double_precision_compute": False,
                         "is_double_precision_output":  False}
    ns64 = json.loads(json.dumps(base_setup))
    ns64["precision"] = {"is_double_precision_compute": True,
                         "is_double_precision_output":  True}
    p32 = tmp_dir / "ns_f32.json"; p32.write_text(json.dumps(ns32))
    p64 = tmp_dir / "ns_f64.json"; p64.write_text(json.dumps(ns64))
    return p32, p64


# ── CSV update ────────────────────────────────────────────────────────────────

def update_csv(csv_path: Path, new_data: dict):
    """Merge new_data {N: {col: val}} into existing CSV."""
    rows = []
    fieldnames = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames[:]
        for row in reader:
            n = int(row["N"])
            if n in new_data:
                for col, val in new_data[n].items():
                    if col not in fieldnames:
                        fieldnames.append(col)
                    row[col] = f"{val:.4f}" if val is not None else ""
            rows.append(row)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            for fn in fieldnames:
                if fn not in row:
                    row[fn] = ""
            w.writerow(row)


# ── Plot regeneration ─────────────────────────────────────────────────────────

def regen_plot(csv_path: Path, png_path: Path, title: str, subtitle: str):
    rows = list(csv.DictReader(open(csv_path)))

    def col(key):
        return [float(r[key]) if r.get(key) else None for r in rows]

    ns = [int(r["N"]) for r in rows]

    styles = {
        "warp_f32_Mcells":        ("^-",  "#009e73", "Warp f32 (baseline)"),
        "warp_f32_graph_Mcells":  ("^--", "#00c896", "Warp f32 (CUDA graph)"),
        "warp_f64_Mcells":        ("s-",  "#0072b2", "Warp f64 (baseline)"),
        "warp_f64_graph_Mcells":  ("s--", "#56b4e9", "Warp f64 (CUDA graph)"),
        "jxf_f32_Mcells":         ("D-",  "#e07b00", "JaxFluids f32 (as-shipped)"),
        "jxf_f64_Mcells":         ("D--", "#cc3311", "JaxFluids f64 (as-shipped)"),
        "jxf_f32_fair_Mcells":    ("o-",  "#e07b00", "JaxFluids f32 (no per-step sync)"),
        "jxf_f64_fair_Mcells":    ("o--", "#cc3311", "JaxFluids f64 (no per-step sync)"),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(subtitle, fontsize=8, color="0.4")

    all_n = set()
    for key, (marker, color, label) in styles.items():
        vals = col(key)
        xs = [n for n, v in zip(ns, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        if xs:
            lw = 1.8; alpha = 1.0
            if "fair" not in key and key.startswith("jxf"):
                lw = 1.2; alpha = 0.5
            ax.plot(xs, ys, marker, color=color, lw=lw, ms=7, label=label, alpha=alpha)
            all_n.update(xs)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel("Throughput  (Mcell-updates / s)", fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", lw=0.4, alpha=0.5)
    if all_n:
        ticks = sorted(all_n)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{n:,}" for n in ticks], rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {png_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

CASES = [
    {
        "name":      "sod",
        "jxf_ex":    JXF_SOD,
        "case_file": "sod.json",
        "csv":       BENCH_ROOT / "sod" / "cuda_graph_benchmark.csv",
        "png":       BENCH_ROOT / "sod" / "cuda_graph_scaling.png",
        "title":     "Sod  —  Warp CUDA Graph vs Baseline vs JaxFluids",
        "subtitle":  ("1-D Euler  ·  Sod shock tube  ·  WENO5-Z+HLLC+RK3  ·  3 runs median\n"
                      "WSL2 Ubuntu 22.04  ·  RTX 5000 Ada  ·  Warp 1.12.1"),
    },
    {
        "name":      "shu_osher",
        "jxf_ex":    JXF_SHU,
        "case_file": None,  # auto-detect
        "csv":       BENCH_ROOT / "shu_osher" / "cuda_graph_benchmark.csv",
        "png":       BENCH_ROOT / "shu_osher" / "cuda_graph_scaling.png",
        "title":     "Shu-Osher  —  Warp CUDA Graph vs Baseline vs JaxFluids",
        "subtitle":  ("1-D Euler  ·  Shu-Osher  ·  WENO5-Z+HLLC+RK3  ·  3 runs median\n"
                      "WSL2 Ubuntu 22.04  ·  RTX 5000 Ada  ·  Warp 1.12.1"),
    },
]


def main():
    tmp_dir = Path(tempfile.mkdtemp(prefix="jxf_fairext_"))

    hdr = f"{'case':>12}  {'N':>6}  {'prec':>5}  {'Mcell/s':>10}"
    print(f"\n{hdr}\n{'-'*len(hdr)}")

    for case in CASES:
        jxf_ex = case["jxf_ex"]
        if not jxf_ex.exists():
            print(f"[skip] {case['name']} — JaxFluids dir not found: {jxf_ex}")
            continue

        # Load case JSON
        if case["case_file"]:
            case_json = jxf_ex / case["case_file"]
        else:
            cands = ["shock_density_interaction.json", "shu_osher.json",
                     "shock-density-interaction.json"]
            case_json = next(
                (jxf_ex / c for c in cands if (jxf_ex / c).exists()),
                next(f for f in jxf_ex.glob("*.json") if "numerical" not in f.name)
            )
        case_tmpl, base_setup = load_case(jxf_ex, case_json.name)
        num_path_f32, num_path_f64 = write_num_paths(base_setup, tmp_dir)

        new_data = {}  # {N: {"jxf_f32_fair_Mcells": v, "jxf_f64_fair_Mcells": v}}

        for N in GRID_SIZES:
            new_data[N] = {}
            for prec, num_path in [("f32", num_path_f32), ("f64", num_path_f64)]:
                col = f"jxf_{prec}_fair_Mcells"
                print(f"  {case['name']:>12}  {N:>6}  {prec:>5}  ", end="", flush=True)
                tp = bench_fair(N, case_tmpl, num_path, tmp_dir)
                new_data[N][col] = tp
                print(f"{tp:>10.2f}" if tp is not None else f"{'--':>10}", flush=True)

        update_csv(case["csv"], new_data)
        print(f"  Updated -> {case['csv']}")
        regen_plot(case["csv"], case["png"], case["title"], case["subtitle"])

    import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
