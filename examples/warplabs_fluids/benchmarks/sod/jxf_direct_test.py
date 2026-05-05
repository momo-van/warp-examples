"""
Standalone JaxFluids GPU throughput test — no subprocess, no Warp context.
Run directly:
  source /root/venv-jf/bin/activate
  XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/sod/jxf_direct_test.py
"""
import json, os, shutil, statistics, tempfile, time
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORMS"] = "cuda"

import jax
print("JAX devices:", jax.devices())

from pathlib import Path
from jaxfluids import InputManager, InitializationManager, SimulationManager

JXF_EX = Path("/root/JAXFLUIDS/examples/examples_1D/02_sod_shock_tube")
GRID_SIZES = [256, 512, 1024, 2048, 4096]
N_BENCH = 3
GAMMA = 1.4
A_MAX = GAMMA ** 0.5   # ~1.183
CFL_JXF = 0.5

case_tmpl = json.load(open(JXF_EX / "sod.json"))
num_setup = json.load(open(JXF_EX / "numerical_setup.json"))
# Force fp32 and silence per-step logging for fair/fast benchmarking
num_setup.setdefault("precision", {})["is_double_precision_compute"] = False
num_setup.setdefault("precision", {})["is_double_precision_output"] = False
num_setup.setdefault("output", {}).setdefault("logging", {})["level"] = "NONE"

print(f"\n{'N':>8}  {'median_s':>12}  {'n_steps':>8}  {'Mcell/s':>12}")
print("-" * 50)

for N in GRID_SIZES:
    with tempfile.TemporaryDirectory(prefix=f"jxf_direct_{N}_") as td:
        td = Path(td)
        case = json.loads(json.dumps(case_tmpl))
        case["domain"]["x"]["cells"] = N
        case["general"]["save_path"] = str(td)
        case["general"]["save_dt"] = 999.0
        cp = td / "case.json"
        cp.write_text(json.dumps(case))
        (td / "numerical_setup.json").write_text(json.dumps(num_setup))

        im  = InputManager(str(cp), str(td / "numerical_setup.json"))
        ini = InitializationManager(im)
        sim = SimulationManager(im)

        # warmup — triggers JIT compilation
        buf = ini.initialization()
        sim.simulate(buf)
        jax.block_until_ready(buf)

        times = []
        for _ in range(N_BENCH):
            buf = ini.initialization()
            t0 = time.perf_counter()
            sim.simulate(buf)
            jax.block_until_ready(buf)
            times.append(time.perf_counter() - t0)

        T_END = case["general"]["end_time"]
        dx = 1.0 / N
        n_steps = max(1, round(T_END / (CFL_JXF * dx / A_MAX)))
        med = statistics.median(times)
        tp = N * n_steps / med / 1e6
        print(f"{N:>8}  {med:>12.3f}  {n_steps:>8}  {tp:>12.4f}", flush=True)
