# warplabs-fluids

Experimental GPU-accelerated compressible flow solver built on [NVIDIA Warp](https://github.com/NVIDIA/warp).
Goal: a full Warp backend for [JaxFluids](https://github.com/tumaer/JAXFLUIDS) — same algorithm, same results, orders-of-magnitude faster on GPU.

**Phase 1 complete.** 1-D compressible Euler, WENO5-Z + HLLC + SSP-RK3, float32.

---

## Solver

| Component | Choice |
|---|---|
| Equations | 1-D compressible Euler  [ρ, ρu, E] |
| Reconstruction | WENO5-Z (Borges et al. 2008) |
| Riemann solver | HLLC (Toro 2009) |
| Time integration | SSP-RK3 (Shu & Osher 1988) |
| Ghost cells | ng = 3  (7-cell stencil) |
| Default CFL | 0.4 |
| Precision | float32 |

Matches JaxFluids' numerical scheme exactly — the only difference is float32 vs float64.

## Kernel architecture

Each fused kernel loads a 7-cell stencil, computes both interface fluxes in registers, and writes the RK update. No global flux array. BC handled inline (clamp for outflow, modulo for periodic).

```
fused_rk_stage_1d   1 launch per RK stage  (3 per timestep)
  thread i:
    load Q[i-3..i+3]         ← 7 cells from global memory
    WENO5-Z + HLLC → F_l     ← left interface, registers only
    WENO5-Z + HLLC → F_r     ← right interface, registers only
    Q_out[i] = RK update     ← single global write
```

---

## Install

```bash
pip install warp-lang numpy scipy matplotlib
```

---

## Quick start

```python
import numpy as np
from warplabs_fluids import WarpEuler1D, prim_to_cons

N, gamma = 512, 1.4
dx = 1.0 / N
x  = (np.arange(N) + 0.5) * dx

rho = np.where(x < 0.5, 1.0, 0.125)
u   = np.zeros(N)
p   = np.where(x < 0.5, 1.0, 0.1)
Q0  = prim_to_cons(rho, u, p, gamma)

solver = WarpEuler1D(N, dx, gamma=gamma, bc="outflow", scheme="weno5z-rk3")
solver.initialize(Q0)
solver.run(t_end=0.2, cfl=0.4)

rho_out = solver.state[0]
```

---

## Benchmarks

| Directory | Test case | What's in it |
|---|---|---|
| [`benchmarks/sod/`](benchmarks/sod/) | Sod shock tube | Profiles vs exact Riemann, convergence study, throughput vs JaxFluids, GPU scaling |
| [`benchmarks/shu_osher/`](benchmarks/shu_osher/) | Shu-Osher density wave | Profiles, self-convergence, throughput vs JaxFluids, GPU scaling |

Both run against JaxFluids (WENO5-Z + HLLC + SSP-RK3, float64) on RTX 5000 Ada.
Warp f32 and JaxFluids f64 produce effectively identical results — the 2.3% L1 gap is the cost of float32 vs float64, negligible at shock-capturing accuracy.
Warp CUDA throughput: **183× faster** (Sod) and **111× faster** (Shu-Osher) vs JaxFluids at N = 4 096.

---

## Tests

```bash
# From examples/warplabs_fluids/
python -m pytest tests/ -v
```

All tests run on Warp CPU — no CUDA required.

---

## Roadmap

Long-term goal: a complete Warp GPU backend for JaxFluids — drop-in replacement, identical Python API, targeting 100× throughput at production scale.

| Phase | Status | Scope |
|---|---|---|
| 1 | ✅ Done | 1-D Euler · WENO5-Z + HLLC + SSP-RK3 · Sod + Shu-Osher V&V · 183× vs JaxFluids |
| 2 | Next | 2-D Euler · Strang dimensional splitting · Kelvin-Helmholtz instability V&V |
| 3 | Planned | 3-D Euler · Rayleigh-Taylor + shock-vortex V&V · 3-D N³ scaling |
| 4 | Goal | Compressible Navier-Stokes · viscous + heat conduction · Taylor-Green vortex |
| 5 | Goal | Higher-order schemes · WENO7, TENO, MP-WENO · scheme comparison suite |
| 6 | Goal | Two-phase & interface methods · diffuse interface · bubble collapse V&V |
| 7 | Goal | Multi-GPU · domain decomposition (MPI/NCCL) · weak & strong scaling |
| 8 | Goal | Full JaxFluids Warp backend · drop-in Python API · open-source release |
