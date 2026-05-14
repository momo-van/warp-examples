# Warp Projects — Claude Context

## Repo

`C:\Vibe Coding\warp-projects` — a growing collection of GPU-accelerated projects built on NVIDIA Warp,
focused on computational physics and geometry processing.

GitHub: `momo-van/warp-projects` (private)
Owner: mmohajerani@nvidia.com (NVIDIA)

---

## Purpose

Hands-on exploration of Warp standalone (no Newton). Two goals running in parallel:
1. Build `warplabs-fluids` — a Warp-native compressible CFD solver validated against JAX reference implementations, with throughput benchmarks demonstrating GPU advantage
2. Expand to other physics domains over time (FEM, geometry processing, waves, etc.)

It is **not** tied to Newton (that lives in `newton-examples`).

---

## Active project: warplabs-fluids

`projects/warplabs_fluids/` — 1-D compressible Euler solver, Phase 1 complete.
See `projects/warplabs_fluids/CLAUDE.md` for full context.

**Phase 1 status: COMPLETE**
- WarpEuler1D: WENO3-HLLC-RK2, 1D, float32
- JaxEuler1D: identical scheme in JAX (reference)
- 15/15 tests pass (CPU, no GPU needed)
- Sod V&V: L1(rho)=1.73e-3 at N=512, identical across JAX and Warp
- Scaling benchmark: Warp CUDA crossover vs JAX CPU at N~4096; 5.5× faster at N=32768

**Next: Phase 2**
- 2-D Euler with Strang splitting
- Kelvin-Helmholtz instability V&V
- 2-D scaling benchmark

---

## Stack

| Layer | Library |
|---|---|
| GPU kernels | [Warp](https://github.com/NVIDIA/warp) (`warp-lang`) |
| JAX reference | `jax[cpu]` (CPU only on Windows; GPU requires Linux/WSL2) |
| Numerics | numpy, scipy |
| Visualization | matplotlib (2-D plots), Rerun (3-D, future) |
| Tests | pytest — Warp CPU backend, no CUDA required |

Core install: `pip install warp-lang numpy scipy jax matplotlib`

---

## Conventions

- All units SI
- `@wp.func` = logical unit, runs in registers (no global memory I/O)
- `@wp.kernel` = memory boundary (one launch per global write point)
- 3 kernel launches × 2 RK stages = 6 launches per timestep
- Ghost cells embedded in state array (ng=2 for WENO3)
- Tests run on Warp CPU backend — no CUDA required

---

## Running

```powershell
cd projects\warplabs_fluids
python -m pytest tests/ -v                     # all 15 tests
python benchmarks\compare_sod.py               # Sod accuracy + throughput
python benchmarks\scaling_benchmark.py         # N-scaling across all backends
```
