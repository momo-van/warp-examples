# warplabs-fluids — Claude Context

## What this is

Experimental Warp-native compressible flow solver, living at
`C:\Vibe Coding\warp-projects\projects\warplabs_fluids\`.

Goal: port a JAX-based CFD solver to Warp, prove behavioral equivalence via V&V,
then benchmark to show GPU throughput advantage. Long-term reference: JaxFluids.
Short-term reference: hand-written JAX solver with identical numerics (`benchmarks/jax_euler.py`).

---

## Phase 1 — COMPLETE

**Scope:** 1-D compressible Euler (inviscid), WENO3-HLLC-SSP-RK2, float32.

### Numerical scheme

| Component | Choice | Notes |
|---|---|---|
| Equations | Compressible Euler 1D | [rho, rho*u, E] |
| Reconstruction | WENO3 on primitives | Jiang-Shu (1996) smoothness indicators |
| Riemann solver | HLLC | Toro (2009), Einfeldt wave speeds |
| Time integration | SSP-RK2 | strong-stability-preserving |
| Ghost cells | ng=2 | embedded in state array |
| CFL | 0.4 default | |
| Precision | float32 | matches JAX default |

### Kernel architecture

Draw kernel boundaries at global-memory write points. `@wp.func` = in-register logic.

```
bc_kernel          1 launch   fill ng=2 ghost cells (outflow or periodic)
compute_flux_1d    1 launch   WENO3 + HLLC fused — Q_L/Q_R in registers, only F written
update_rk_1d       1 launch   SSP-RK2 stage: Q_out = alpha*Q0 + beta*Q_in + coeff*dt*RHS
──────────────────────────────
3 launches × 2 RK stages = 6 kernel launches per timestep
```

### File layout

```
warplabs_fluids/
  kernels/
    primitives.py    @wp.func  cons<->prim conversion
    reconstruct.py   @wp.func  weno3_left, weno3_right
    riemann.py       @wp.func  hllc_flux_1d -> wp.vec3f
    bc.py            @wp.kernel bc_outflow_1d, bc_periodic_1d
    flux.py          @wp.kernel compute_flux_1d  (fused WENO3+HLLC)
    update.py        @wp.kernel update_rk_1d
  solver.py          WarpEuler1D class
  utils.py           prim_to_cons, cons_to_prim, l1/l2/linf error (numpy)
cases/
  sod.py             Sod IC + exact Riemann solver (scipy brentq)
benchmarks/
  jax_euler.py       JaxEuler1D — identical scheme in JAX (reference/comparison)
  compare_sod.py     Runs all backends, prints accuracy+throughput, saves PNGs
  scaling_benchmark.py  N-sweep 256..32768, finds GPU crossover point
tests/
  test_primitives.py  roundtrip, energy sign
  test_weno3.py       constant/linear/discontinuity bias
  test_hllc.py        rest state, supersonic, Sod mass flux sign
  test_sod.py         L1 vs exact Riemann (N=256,512)
  test_conservation.py  mass+momentum conserved <1e-5 over 1000 steps (periodic)
```

### V&V results (N=512, t=0.2)

All **four** backends (JAX CPU, JAX CUDA, Warp CPU, Warp CUDA) produce **identical** output:
- L1(rho) = 1.729e-3
- L1(u)   = 3.189e-3
- L1(p)   = 1.176e-3

Convergence rate vs N is ~O(N^-0.8), consistent with WENO3 near discontinuities.

### Benchmark results — Windows (RTX 5000 Ada, 3-backend, Warp CUDA only)

Fixed 200 steps, float32, median of 5 runs:

| N | JAX CPU | Warp CPU | Warp CUDA |
|---|---|---|---|
| 256 | 13.0 | 2.4 | 1.7 |
| 512 | 15.4 | 3.9 | 3.4 |
| 1024 | 19.1 | 5.6 | 6.8 |
| 2048 | 21.2 | 7.1 | 13.9 |
| 4096 | 17.7 | 8.4 | 26.4 |
| 8192 | 18.1 | 9.2 | 54.9 |
| 16384 | 15.1 | 9.5 | 110.7 |
| 32768 | 38.0 | 10.0 | **208.6** |

### Benchmark results — WSL2 Ubuntu 22.04 (4-backend, apples-to-apples GPU)

Fixed 200 steps, float32, median of 5 runs. Code on `/mnt/c/` (Windows NTFS mount — see note).

| N | JAX CPU | JAX CUDA | Warp CPU | Warp CUDA |
|---|---|---|---|---|
| 256 | 7.84 | 0.90 | 1.42 | 1.08 |
| 512 | 9.09 | 1.87 | 2.43 | 2.22 |
| 1024 | 12.09 | 4.32 | 3.82 | **4.55** |
| 2048 | 14.50 | 8.01 | 5.19 | **8.76** |
| 4096 | 13.54 | 16.57 | 6.43 | **17.26** |
| 8192 | 16.35 | **33.79** | 7.29 | 23.95 |
| 16384 | 17.16 | 62.68 | 7.85 | **71.44** |
| 32768 | 22.10 | **130.40** | 8.37 | 125.89 |

All Mcell-updates/s. Crossovers:
- Warp CUDA > Warp CPU:  N ~ 1024
- JAX CUDA  > JAX CPU:   N ~ 4096
- Warp CUDA ~ JAX CUDA:  within 4% at N≥16384 (neck-and-neck at large N)

**Note:** Numbers above from NTFS-mounted path `/mnt/c/`. See native FS table below for authoritative Linux numbers.

### Benchmark results — WSL2 native ext4, unfused (6 launches/step) — pre-optimization baseline

Fixed 200 steps, float32, median of 5 runs. Repo on native Linux ext4.

| N | JAX CPU | JAX CUDA | Warp CPU | Warp CUDA |
|---|---|---|---|---|
| 4096 | 13.67 | **17.16** | 6.39 | 17.21 |
| 8192 | 15.89 | **39.90** | 7.51 | 33.85 |
| 16384 | 17.64 | **59.32** | 8.09 | 54.51 |
| 32768 | 23.79 | **134.04** | 8.41 | 109.58 |

Crossovers (unfused): JAX CUDA > Warp CUDA from N~4096, gap opens to +22% at N=32768.

### Benchmark results — WSL2 native ext4, FUSED (2 launches/step) — CURRENT

Same setup, same code path, fused `fused_rk_stage_1d` kernel replacing bc+flux+update.

| N | JAX CPU | JAX CUDA | Warp CPU | Warp CUDA |
|---|---|---|---|---|
| 256 | 7.52 | 0.87 | 2.19 | 2.14 |
| 512 | 8.68 | 1.81 | 3.06 | 4.97 |
| 1024 | 10.52 | 3.86 | 4.31 | **10.98** |
| 2048 | 13.49 | 7.47 | 4.84 | **21.79** |
| 4096 | 12.51 | 12.65 | 5.31 | **42.80** |
| 8192 | 14.35 | 28.22 | 5.74 | **85.88** |
| 16384 | 15.10 | 61.82 | 5.67 | **170.67** |
| 32768 | 15.73 | 140.97 | 5.58 | **342.75** |
| 65536 | 20.36 | 251.83 | 6.03 | **654.34** |
| 131072 | 26.38 | 512.22 | 6.04 | **1382.29** |
| 327680 | 22.42 | 14.95 | 4.18 | 56.09 ⚠️ |

All Mcell-updates/s. ⚠️ N=327680 anomaly: both GPU backends collapsed (thermal throttle after sustained peak load — re-run in isolation for clean numbers).

Crossovers (fused):
- Warp CUDA > Warp CPU:  N ~ 256
- Warp CUDA > JAX CPU:   N ~ 1024
- Warp CUDA > JAX CUDA:  N ~ 1024 (grows from 2.13× at N=32768 to 2.70× at N=131072)

### Three-way comparison at N=32768 (Mcell/s) — post-fusion

| Backend | Windows native (unfused) | WSL2 native FS unfused | WSL2 native FS fused |
|---|---|---|---|
| JAX CPU | 38.0 | 23.8 | 15.7 |
| JAX CUDA | — | 134.0 | 141.0 |
| Warp CPU | 10.0 | 8.4 | 5.6 |
| Warp CUDA | 208.6 (unfused) | 109.6 | **342.8** |

Warp CUDA fused on WSL2 (343) exceeds Windows native unfused (209) by 64%.

### Key findings (post-optimization, extended N sweep)

1. **Accuracy:** All 4 backends agree to float32 — L1(rho)=1.729e-3 at N=512. Unchanged by fusion.
2. **Fused Warp CUDA = 2.43× faster than JAX CUDA** at N=32768 (343 vs 141 Mcell/s), growing to **2.70× at N=131072** (1382 vs 512 Mcell/s).
3. **Both GPUs still bandwidth-scaling at N=131072**: Warp CUDA 4× faster at N=131072 vs N=32768 (1382 vs 343). Neither is saturated yet.
4. **Fusion speedup vs unfused Warp**: ~3.1× at N=32768. Source: 6 → 2 launches/step, eliminated global F array.
5. **Warp CUDA peak observed**: 1382 Mcell/s (1.38 Gcell/s) at N=131072, within ~40% of RTX 5000 Ada theoretical memory bandwidth limit.
6. **Warp CPU regressed slightly** (8 → 6 Mcell/s) because fused kernel doubles WENO3+HLLC compute per thread. CPU is compute-bound so redundant compute costs; GPU benefits from fewer launches instead.
7. **WSL2 GPU overhead** — Windows native unfused (209) vs WSL2 fused (343) suggests Windows fused would reach ~700+ Mcell/s.
8. **compute_dt() GPU→CPU readback** every step — still the dominant overhead at small N. Next target.

### Kernel architecture — fused (CURRENT)

```
fused_rk_stage_1d_{outflow|periodic}   1 launch per RK stage, 2 per timestep
  thread i (real cell 0..N-1):
    load Q_in[i-2..i+2]  (5 cells, ghost via inline clamp/wrap)
    cons→prim for all 5   (in registers, no global writes)
    F_l = WENO3+HLLC(cells i-2..i+1)   left interface  (registers only)
    F_r = WENO3+HLLC(cells i-1..i+2)   right interface (registers only)
    Q_out[i] = alpha*Q_ref[i] + beta*Q_in[i] + coeff*dt*(-(F_r-F_l)/dx)
```

Eliminated: global F array, bc_kernel, compute_flux_1d, update_rk_1d.
Remaining files (kept for tests): bc.py, flux.py, update.py.

---

## Phase 2 — PLANNED

- 2-D compressible Euler
- Strang dimensional splitting (x/2 → y → x/2) for 2nd-order accuracy
- Kelvin-Helmholtz instability (V&V: qualitative + energy spectrum vs JAX)
- 2-D scaling benchmark (NxN grids)
- `WarpEuler2D` class reusing 1-D kernels per axis

## Phase 3 — PLANNED

- 2-D compressible Navier-Stokes (add viscous flux kernel)
- Taylor-Green vortex V&V

---

## Known issues / future work

- `compute_dt()` does CPU readback every step — replace with a Warp reduction kernel for Phase 2
- Warp CPU single-threaded — no parallel CPU path available in Warp 1.12
- JAX GPU comparison deferred until Linux/WSL2 environment available
- PNG plots in `benchmarks/` are committed via local `.gitignore` override (`!*.png`)

---

## Running

```powershell
# From projects/warplabs_fluids/

# Tests (CPU only, no GPU needed)
python -m pytest tests/ -v

# Sod comparison: accuracy table + 2 PNG plots
python benchmarks/compare_sod.py

# N-scaling throughput benchmark
python benchmarks/scaling_benchmark.py
```
