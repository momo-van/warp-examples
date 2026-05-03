# warplabs-fluids

Experimental GPU-accelerated compressible flow solver built on [NVIDIA Warp](https://github.com/NVIDIA/warp).

**Status:** Phase 1 — 1-D compressible Euler, V&V in progress.

---

## Solver

| Component | Choice |
|---|---|
| Equations | Compressible Euler (inviscid) |
| Reconstruction | WENO3 on primitive variables (Jiang & Shu 1996) |
| Riemann solver | HLLC (Toro 2009) |
| Time integration | SSP-RK2 |
| Ghost cells | ng = 2 |
| Default CFL | 0.4 |
| Precision | float32 |

## Kernel architecture

Kernel boundaries sit at global-memory write points.
`@wp.func` functions (WENO3, HLLC) run entirely in registers inside `compute_flux_1d`.

```
bc_kernel          1 launch   fill ghost cells
compute_flux_1d    1 launch   WENO3 + HLLC → F   (Q_L/Q_R stay in registers)
update_rk_1d       1 launch   RK stage update
─────────────────────────────
3 launches × 2 RK stages = 6 launches per timestep
```

---

## Install

```powershell
pip install warp-lang numpy scipy
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

solver = WarpEuler1D(N, dx, gamma=gamma, bc="outflow")
solver.initialize(Q0)
solver.run(t_end=0.2, cfl=0.4)

rho_out = solver.state[0]
```

---

## Tests

```powershell
# From examples/warplabs_fluids/
python -m pytest tests/ -v
```

Unit tests (`test_primitives`, `test_weno3`, `test_hllc`) run on CPU — no GPU required.
V&V tests (`test_sod`, `test_conservation`) also run on CPU via Warp's LLVM backend.

---

## Roadmap

| Phase | Status | Scope |
|---|---|---|
| 1 | **active** | 1-D Euler, WENO3-HLLC, Sod V&V |
| 2 | planned | 2-D Euler, Strang splitting, Kelvin-Helmholtz |
| 3 | planned | 2-D Navier-Stokes (viscous + heat) |
| Bench | planned | Warp vs JaxFluids throughput comparison |
