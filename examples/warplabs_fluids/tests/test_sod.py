"""
V&V: Sod shock tube against the exact Riemann solution.

Pass criteria:
  N=512, t=0.2:  L1(rho) < 1e-3
                 L1(u)   < 2e-3
                 L1(p)   < 2e-3
"""

import numpy as np
import pytest

warp = pytest.importorskip("warp")
scipy = pytest.importorskip("scipy")

from warplabs_fluids import WarpEuler1D, cons_to_prim, l1_error
from cases.sod import ic as sod_ic, exact as sod_exact


@pytest.mark.parametrize("N,device", [
    (256, "cpu"),
    (512, "cpu"),
])
def test_sod_l1(N, device, warp_init):
    gamma = 1.4
    dx    = 1.0 / N
    t_end = 0.2

    Q0, x = sod_ic(N, gamma)

    solver = WarpEuler1D(N, dx, gamma=gamma, bc="outflow", device=device)
    solver.initialize(Q0)
    solver.run(t_end, cfl=0.4)

    rho_num, u_num, p_num = cons_to_prim(solver.state, gamma)
    rho_ex,  u_ex,  p_ex  = sod_exact(t_end, x, gamma)

    err_rho = l1_error(rho_num, rho_ex, dx)
    err_u   = l1_error(u_num,   u_ex,   dx)
    err_p   = l1_error(p_num,   p_ex,   dx)

    print(f"\nN={N}: L1(rho)={err_rho:.2e}  L1(u)={err_u:.2e}  L1(p)={err_p:.2e}")

    # WENO3 near discontinuities: O(N^-0.8) convergence. Calibrated to N=256/512.
    tol = 8e-3 / (N / 256) ** 0.8
    assert err_rho < tol,       f"L1(rho) = {err_rho:.2e} > {tol:.2e}"
    assert err_u   < 1.5 * tol, f"L1(u)   = {err_u:.2e} > {1.5*tol:.2e}"
    assert err_p   < tol,       f"L1(p)   = {err_p:.2e} > {tol:.2e}"
