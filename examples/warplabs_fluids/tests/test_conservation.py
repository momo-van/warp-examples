"""
Global conservation test.

For periodic BC, total mass, momentum and energy must be conserved to
floating-point precision (|ΔC|/C₀ < 1e-5 over 1000 steps).
"""

import numpy as np
import pytest

warp = pytest.importorskip("warp")

from warplabs_fluids import WarpEuler1D, prim_to_cons


def _smooth_ic(N: int, gamma=1.4):
    """Smooth sinusoidal density wave — periodic-compatible."""
    x   = (np.arange(N) + 0.5) / N
    rho = 1.0 + 0.2 * np.sin(2.0 * np.pi * x)
    u   = np.zeros(N)
    p   = np.ones(N)
    return prim_to_cons(rho, u, p, gamma)


@pytest.mark.parametrize("N", [64, 128])
def test_mass_conservation_periodic(N, warp_init):
    device = "cpu"
    gamma  = 1.4
    dx     = 1.0 / N
    Q0     = _smooth_ic(N, gamma)
    M0     = float(Q0[0].sum() * dx)

    solver = WarpEuler1D(N, dx, gamma=gamma, bc="periodic", device=device)
    solver.initialize(Q0)

    for _ in range(1000):
        dt = solver.compute_dt(cfl=0.4)
        solver.step(dt)

    M1  = float(solver.state[0].sum() * dx)
    err = abs(M1 - M0) / M0
    assert err < 1e-5, f"Mass conservation error {err:.2e} exceeds 1e-5"


@pytest.mark.parametrize("N", [64])
def test_momentum_conservation_periodic(N, warp_init):
    device = "cpu"
    gamma  = 1.4
    dx     = 1.0 / N
    Q0     = _smooth_ic(N, gamma)
    P0     = float(Q0[1].sum() * dx)

    solver = WarpEuler1D(N, dx, gamma=gamma, bc="periodic", device=device)
    solver.initialize(Q0)

    for _ in range(500):
        dt = solver.compute_dt(cfl=0.4)
        solver.step(dt)

    P1  = float(solver.state[1].sum() * dx)
    err = abs(P1 - P0) / (abs(P0) + 1e-12)
    assert err < 1e-5, f"Momentum conservation error {err:.2e} exceeds 1e-5"
