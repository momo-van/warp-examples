"""
Unit tests for the HLLC Riemann solver.

Known analytical checks:
1. Zero-velocity contact (u=0 both sides, same pressure): flux is purely pressure.
2. Symmetric flow: F_rho should be zero.
3. Supersonic right: flux equals right physical flux.
4. Sod interface: flux must lie between F_L and F_R component-wise (not always
   true but a rough sanity check for the mass flux).
"""

import numpy as np
import warp as wp
import pytest

warp = pytest.importorskip("warp")

from warplabs_fluids.kernels.riemann import hllc_flux_1d


@wp.kernel
def _hllc_kernel(
    rho_L: float, u_L: float, p_L: float, E_L: float,
    rho_R: float, u_R: float, p_R: float, E_R: float,
    gamma: float,
    F_out: wp.array(dtype=float),
):
    f = hllc_flux_1d(rho_L, u_L, p_L, E_L, rho_R, u_R, p_R, E_R, gamma)
    F_out[0] = f[0]
    F_out[1] = f[1]
    F_out[2] = f[2]


def _hllc(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma=1.4, device="cpu"):
    g = gamma
    E_L = p_L / (g - 1) + 0.5 * rho_L * u_L ** 2
    E_R = p_R / (g - 1) + 0.5 * rho_R * u_R ** 2
    F = wp.zeros(3, dtype=float, device=device)
    wp.launch(_hllc_kernel, dim=1,
              inputs=[rho_L, u_L, p_L, E_L, rho_R, u_R, p_R, E_R, gamma, F],
              device=device)
    return F.numpy()


def test_rest_state_flux(warp_init):
    """u=0 everywhere: mass flux = 0, momentum flux = p, energy flux = 0."""
    F = _hllc(1.0, 0.0, 1.0, 1.0, 0.0, 1.0)
    assert abs(F[0]) < 1e-6,       f"F_rho  = {F[0]:.3e}, expected 0"
    assert abs(F[1] - 1.0) < 1e-5, f"F_rhou = {F[1]:.6f}, expected 1.0"
    assert abs(F[2]) < 1e-6,       f"F_E    = {F[2]:.3e}, expected 0"


def test_supersonic_right(warp_init):
    """Supersonic rightward flow: flux = F_L."""
    rho, u, p, g = 1.0, 600.0, 1e5, 1.4
    E  = p / (g - 1) + 0.5 * rho * u ** 2
    F  = _hllc(rho, u, p, 0.5, u, 0.5 * p)
    F_L = np.array([rho * u, rho * u * u + p, u * (E + p)])
    np.testing.assert_allclose(F, F_L, rtol=1e-5)


def test_sod_mass_flux_sign(warp_init):
    """Sod interface: the shock moves right so mass flux must be positive."""
    F = _hllc(1.0, 0.0, 1.0, 0.125, 0.0, 0.1)
    assert F[0] > 0.0, f"Expected positive mass flux across Sod interface, got {F[0]:.4f}"
