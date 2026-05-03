"""Unit tests for primitive ↔ conserved variable conversions."""

import numpy as np
import warp as wp
import pytest

warp = pytest.importorskip("warp")

from warplabs_fluids.kernels.primitives import cons_to_prim_1d, prim_to_cons_1d


# Thin kernel to exercise @wp.func from a test context
@wp.kernel
def _roundtrip(
    rho_in: float, u_in: float, p_in: float,
    gamma: float,
    rho_out: wp.array(dtype=float),
    u_out:   wp.array(dtype=float),
    p_out:   wp.array(dtype=float),
):
    cons = prim_to_cons_1d(rho_in, u_in, p_in, gamma)
    prim = cons_to_prim_1d(cons[0], cons[1], cons[2], gamma)
    rho_out[0] = prim[0]
    u_out[0]   = prim[1]
    p_out[0]   = prim[2]


@pytest.mark.parametrize("rho,u,p", [
    (1.0, 0.0, 1.0),
    (0.125, 0.0, 0.1),
    (1.225, 340.0, 101325.0),
])
def test_prim_cons_roundtrip(rho, u, p, warp_init):
    device = "cpu"
    rho_out = wp.zeros(1, dtype=float, device=device)
    u_out   = wp.zeros(1, dtype=float, device=device)
    p_out   = wp.zeros(1, dtype=float, device=device)

    wp.launch(_roundtrip, dim=1,
              inputs=[rho, u, p, 1.4, rho_out, u_out, p_out],
              device=device)

    assert abs(rho_out.numpy()[0] - rho) < 1e-5, "rho roundtrip failed"
    assert abs(u_out.numpy()[0]   - u)   < 1e-4, "u roundtrip failed"
    assert abs(p_out.numpy()[0]   - p)   < 1e-2, "p roundtrip failed"


def test_energy_nonnegative(warp_init):
    """Kinetic + internal energy should both be positive."""
    device = "cpu"
    rho_out = wp.zeros(1, dtype=float, device=device)
    u_out   = wp.zeros(1, dtype=float, device=device)
    p_out   = wp.zeros(1, dtype=float, device=device)

    wp.launch(_roundtrip, dim=1,
              inputs=[0.5, 200.0, 0.5, 1.4, rho_out, u_out, p_out],
              device=device)

    assert p_out.numpy()[0] > 0.0, "pressure must be positive"
