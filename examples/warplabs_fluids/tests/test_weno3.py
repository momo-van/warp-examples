"""
Unit tests for WENO3 reconstruction.

For a smooth linear function f(x)=x, WENO3 should be exact (≤ machine eps).
For a constant field, both left and right reconstructions must equal the constant.
Near a discontinuity, weights must be dominated by the smooth stencil.
"""

import numpy as np
import warp as wp
import pytest

warp = pytest.importorskip("warp")

from warplabs_fluids.kernels.reconstruct import weno3_left, weno3_right


@wp.kernel
def _weno3_left_kernel(
    qm1: float, q0: float, qp1: float,
    out: wp.array(dtype=float),
):
    out[0] = weno3_left(qm1, q0, qp1)


@wp.kernel
def _weno3_right_kernel(
    q0: float, qp1: float, qp2: float,
    out: wp.array(dtype=float),
):
    out[0] = weno3_right(q0, qp1, qp2)


def _left(qm1, q0, qp1, device="cpu"):
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(_weno3_left_kernel, dim=1, inputs=[qm1, q0, qp1, out], device=device)
    return float(out.numpy()[0])


def _right(q0, qp1, qp2, device="cpu"):
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(_weno3_right_kernel, dim=1, inputs=[q0, qp1, qp2, out], device=device)
    return float(out.numpy()[0])


def test_constant_field(warp_init):
    """Constant field: reconstruction must return the constant exactly."""
    for c in [0.0, 1.0, -2.5]:
        assert abs(_left(c, c, c)  - c) < 1e-6
        assert abs(_right(c, c, c) - c) < 1e-6


def test_linear_field(warp_init):
    """Linear field: WENO3 is exact for polynomials of degree ≤ 2."""
    dx = 0.1
    # cells at x = -dx, 0, dx, 2*dx
    xs = np.array([-1.0, 0.0, 1.0, 2.0]) * dx
    q  = 3.0 * xs + 1.0   # f(x) = 3x + 1

    # Interface at x = 0.5*dx (between cells 1 and 2)
    exact = 3.0 * 0.5 * dx + 1.0
    assert abs(_left(q[0], q[1], q[2])  - exact) < 1e-5, "left linear"
    assert abs(_right(q[1], q[2], q[3]) - exact) < 1e-5, "right linear"


def test_discontinuity_weight_bias(warp_init):
    """
    Near a shock: the smooth stencil should dominate.
    Left-biased at a jump: qm1=1, q0=1, qp1=0 → smooth stencil is {i-1,i},
    so the result should be closer to q0=1 than the simple average 0.5*(1+0) = 0.5.
    """
    val = _left(1.0, 1.0, 0.0)
    assert val > 0.8, f"expected weight bias toward smooth side, got {val:.4f}"
