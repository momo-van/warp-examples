import warp as wp

# WENO3 (r=2) reconstruction. Smoothness indicators follow Jiang & Shu (1996).
# Optimal weights: d0=1/3 (wide stencil), d1=2/3 (compact stencil).

_WENO_EPS = 1.0e-6


@wp.func
def weno3_left(qm1: float, q0: float, qp1: float) -> float:
    """Left-biased reconstruction at i+1/2 from Q[i-1], Q[i], Q[i+1]."""
    p0 = -0.5 * qm1 + 1.5 * q0          # stencil {i-1, i}
    p1 =  0.5 * q0  + 0.5 * qp1         # stencil {i,   i+1}
    b0 = (q0  - qm1) * (q0  - qm1)
    b1 = (qp1 - q0 ) * (qp1 - q0 )
    a0 = (1.0 / 3.0) / ((_WENO_EPS + b0) * (_WENO_EPS + b0))
    a1 = (2.0 / 3.0) / ((_WENO_EPS + b1) * (_WENO_EPS + b1))
    return (a0 * p0 + a1 * p1) / (a0 + a1)


@wp.func
def weno3_right(q0: float, qp1: float, qp2: float) -> float:
    """Right-biased reconstruction at i+1/2 from Q[i], Q[i+1], Q[i+2]."""
    p0 =  1.5 * qp1 - 0.5 * qp2         # stencil {i+1, i+2}
    p1 =  0.5 * q0  + 0.5 * qp1         # stencil {i,   i+1}
    b0 = (qp2 - qp1) * (qp2 - qp1)
    b1 = (qp1 - q0 ) * (qp1 - q0 )
    a0 = (1.0 / 3.0) / ((_WENO_EPS + b0) * (_WENO_EPS + b0))
    a1 = (2.0 / 3.0) / ((_WENO_EPS + b1) * (_WENO_EPS + b1))
    return (a0 * p0 + a1 * p1) / (a0 + a1)
