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


# ── WENO5-Z (r=3) reconstruction ──────────────────────────────────────────────
# Borges, Carmona, Costa, Don (2008). Smoothness indicators: Jiang & Shu (1996).
# Optimal weights: d0=1/10, d1=6/10, d2=3/10 (left); reversed for right.
# Z-weights: α_k = d_k * (1 + τ5 / (β_k + ε)), τ5 = |β0 - β2|.
# Float32: ε=1e-6 prevents division by zero without polluting τ5 numerics.

_W5_EPS = 1.0e-6


@wp.func
def weno5z_left(f0: float, f1: float, f2: float, f3: float, f4: float) -> float:
    """WENO5-Z left-biased reconstruction at i+1/2.
    f0=v[i-2], f1=v[i-1], f2=v[i], f3=v[i+1], f4=v[i+2]."""
    q0 = ( 2.0*f0 -  7.0*f1 + 11.0*f2) / 6.0
    q1 = (-1.0*f1 +  5.0*f2 +  2.0*f3) / 6.0
    q2 = ( 2.0*f2 +  5.0*f3 -  1.0*f4) / 6.0

    b0 = (13.0/12.0)*((f0 - 2.0*f1 + f2)*(f0 - 2.0*f1 + f2)) + \
         (1.0/4.0)*((f0 - 4.0*f1 + 3.0*f2)*(f0 - 4.0*f1 + 3.0*f2))
    b1 = (13.0/12.0)*((f1 - 2.0*f2 + f3)*(f1 - 2.0*f2 + f3)) + \
         (1.0/4.0)*((f1 - f3)*(f1 - f3))
    b2 = (13.0/12.0)*((f2 - 2.0*f3 + f4)*(f2 - 2.0*f3 + f4)) + \
         (1.0/4.0)*((3.0*f2 - 4.0*f3 + f4)*(3.0*f2 - 4.0*f3 + f4))

    tau5 = wp.abs(b0 - b2)
    a0 = 0.1 * (1.0 + tau5 / (b0 + _W5_EPS))
    a1 = 0.6 * (1.0 + tau5 / (b1 + _W5_EPS))
    a2 = 0.3 * (1.0 + tau5 / (b2 + _W5_EPS))
    return (a0*q0 + a1*q1 + a2*q2) / (a0 + a1 + a2)


@wp.func
def weno5z_right(f0: float, f1: float, f2: float, f3: float, f4: float) -> float:
    """WENO5-Z right-biased reconstruction at i+1/2.
    f0=v[i-1], f1=v[i], f2=v[i+1], f3=v[i+2], f4=v[i+3]."""
    q0 = (11.0*f2 -  7.0*f3 +  2.0*f4) / 6.0
    q1 = ( 2.0*f1 +  5.0*f2 -  1.0*f3) / 6.0
    q2 = (-1.0*f0 +  5.0*f1 +  2.0*f2) / 6.0

    b0 = (13.0/12.0)*((f2 - 2.0*f3 + f4)*(f2 - 2.0*f3 + f4)) + \
         (1.0/4.0)*((3.0*f2 - 4.0*f3 + f4)*(3.0*f2 - 4.0*f3 + f4))
    b1 = (13.0/12.0)*((f1 - 2.0*f2 + f3)*(f1 - 2.0*f2 + f3)) + \
         (1.0/4.0)*((f1 - f3)*(f1 - f3))
    b2 = (13.0/12.0)*((f0 - 2.0*f1 + f2)*(f0 - 2.0*f1 + f2)) + \
         (1.0/4.0)*((f0 - 4.0*f1 + 3.0*f2)*(f0 - 4.0*f1 + 3.0*f2))

    tau5 = wp.abs(b0 - b2)
    a0 = 0.3 * (1.0 + tau5 / (b0 + _W5_EPS))
    a1 = 0.6 * (1.0 + tau5 / (b1 + _W5_EPS))
    a2 = 0.1 * (1.0 + tau5 / (b2 + _W5_EPS))
    return (a0*q0 + a1*q1 + a2*q2) / (a0 + a1 + a2)


# ── WENO5-Z float64 variants ───────────────────────────────────────────────────

@wp.func
def weno5z_left_f64(f0: wp.float64, f1: wp.float64, f2: wp.float64, f3: wp.float64, f4: wp.float64) -> wp.float64:
    eps = wp.float64(1.0e-36)
    q0 = (wp.float64( 2.0)*f0 - wp.float64( 7.0)*f1 + wp.float64(11.0)*f2) / wp.float64(6.0)
    q1 = (wp.float64(-1.0)*f1 + wp.float64( 5.0)*f2 + wp.float64( 2.0)*f3) / wp.float64(6.0)
    q2 = (wp.float64( 2.0)*f2 + wp.float64( 5.0)*f3 - wp.float64( 1.0)*f4) / wp.float64(6.0)
    b0 = (wp.float64(13.0)/wp.float64(12.0))*((f0-wp.float64(2.0)*f1+f2)*(f0-wp.float64(2.0)*f1+f2)) + \
         (wp.float64(1.0)/wp.float64(4.0))*((f0-wp.float64(4.0)*f1+wp.float64(3.0)*f2)*(f0-wp.float64(4.0)*f1+wp.float64(3.0)*f2))
    b1 = (wp.float64(13.0)/wp.float64(12.0))*((f1-wp.float64(2.0)*f2+f3)*(f1-wp.float64(2.0)*f2+f3)) + \
         (wp.float64(1.0)/wp.float64(4.0))*((f1-f3)*(f1-f3))
    b2 = (wp.float64(13.0)/wp.float64(12.0))*((f2-wp.float64(2.0)*f3+f4)*(f2-wp.float64(2.0)*f3+f4)) + \
         (wp.float64(1.0)/wp.float64(4.0))*((wp.float64(3.0)*f2-wp.float64(4.0)*f3+f4)*(wp.float64(3.0)*f2-wp.float64(4.0)*f3+f4))
    tau5 = wp.abs(b0 - b2)
    a0 = wp.float64(0.1) * (wp.float64(1.0) + tau5 / (b0 + eps))
    a1 = wp.float64(0.6) * (wp.float64(1.0) + tau5 / (b1 + eps))
    a2 = wp.float64(0.3) * (wp.float64(1.0) + tau5 / (b2 + eps))
    return (a0*q0 + a1*q1 + a2*q2) / (a0 + a1 + a2)


@wp.func
def weno5z_right_f64(f0: wp.float64, f1: wp.float64, f2: wp.float64, f3: wp.float64, f4: wp.float64) -> wp.float64:
    eps = wp.float64(1.0e-36)
    q0 = (wp.float64(11.0)*f2 - wp.float64( 7.0)*f3 + wp.float64(2.0)*f4) / wp.float64(6.0)
    q1 = (wp.float64( 2.0)*f1 + wp.float64( 5.0)*f2 - wp.float64(1.0)*f3) / wp.float64(6.0)
    q2 = (wp.float64(-1.0)*f0 + wp.float64( 5.0)*f1 + wp.float64(2.0)*f2) / wp.float64(6.0)
    b0 = (wp.float64(13.0)/wp.float64(12.0))*((f2-wp.float64(2.0)*f3+f4)*(f2-wp.float64(2.0)*f3+f4)) + \
         (wp.float64(1.0)/wp.float64(4.0))*((wp.float64(3.0)*f2-wp.float64(4.0)*f3+f4)*(wp.float64(3.0)*f2-wp.float64(4.0)*f3+f4))
    b1 = (wp.float64(13.0)/wp.float64(12.0))*((f1-wp.float64(2.0)*f2+f3)*(f1-wp.float64(2.0)*f2+f3)) + \
         (wp.float64(1.0)/wp.float64(4.0))*((f1-f3)*(f1-f3))
    b2 = (wp.float64(13.0)/wp.float64(12.0))*((f0-wp.float64(2.0)*f1+f2)*(f0-wp.float64(2.0)*f1+f2)) + \
         (wp.float64(1.0)/wp.float64(4.0))*((f0-wp.float64(4.0)*f1+wp.float64(3.0)*f2)*(f0-wp.float64(4.0)*f1+wp.float64(3.0)*f2))
    tau5 = wp.abs(b0 - b2)
    a0 = wp.float64(0.3) * (wp.float64(1.0) + tau5 / (b0 + eps))
    a1 = wp.float64(0.6) * (wp.float64(1.0) + tau5 / (b1 + eps))
    a2 = wp.float64(0.1) * (wp.float64(1.0) + tau5 / (b2 + eps))
    return (a0*q0 + a1*q1 + a2*q2) / (a0 + a1 + a2)
