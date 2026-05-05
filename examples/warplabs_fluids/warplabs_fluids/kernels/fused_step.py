import warp as wp

from .reconstruct import weno3_left, weno3_right, weno5z_left, weno5z_right, weno5z_left_f64, weno5z_right_f64
from .riemann import hllc_flux_1d, hllc_flux_1d_f64

# Fused bc+WENO3+HLLC+RK-stage kernel — 1 launch per RK stage (2 per step).
#
# Design vs the original 3-kernel-per-stage approach:
#   OLD: bc_kernel → compute_flux_1d → update_rk_1d   (3 launches, global F array)
#   NEW: fused_rk_stage_1d                             (1 launch, no global F array)
#
# Each thread owns one real cell. It computes the two interface fluxes that
# bound its cell (F_l at i-1/2, F_r at i+1/2) entirely in registers using a
# 5-cell stencil, then writes the RK-stage result directly to Q_out.
# Ghost-cell accesses are remapped inline via clamping (outflow) or modular
# arithmetic (periodic) — no separate bc launch needed.
#
# Memory traffic per RK stage vs old design:
#   OLD: Q read twice (flux + update) + F written + F read = 4 array passes
#   NEW: Q_in read once + Q_ref read once + Q_out written = 3 array passes
#
# RK update formula (same coefficients as update_rk_1d):
#   Q_out[i] = alpha*Q_ref[i] + beta*Q_in[i] + coeff*dt*(-(F_r - F_l)/dx)
#   Stage 1: alpha=1, beta=0, coeff=1   (Q_in = Q_ref = Q0 → Q1 = Q0 + dt*L(Q0))
#   Stage 2: alpha=0.5, beta=0.5, coeff=0.5  (Q_in=Q1, Q_ref=Q0 → Q_final)


@wp.func
def _flux_at_interface(
    rho_a: float, u_a: float, p_a: float,
    rho_b: float, u_b: float, p_b: float,
    rho_c: float, u_c: float, p_c: float,
    rho_d: float, u_d: float, p_d: float,
    gamma: float,
) -> wp.vec3f:
    """
    HLLC flux at interface b+1/2 from 4-cell stencil (a, b, c, d).
      Q_L = weno3_left (a, b, c)   left-biased reconstruction
      Q_R = weno3_right(b, c, d)   right-biased reconstruction
    All 12 primitive values stay in registers — no global writes.
    """
    rho_L = weno3_left(rho_a, rho_b, rho_c)
    u_L   = weno3_left(u_a,   u_b,   u_c)
    p_L   = weno3_left(p_a,   p_b,   p_c)
    rho_R = weno3_right(rho_b, rho_c, rho_d)
    u_R   = weno3_right(u_b,   u_c,   u_d)
    p_R   = weno3_right(p_b,   p_c,   p_d)
    E_L   = p_L / (gamma - 1.0) + 0.5 * rho_L * u_L * u_L
    E_R   = p_R / (gamma - 1.0) + 0.5 * rho_R * u_R * u_R
    return hllc_flux_1d(rho_L, u_L, p_L, E_L, rho_R, u_R, p_R, E_R, gamma)


@wp.kernel
def fused_rk_stage_1d_outflow(
    Q_in:  wp.array2d(dtype=float),   # spatial-op input   (3, N+2*ng)
    Q_ref: wp.array2d(dtype=float),   # RK reference state (3, N+2*ng)
    Q_out: wp.array2d(dtype=float),   # output             (3, N+2*ng)
    ng: int, N: int, gamma: float,
    dt: float, dx: float,
    alpha: float, beta: float, coeff: float,
):
    i  = wp.tid()   # real-cell index, 0 .. N-1
    i0 = ng + i

    # Stencil indices — out-of-bounds clamped to nearest real cell (outflow)
    ia  = wp.clamp(i0 - 2, ng, ng + N - 1)
    ib  = wp.clamp(i0 - 1, ng, ng + N - 1)
    # ic = i0 always in [ng, ng+N-1]
    id_ = wp.clamp(i0 + 1, ng, ng + N - 1)
    ie  = wp.clamp(i0 + 2, ng, ng + N - 1)

    # Conserved → primitives for 5 stencil cells (all in registers)
    rho_a_c = Q_in[0, ia];  rho_a = rho_a_c;  u_a = Q_in[1, ia] / rho_a;  p_a = (gamma - 1.0) * (Q_in[2, ia] - 0.5 * rho_a * u_a * u_a)
    rho_b_c = Q_in[0, ib];  rho_b = rho_b_c;  u_b = Q_in[1, ib] / rho_b;  p_b = (gamma - 1.0) * (Q_in[2, ib] - 0.5 * rho_b * u_b * u_b)
    rho_c_c = Q_in[0, i0];  rhou_c = Q_in[1, i0]; E_c = Q_in[2, i0]
    rho_c = rho_c_c;  u_c = rhou_c / rho_c;  p_c = (gamma - 1.0) * (E_c - 0.5 * rho_c * u_c * u_c)
    rho_d_c = Q_in[0, id_]; rho_d = rho_d_c; u_d = Q_in[1, id_] / rho_d; p_d = (gamma - 1.0) * (Q_in[2, id_] - 0.5 * rho_d * u_d * u_d)
    rho_e_c = Q_in[0, ie];  rho_e = rho_e_c; u_e = Q_in[1, ie] / rho_e;  p_e = (gamma - 1.0) * (Q_in[2, ie] - 0.5 * rho_e * u_e * u_e)

    # Interface fluxes — both computed from register-resident primitives, never written globally
    F_l = _flux_at_interface(rho_a, u_a, p_a, rho_b, u_b, p_b, rho_c, u_c, p_c, rho_d, u_d, p_d, gamma)
    F_r = _flux_at_interface(rho_b, u_b, p_b, rho_c, u_c, p_c, rho_d, u_d, p_d, rho_e, u_e, p_e, gamma)

    # SSP-RK stage update — one global write per variable
    inv_dx = 1.0 / dx
    Q_out[0, i0] = alpha * Q_ref[0, i0] + beta * rho_c_c  + coeff * dt * (-(F_r[0] - F_l[0]) * inv_dx)
    Q_out[1, i0] = alpha * Q_ref[1, i0] + beta * rhou_c   + coeff * dt * (-(F_r[1] - F_l[1]) * inv_dx)
    Q_out[2, i0] = alpha * Q_ref[2, i0] + beta * E_c      + coeff * dt * (-(F_r[2] - F_l[2]) * inv_dx)


@wp.kernel
def fused_rk_stage_1d_periodic(
    Q_in:  wp.array2d(dtype=float),
    Q_ref: wp.array2d(dtype=float),
    Q_out: wp.array2d(dtype=float),
    ng: int, N: int, gamma: float,
    dt: float, dx: float,
    alpha: float, beta: float, coeff: float,
):
    i  = wp.tid()
    i0 = ng + i

    # Periodic wrapping: Warp uses C-style %, so shift by N before taking % to avoid negative results
    ia  = ng + (i - 2 + N) % N
    ib  = ng + (i - 1 + N) % N
    id_ = ng + (i + 1) % N
    ie  = ng + (i + 2) % N

    rho_a_c = Q_in[0, ia];  rho_a = rho_a_c;  u_a = Q_in[1, ia] / rho_a;  p_a = (gamma - 1.0) * (Q_in[2, ia] - 0.5 * rho_a * u_a * u_a)
    rho_b_c = Q_in[0, ib];  rho_b = rho_b_c;  u_b = Q_in[1, ib] / rho_b;  p_b = (gamma - 1.0) * (Q_in[2, ib] - 0.5 * rho_b * u_b * u_b)
    rho_c_c = Q_in[0, i0];  rhou_c = Q_in[1, i0]; E_c = Q_in[2, i0]
    rho_c = rho_c_c;  u_c = rhou_c / rho_c;  p_c = (gamma - 1.0) * (E_c - 0.5 * rho_c * u_c * u_c)
    rho_d_c = Q_in[0, id_]; rho_d = rho_d_c; u_d = Q_in[1, id_] / rho_d; p_d = (gamma - 1.0) * (Q_in[2, id_] - 0.5 * rho_d * u_d * u_d)
    rho_e_c = Q_in[0, ie];  rho_e = rho_e_c; u_e = Q_in[1, ie] / rho_e;  p_e = (gamma - 1.0) * (Q_in[2, ie] - 0.5 * rho_e * u_e * u_e)

    F_l = _flux_at_interface(rho_a, u_a, p_a, rho_b, u_b, p_b, rho_c, u_c, p_c, rho_d, u_d, p_d, gamma)
    F_r = _flux_at_interface(rho_b, u_b, p_b, rho_c, u_c, p_c, rho_d, u_d, p_d, rho_e, u_e, p_e, gamma)

    inv_dx = 1.0 / dx
    Q_out[0, i0] = alpha * Q_ref[0, i0] + beta * rho_c_c  + coeff * dt * (-(F_r[0] - F_l[0]) * inv_dx)
    Q_out[1, i0] = alpha * Q_ref[1, i0] + beta * rhou_c   + coeff * dt * (-(F_r[1] - F_l[1]) * inv_dx)
    Q_out[2, i0] = alpha * Q_ref[2, i0] + beta * E_c      + coeff * dt * (-(F_r[2] - F_l[2]) * inv_dx)


# ── WENO5-Z + SSP-RK3 fused kernels ───────────────────────────────────────────
# 7-cell stencil (i-3..i+3), ng=3.
# SSP-RK3 coefficients (Shu-Osher 1988):
#   Stage 1: alpha=1,   beta=0,   coeff=1     Q1 = Q0 + dt*L(Q0)
#   Stage 2: alpha=3/4, beta=1/4, coeff=1/4   Q2 = 3/4*Q0 + 1/4*Q1 + 1/4*dt*L(Q1)
#   Stage 3: alpha=1/3, beta=2/3, coeff=2/3   Q3 = 1/3*Q0 + 2/3*Q2 + 2/3*dt*L(Q2)


@wp.func
def _flux_at_interface_w5z(
    rho_a: float, u_a: float, p_a: float,
    rho_b: float, u_b: float, p_b: float,
    rho_c: float, u_c: float, p_c: float,
    rho_d: float, u_d: float, p_d: float,
    rho_e: float, u_e: float, p_e: float,
    rho_f: float, u_f: float, p_f: float,
    gamma: float,
) -> wp.vec3f:
    """
    HLLC flux at interface c+1/2 from 6-cell stencil (a, b, c, d, e, f).
      Q_L = weno5z_left (a, b, c, d, e)   left-biased  reconstruction at c+1/2
      Q_R = weno5z_right(b, c, d, e, f)   right-biased reconstruction at c+1/2
    """
    rho_L = weno5z_left(rho_a, rho_b, rho_c, rho_d, rho_e)
    u_L   = weno5z_left(u_a,   u_b,   u_c,   u_d,   u_e)
    p_L   = weno5z_left(p_a,   p_b,   p_c,   p_d,   p_e)
    rho_R = weno5z_right(rho_b, rho_c, rho_d, rho_e, rho_f)
    u_R   = weno5z_right(u_b,   u_c,   u_d,   u_e,   u_f)
    p_R   = weno5z_right(p_b,   p_c,   p_d,   p_e,   p_f)
    E_L   = p_L / (gamma - 1.0) + 0.5 * rho_L * u_L * u_L
    E_R   = p_R / (gamma - 1.0) + 0.5 * rho_R * u_R * u_R
    return hllc_flux_1d(rho_L, u_L, p_L, E_L, rho_R, u_R, p_R, E_R, gamma)


@wp.kernel
def fused_rk_stage_1d_outflow_w5z(
    Q_in:  wp.array2d(dtype=float),   # (3, N+2*ng), ng=3
    Q_ref: wp.array2d(dtype=float),
    Q_out: wp.array2d(dtype=float),
    ng: int, N: int, gamma: float,
    dt: float, dx: float,
    alpha: float, beta: float, coeff: float,
):
    i  = wp.tid()   # real-cell index, 0 .. N-1
    i0 = ng + i

    # 7-cell stencil clamped to real domain (outflow BC)
    im3 = wp.clamp(i0 - 3, ng, ng + N - 1)
    im2 = wp.clamp(i0 - 2, ng, ng + N - 1)
    im1 = wp.clamp(i0 - 1, ng, ng + N - 1)
    ip1 = wp.clamp(i0 + 1, ng, ng + N - 1)
    ip2 = wp.clamp(i0 + 2, ng, ng + N - 1)
    ip3 = wp.clamp(i0 + 3, ng, ng + N - 1)

    # Load conserved → primitives for all 7 cells (registers only)
    rho_m3 = Q_in[0, im3]; u_m3 = Q_in[1, im3] / rho_m3; p_m3 = (gamma - 1.0) * (Q_in[2, im3] - 0.5 * rho_m3 * u_m3 * u_m3)
    rho_m2 = Q_in[0, im2]; u_m2 = Q_in[1, im2] / rho_m2; p_m2 = (gamma - 1.0) * (Q_in[2, im2] - 0.5 * rho_m2 * u_m2 * u_m2)
    rho_m1 = Q_in[0, im1]; u_m1 = Q_in[1, im1] / rho_m1; p_m1 = (gamma - 1.0) * (Q_in[2, im1] - 0.5 * rho_m1 * u_m1 * u_m1)
    rho_c  = Q_in[0, i0];  rhou_c = Q_in[1, i0]; E_c = Q_in[2, i0]
    u_c  = rhou_c / rho_c;  p_c = (gamma - 1.0) * (E_c - 0.5 * rho_c * u_c * u_c)
    rho_p1 = Q_in[0, ip1]; u_p1 = Q_in[1, ip1] / rho_p1; p_p1 = (gamma - 1.0) * (Q_in[2, ip1] - 0.5 * rho_p1 * u_p1 * u_p1)
    rho_p2 = Q_in[0, ip2]; u_p2 = Q_in[1, ip2] / rho_p2; p_p2 = (gamma - 1.0) * (Q_in[2, ip2] - 0.5 * rho_p2 * u_p2 * u_p2)
    rho_p3 = Q_in[0, ip3]; u_p3 = Q_in[1, ip3] / rho_p3; p_p3 = (gamma - 1.0) * (Q_in[2, ip3] - 0.5 * rho_p3 * u_p3 * u_p3)

    # F_l at i-1/2: stencil (i-3, i-2, i-1, i, i+1, i+2)
    F_l = _flux_at_interface_w5z(
        rho_m3, u_m3, p_m3,
        rho_m2, u_m2, p_m2,
        rho_m1, u_m1, p_m1,
        rho_c,  u_c,  p_c,
        rho_p1, u_p1, p_p1,
        rho_p2, u_p2, p_p2,
        gamma)

    # F_r at i+1/2: stencil (i-2, i-1, i, i+1, i+2, i+3)
    F_r = _flux_at_interface_w5z(
        rho_m2, u_m2, p_m2,
        rho_m1, u_m1, p_m1,
        rho_c,  u_c,  p_c,
        rho_p1, u_p1, p_p1,
        rho_p2, u_p2, p_p2,
        rho_p3, u_p3, p_p3,
        gamma)

    inv_dx = 1.0 / dx
    Q_out[0, i0] = alpha * Q_ref[0, i0] + beta * rho_c  + coeff * dt * (-(F_r[0] - F_l[0]) * inv_dx)
    Q_out[1, i0] = alpha * Q_ref[1, i0] + beta * rhou_c + coeff * dt * (-(F_r[1] - F_l[1]) * inv_dx)
    Q_out[2, i0] = alpha * Q_ref[2, i0] + beta * E_c    + coeff * dt * (-(F_r[2] - F_l[2]) * inv_dx)


@wp.kernel
def fused_rk_stage_1d_periodic_w5z(
    Q_in:  wp.array2d(dtype=float),
    Q_ref: wp.array2d(dtype=float),
    Q_out: wp.array2d(dtype=float),
    ng: int, N: int, gamma: float,
    dt: float, dx: float,
    alpha: float, beta: float, coeff: float,
):
    i  = wp.tid()
    i0 = ng + i

    im3 = ng + (i - 3 + N) % N
    im2 = ng + (i - 2 + N) % N
    im1 = ng + (i - 1 + N) % N
    ip1 = ng + (i + 1) % N
    ip2 = ng + (i + 2) % N
    ip3 = ng + (i + 3) % N

    rho_m3 = Q_in[0, im3]; u_m3 = Q_in[1, im3] / rho_m3; p_m3 = (gamma - 1.0) * (Q_in[2, im3] - 0.5 * rho_m3 * u_m3 * u_m3)
    rho_m2 = Q_in[0, im2]; u_m2 = Q_in[1, im2] / rho_m2; p_m2 = (gamma - 1.0) * (Q_in[2, im2] - 0.5 * rho_m2 * u_m2 * u_m2)
    rho_m1 = Q_in[0, im1]; u_m1 = Q_in[1, im1] / rho_m1; p_m1 = (gamma - 1.0) * (Q_in[2, im1] - 0.5 * rho_m1 * u_m1 * u_m1)
    rho_c  = Q_in[0, i0];  rhou_c = Q_in[1, i0]; E_c = Q_in[2, i0]
    u_c  = rhou_c / rho_c;  p_c = (gamma - 1.0) * (E_c - 0.5 * rho_c * u_c * u_c)
    rho_p1 = Q_in[0, ip1]; u_p1 = Q_in[1, ip1] / rho_p1; p_p1 = (gamma - 1.0) * (Q_in[2, ip1] - 0.5 * rho_p1 * u_p1 * u_p1)
    rho_p2 = Q_in[0, ip2]; u_p2 = Q_in[1, ip2] / rho_p2; p_p2 = (gamma - 1.0) * (Q_in[2, ip2] - 0.5 * rho_p2 * u_p2 * u_p2)
    rho_p3 = Q_in[0, ip3]; u_p3 = Q_in[1, ip3] / rho_p3; p_p3 = (gamma - 1.0) * (Q_in[2, ip3] - 0.5 * rho_p3 * u_p3 * u_p3)

    F_l = _flux_at_interface_w5z(
        rho_m3, u_m3, p_m3,
        rho_m2, u_m2, p_m2,
        rho_m1, u_m1, p_m1,
        rho_c,  u_c,  p_c,
        rho_p1, u_p1, p_p1,
        rho_p2, u_p2, p_p2,
        gamma)

    F_r = _flux_at_interface_w5z(
        rho_m2, u_m2, p_m2,
        rho_m1, u_m1, p_m1,
        rho_c,  u_c,  p_c,
        rho_p1, u_p1, p_p1,
        rho_p2, u_p2, p_p2,
        rho_p3, u_p3, p_p3,
        gamma)

    inv_dx = 1.0 / dx
    Q_out[0, i0] = alpha * Q_ref[0, i0] + beta * rho_c  + coeff * dt * (-(F_r[0] - F_l[0]) * inv_dx)
    Q_out[1, i0] = alpha * Q_ref[1, i0] + beta * rhou_c + coeff * dt * (-(F_r[1] - F_l[1]) * inv_dx)
    Q_out[2, i0] = alpha * Q_ref[2, i0] + beta * E_c    + coeff * dt * (-(F_r[2] - F_l[2]) * inv_dx)


# ── WENO5-Z + SSP-RK3 float64 fused kernels ───────────────────────────────────

@wp.func
def _flux_at_interface_w5z_f64(
    rho_a: wp.float64, u_a: wp.float64, p_a: wp.float64,
    rho_b: wp.float64, u_b: wp.float64, p_b: wp.float64,
    rho_c: wp.float64, u_c: wp.float64, p_c: wp.float64,
    rho_d: wp.float64, u_d: wp.float64, p_d: wp.float64,
    rho_e: wp.float64, u_e: wp.float64, p_e: wp.float64,
    rho_f: wp.float64, u_f: wp.float64, p_f: wp.float64,
    gamma: wp.float64,
) -> wp.vec3d:
    rho_L = weno5z_left_f64(rho_a, rho_b, rho_c, rho_d, rho_e)
    u_L   = weno5z_left_f64(u_a,   u_b,   u_c,   u_d,   u_e)
    p_L   = weno5z_left_f64(p_a,   p_b,   p_c,   p_d,   p_e)
    rho_R = weno5z_right_f64(rho_b, rho_c, rho_d, rho_e, rho_f)
    u_R   = weno5z_right_f64(u_b,   u_c,   u_d,   u_e,   u_f)
    p_R   = weno5z_right_f64(p_b,   p_c,   p_d,   p_e,   p_f)
    gm1   = gamma - wp.float64(1.0)
    E_L   = p_L / gm1 + wp.float64(0.5) * rho_L * u_L * u_L
    E_R   = p_R / gm1 + wp.float64(0.5) * rho_R * u_R * u_R
    return hllc_flux_1d_f64(rho_L, u_L, p_L, E_L, rho_R, u_R, p_R, E_R, gamma)


@wp.kernel
def fused_rk_stage_1d_outflow_w5z_f64(
    Q_in:  wp.array2d(dtype=wp.float64),
    Q_ref: wp.array2d(dtype=wp.float64),
    Q_out: wp.array2d(dtype=wp.float64),
    ng: int, N: int, gamma: wp.float64,
    dt: wp.float64, dx: wp.float64,
    alpha: wp.float64, beta: wp.float64, coeff: wp.float64,
):
    i  = wp.tid()
    i0 = ng + i

    im3 = wp.clamp(i0 - 3, ng, ng + N - 1)
    im2 = wp.clamp(i0 - 2, ng, ng + N - 1)
    im1 = wp.clamp(i0 - 1, ng, ng + N - 1)
    ip1 = wp.clamp(i0 + 1, ng, ng + N - 1)
    ip2 = wp.clamp(i0 + 2, ng, ng + N - 1)
    ip3 = wp.clamp(i0 + 3, ng, ng + N - 1)

    gm1  = gamma - wp.float64(1.0)
    half = wp.float64(0.5)

    rho_m3 = Q_in[0, im3]; u_m3 = Q_in[1, im3] / rho_m3; p_m3 = gm1 * (Q_in[2, im3] - half * rho_m3 * u_m3 * u_m3)
    rho_m2 = Q_in[0, im2]; u_m2 = Q_in[1, im2] / rho_m2; p_m2 = gm1 * (Q_in[2, im2] - half * rho_m2 * u_m2 * u_m2)
    rho_m1 = Q_in[0, im1]; u_m1 = Q_in[1, im1] / rho_m1; p_m1 = gm1 * (Q_in[2, im1] - half * rho_m1 * u_m1 * u_m1)
    rho_c  = Q_in[0, i0];  rhou_c = Q_in[1, i0]; E_c = Q_in[2, i0]
    u_c  = rhou_c / rho_c;  p_c = gm1 * (E_c - half * rho_c * u_c * u_c)
    rho_p1 = Q_in[0, ip1]; u_p1 = Q_in[1, ip1] / rho_p1; p_p1 = gm1 * (Q_in[2, ip1] - half * rho_p1 * u_p1 * u_p1)
    rho_p2 = Q_in[0, ip2]; u_p2 = Q_in[1, ip2] / rho_p2; p_p2 = gm1 * (Q_in[2, ip2] - half * rho_p2 * u_p2 * u_p2)
    rho_p3 = Q_in[0, ip3]; u_p3 = Q_in[1, ip3] / rho_p3; p_p3 = gm1 * (Q_in[2, ip3] - half * rho_p3 * u_p3 * u_p3)

    F_l = _flux_at_interface_w5z_f64(
        rho_m3, u_m3, p_m3, rho_m2, u_m2, p_m2, rho_m1, u_m1, p_m1,
        rho_c,  u_c,  p_c,  rho_p1, u_p1, p_p1, rho_p2, u_p2, p_p2, gamma)

    F_r = _flux_at_interface_w5z_f64(
        rho_m2, u_m2, p_m2, rho_m1, u_m1, p_m1, rho_c,  u_c,  p_c,
        rho_p1, u_p1, p_p1, rho_p2, u_p2, p_p2, rho_p3, u_p3, p_p3, gamma)

    inv_dx = wp.float64(1.0) / dx
    Q_out[0, i0] = alpha * Q_ref[0, i0] + beta * rho_c  + coeff * dt * (-(F_r[0] - F_l[0]) * inv_dx)
    Q_out[1, i0] = alpha * Q_ref[1, i0] + beta * rhou_c + coeff * dt * (-(F_r[1] - F_l[1]) * inv_dx)
    Q_out[2, i0] = alpha * Q_ref[2, i0] + beta * E_c    + coeff * dt * (-(F_r[2] - F_l[2]) * inv_dx)


@wp.kernel
def fused_rk_stage_1d_periodic_w5z_f64(
    Q_in:  wp.array2d(dtype=wp.float64),
    Q_ref: wp.array2d(dtype=wp.float64),
    Q_out: wp.array2d(dtype=wp.float64),
    ng: int, N: int, gamma: wp.float64,
    dt: wp.float64, dx: wp.float64,
    alpha: wp.float64, beta: wp.float64, coeff: wp.float64,
):
    i  = wp.tid()
    i0 = ng + i

    im3 = ng + (i - 3 + N) % N
    im2 = ng + (i - 2 + N) % N
    im1 = ng + (i - 1 + N) % N
    ip1 = ng + (i + 1) % N
    ip2 = ng + (i + 2) % N
    ip3 = ng + (i + 3) % N

    gm1  = gamma - wp.float64(1.0)
    half = wp.float64(0.5)

    rho_m3 = Q_in[0, im3]; u_m3 = Q_in[1, im3] / rho_m3; p_m3 = gm1 * (Q_in[2, im3] - half * rho_m3 * u_m3 * u_m3)
    rho_m2 = Q_in[0, im2]; u_m2 = Q_in[1, im2] / rho_m2; p_m2 = gm1 * (Q_in[2, im2] - half * rho_m2 * u_m2 * u_m2)
    rho_m1 = Q_in[0, im1]; u_m1 = Q_in[1, im1] / rho_m1; p_m1 = gm1 * (Q_in[2, im1] - half * rho_m1 * u_m1 * u_m1)
    rho_c  = Q_in[0, i0];  rhou_c = Q_in[1, i0]; E_c = Q_in[2, i0]
    u_c  = rhou_c / rho_c;  p_c = gm1 * (E_c - half * rho_c * u_c * u_c)
    rho_p1 = Q_in[0, ip1]; u_p1 = Q_in[1, ip1] / rho_p1; p_p1 = gm1 * (Q_in[2, ip1] - half * rho_p1 * u_p1 * u_p1)
    rho_p2 = Q_in[0, ip2]; u_p2 = Q_in[1, ip2] / rho_p2; p_p2 = gm1 * (Q_in[2, ip2] - half * rho_p2 * u_p2 * u_p2)
    rho_p3 = Q_in[0, ip3]; u_p3 = Q_in[1, ip3] / rho_p3; p_p3 = gm1 * (Q_in[2, ip3] - half * rho_p3 * u_p3 * u_p3)

    F_l = _flux_at_interface_w5z_f64(
        rho_m3, u_m3, p_m3, rho_m2, u_m2, p_m2, rho_m1, u_m1, p_m1,
        rho_c,  u_c,  p_c,  rho_p1, u_p1, p_p1, rho_p2, u_p2, p_p2, gamma)

    F_r = _flux_at_interface_w5z_f64(
        rho_m2, u_m2, p_m2, rho_m1, u_m1, p_m1, rho_c,  u_c,  p_c,
        rho_p1, u_p1, p_p1, rho_p2, u_p2, p_p2, rho_p3, u_p3, p_p3, gamma)

    inv_dx = wp.float64(1.0) / dx
    Q_out[0, i0] = alpha * Q_ref[0, i0] + beta * rho_c  + coeff * dt * (-(F_r[0] - F_l[0]) * inv_dx)
    Q_out[1, i0] = alpha * Q_ref[1, i0] + beta * rhou_c + coeff * dt * (-(F_r[1] - F_l[1]) * inv_dx)
    Q_out[2, i0] = alpha * Q_ref[2, i0] + beta * E_c    + coeff * dt * (-(F_r[2] - F_l[2]) * inv_dx)
