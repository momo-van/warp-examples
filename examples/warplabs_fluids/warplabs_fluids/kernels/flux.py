import warp as wp

from .reconstruct import weno3_left, weno3_right
from .riemann import hllc_flux_1d

# One thread per interface. Q_L / Q_R stay in registers — never written to global memory.
# F[j] = flux at interface between real cells j-1 and j.
# Thread range: j = 0 .. N  (N+1 interfaces for N cells).
# Extended-array cell indices for interface j:
#   im2 = ng+j-2,  im1 = ng+j-1,  i0 = ng+j,  ip1 = ng+j+1


@wp.kernel
def compute_flux_1d(
    Q: wp.array2d(dtype=float),   # [3, N+2*ng]
    F: wp.array2d(dtype=float),   # [3, N+1]
    ng: int,
    N: int,
    gamma: float,
):
    j = wp.tid()
    if j > N:
        return

    im2 = ng + j - 2
    im1 = ng + j - 1
    i0  = ng + j
    ip1 = ng + j + 1

    # -- Load conserved variables and convert to primitives in registers --
    rho_a = Q[0, im2];  u_a = Q[1, im2] / rho_a;  p_a = (gamma - 1.0) * (Q[2, im2] - 0.5 * rho_a * u_a * u_a)
    rho_b = Q[0, im1];  u_b = Q[1, im1] / rho_b;  p_b = (gamma - 1.0) * (Q[2, im1] - 0.5 * rho_b * u_b * u_b)
    rho_c = Q[0, i0 ];  u_c = Q[1, i0 ] / rho_c;  p_c = (gamma - 1.0) * (Q[2, i0 ] - 0.5 * rho_c * u_c * u_c)
    rho_d = Q[0, ip1];  u_d = Q[1, ip1] / rho_d;  p_d = (gamma - 1.0) * (Q[2, ip1] - 0.5 * rho_d * u_d * u_d)

    # -- WENO3 reconstruction of primitive variables --
    # Left-biased (a,b,c) = (im2, im1, i0) — left state at interface j
    rho_L = weno3_left(rho_a, rho_b, rho_c)
    u_L   = weno3_left(u_a,   u_b,   u_c)
    p_L   = weno3_left(p_a,   p_b,   p_c)

    # Right-biased (b,c,d) = (im1, i0, ip1) — right state at interface j
    rho_R = weno3_right(rho_b, rho_c, rho_d)
    u_R   = weno3_right(u_b,   u_c,   u_d)
    p_R   = weno3_right(p_b,   p_c,   p_d)

    # Total energy from reconstructed primitives
    E_L = p_L / (gamma - 1.0) + 0.5 * rho_L * u_L * u_L
    E_R = p_R / (gamma - 1.0) + 0.5 * rho_R * u_R * u_R

    # -- HLLC solve → flux written to global memory --
    flux = hllc_flux_1d(rho_L, u_L, p_L, E_L, rho_R, u_R, p_R, E_R, gamma)

    F[0, j] = flux[0]
    F[1, j] = flux[1]
    F[2, j] = flux[2]
