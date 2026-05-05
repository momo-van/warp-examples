import warp as wp

# HLLC Riemann solver for 1D compressible Euler.
# Wave speed estimates: simple Einfeldt bounds (u ± a).
# Star-state formula from Toro (2009), Chapter 10.


@wp.func
def hllc_flux_1d(
    rho_L: float, u_L: float, p_L: float, E_L: float,
    rho_R: float, u_R: float, p_R: float, E_R: float,
    gamma: float,
) -> wp.vec3f:
    """Returns numerical flux (F_rho, F_rhou, F_E) at an interface."""

    a_L = wp.sqrt(gamma * p_L / rho_L)
    a_R = wp.sqrt(gamma * p_R / rho_R)

    # Wave speed bounds
    S_L = wp.min(u_L - a_L, u_R - a_R)
    S_R = wp.max(u_L + a_L, u_R + a_R)

    # Contact (middle) wave speed
    num   = p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)
    denom = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    S_star = num / denom

    # Physical fluxes
    F_L = wp.vec3f(rho_L * u_L,
                   rho_L * u_L * u_L + p_L,
                   u_L * (E_L + p_L))
    F_R = wp.vec3f(rho_R * u_R,
                   rho_R * u_R * u_R + p_R,
                   u_R * (E_R + p_R))

    if S_L >= 0.0:
        return F_L

    if S_R <= 0.0:
        return F_R

    if S_star >= 0.0:
        chi   = rho_L * (S_L - u_L) / (S_L - S_star)
        E_s   = chi * (E_L / rho_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L))))
        U_s   = wp.vec3f(chi, chi * S_star, E_s)
        U_L   = wp.vec3f(rho_L, rho_L * u_L, E_L)
        return F_L + S_L * (U_s - U_L)

    # S_star < 0 <= S_R
    chi   = rho_R * (S_R - u_R) / (S_R - S_star)
    E_s   = chi * (E_R / rho_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R))))
    U_s   = wp.vec3f(chi, chi * S_star, E_s)
    U_R   = wp.vec3f(rho_R, rho_R * u_R, E_R)
    return F_R + S_R * (U_s - U_R)


# ── float64 variant ────────────────────────────────────────────────────────────

@wp.func
def hllc_flux_1d_f64(
    rho_L: wp.float64, u_L: wp.float64, p_L: wp.float64, E_L: wp.float64,
    rho_R: wp.float64, u_R: wp.float64, p_R: wp.float64, E_R: wp.float64,
    gamma: wp.float64,
) -> wp.vec3d:
    """HLLC flux (F_rho, F_rhou, F_E) — float64 variant."""
    a_L = wp.sqrt(gamma * p_L / rho_L)
    a_R = wp.sqrt(gamma * p_R / rho_R)

    S_L = wp.min(u_L - a_L, u_R - a_R)
    S_R = wp.max(u_L + a_L, u_R + a_R)

    num    = p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)
    denom  = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    S_star = num / denom

    F_L = wp.vec3d(rho_L * u_L,
                   rho_L * u_L * u_L + p_L,
                   u_L * (E_L + p_L))
    F_R = wp.vec3d(rho_R * u_R,
                   rho_R * u_R * u_R + p_R,
                   u_R * (E_R + p_R))

    zero = wp.float64(0.0)

    if S_L >= zero:
        return F_L

    if S_R <= zero:
        return F_R

    if S_star >= zero:
        chi = rho_L * (S_L - u_L) / (S_L - S_star)
        E_s = chi * (E_L / rho_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L))))
        U_s = wp.vec3d(chi, chi * S_star, E_s)
        U_L = wp.vec3d(rho_L, rho_L * u_L, E_L)
        return F_L + S_L * (U_s - U_L)

    chi = rho_R * (S_R - u_R) / (S_R - S_star)
    E_s = chi * (E_R / rho_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R))))
    U_s = wp.vec3d(chi, chi * S_star, E_s)
    U_R = wp.vec3d(rho_R, rho_R * u_R, E_R)
    return F_R + S_R * (U_s - U_R)
