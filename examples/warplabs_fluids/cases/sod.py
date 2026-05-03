"""
Sod shock tube (Sod 1978).

Left state:  rho=1.0, u=0, p=1.0
Right state: rho=0.125, u=0, p=0.1
Diaphragm at x=0.5,  domain [0, 1].

Exact solution via iterative pressure solver (Toro 2009, Ch. 4).
"""

import numpy as np
from scipy.optimize import brentq

from warplabs_fluids.utils import prim_to_cons


# ---------- Initial condition -----------------------------------------------

def ic(N: int, gamma: float = 1.4):
    """Return (Q0, x) for the Sod problem on N uniform cells in [0, 1]."""
    dx = 1.0 / N
    x  = (np.arange(N) + 0.5) * dx

    rho = np.where(x < 0.5, 1.0,   0.125)
    u   = np.zeros(N)
    p   = np.where(x < 0.5, 1.0,   0.1)

    return prim_to_cons(rho, u, p, gamma), x


# ---------- Exact solution ---------------------------------------------------

def exact(t: float, x: np.ndarray, gamma: float = 1.4):
    """
    Exact Riemann solution for the Sod problem.

    Returns arrays (rho, u, p) sampled at positions x.
    """
    rho_L, u_L, p_L = 1.0,   0.0, 1.0
    rho_R, u_R, p_R = 0.125, 0.0, 0.1
    x0 = 0.5          # diaphragm position

    g  = gamma
    g1 = (g - 1.0) / (g + 1.0)
    g2 = 2.0 / (g + 1.0)
    g3 = 2.0 * g / (g - 1.0)
    g4 = 2.0 / (g - 1.0)
    g5 = (g - 1.0) / 2.0

    a_L = np.sqrt(g * p_L / rho_L)
    a_R = np.sqrt(g * p_R / rho_R)

    # Pressure in star region (Newton-Raphson via brentq)
    def f_L(p):
        if p <= p_L:
            return g4 * a_L * ((p / p_L) ** ((g - 1.0) / (2.0 * g)) - 1.0)
        A = g2 / rho_L
        B = g1 * p_L
        return (p - p_L) * np.sqrt(A / (p + B))

    def f_R(p):
        if p <= p_R:
            return g4 * a_R * ((p / p_R) ** ((g - 1.0) / (2.0 * g)) - 1.0)
        A = g2 / rho_R
        B = g1 * p_R
        return (p - p_R) * np.sqrt(A / (p + B))

    def f(p):
        return f_L(p) + f_R(p) + (u_R - u_L)

    p_star = brentq(f, 1e-8, max(p_L, p_R) * 100.0)
    u_star = 0.5 * (u_L + u_R) + 0.5 * (f_R(p_star) - f_L(p_star))

    # Density in star regions
    if p_star <= p_L:
        rho_sL = rho_L * (p_star / p_L) ** (1.0 / g)
    else:
        rho_sL = rho_L * (p_star / p_L + g1) / (g1 * p_star / p_L + 1.0)

    rho_sR = rho_R * (p_star / p_R + g1) / (g1 * p_star / p_R + 1.0)

    # Wave speeds
    if p_star <= p_L:
        a_sL  = a_L * (p_star / p_L) ** ((g - 1.0) / (2.0 * g))
        S_HL  = u_L  - a_L          # head of left rarefaction
        S_TL  = u_star - a_sL       # tail of left rarefaction
    else:
        S_L   = u_L - a_L * np.sqrt((g + 1.0) / (2.0 * g) * p_star / p_L + (g - 1.0) / (2.0 * g))
        S_HL  = S_TL = S_L          # shock: head == tail

    S_contact = u_star
    S_R_shock = u_R + a_R * np.sqrt((g + 1.0) / (2.0 * g) * p_star / p_R + (g - 1.0) / (2.0 * g))

    # Sample solution
    xi = (x - x0) / t          # similarity variable

    rho_ex = np.empty_like(x)
    u_ex   = np.empty_like(x)
    p_ex   = np.empty_like(x)

    for k, xv in enumerate(xi):
        if xv <= S_HL:                      # undisturbed left
            rho_ex[k], u_ex[k], p_ex[k] = rho_L, u_L, p_L
        elif xv <= S_TL:                    # left rarefaction fan
            coeff   = g2 + g1 / a_L * (u_L - xv)
            rho_ex[k] = rho_L * coeff ** g4
            u_ex[k]   = g2 * (a_L + (g - 1.0) / 2.0 * u_L + xv)
            p_ex[k]   = p_L * coeff ** (2.0 * g / (g - 1.0))
        elif xv <= S_contact:               # left star region
            rho_ex[k], u_ex[k], p_ex[k] = rho_sL, u_star, p_star
        elif xv <= S_R_shock:               # right star region
            rho_ex[k], u_ex[k], p_ex[k] = rho_sR, u_star, p_star
        else:                               # undisturbed right
            rho_ex[k], u_ex[k], p_ex[k] = rho_R, u_R, p_R

    return rho_ex, u_ex, p_ex
