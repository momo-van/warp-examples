"""
Shu-Osher shock-density interaction problem.

A Mach-3 shock propagates rightward through a sinusoidal density field,
generating complex post-shock density oscillations.

Domain  : [0, 10]
IC      : left of x=1 — post-shock state; right — 1 + 0.2*sin(5x), p=1, u=0
End time: 1.8
Gamma   : 1.4

Reference: Shu & Osher, J. Comput. Phys. 83 (1989), pp. 32-78.
No exact solution — reference is a fine-grid numerical solution.
"""

import numpy as np
from warplabs_fluids import prim_to_cons

# Mach-3 shock post-shock state (Rankine-Hugoniot, ambient rho=1, p=1, u=0)
RHO_L = 3.857143
U_L   = 2.629369
P_L   = 10.33333

X_SHOCK = 1.0   # initial shock position
L       = 10.0  # domain length
GAMMA   = 1.4
T_END   = 1.8


def ic(N: int, gamma: float = GAMMA):
    """
    Returns (Q0, x) where Q0 is the conservative state array (N, 3)
    and x is the cell-centre positions (N,).
    Domain [0, L], dx = L/N.
    """
    dx = L / N
    x  = (np.arange(N) + 0.5) * dx

    rho = np.where(x <= X_SHOCK, RHO_L, 1.0 + 0.2 * np.sin(5.0 * x))
    u   = np.where(x <= X_SHOCK, U_L,   0.0)
    p   = np.where(x <= X_SHOCK, P_L,   1.0)

    rho = rho.astype(np.float32)
    u   = u.astype(np.float32)
    p   = p.astype(np.float32)

    Q0 = prim_to_cons(rho, u, p, gamma)
    return Q0, x
