import warp as wp

# SSP-RK2 update kernel.
# General form: Q_out[i] = alpha*Q0[i] + beta*Q_in[i] + coeff*dt*(divergence)
#
# Stage 1:  alpha=1, beta=0, coeff=1  →  Q1   = Qn + dt*L(Qn)
# Stage 2:  alpha=0.5, beta=0.5, coeff=0.5  →  Qnew = 0.5*Qn + 0.5*(Q1 + dt*L(Q1))
#
# F[j] is the flux at the left face of real cell j  (= right face of cell j-1).
# F[j+1] is the flux at the right face of real cell j.
# Conservative update: dQ/dt = -1/dx * (F[j+1] - F[j])


@wp.kernel
def update_rk_1d(
    Q0:    wp.array2d(dtype=float),   # Qⁿ — read-only
    Q_in:  wp.array2d(dtype=float),   # stage input — read-only
    Q_out: wp.array2d(dtype=float),   # stage output — write
    F:     wp.array2d(dtype=float),   # [3, N+1]
    ng:    int,
    N:     int,
    dt:    float,
    dx:    float,
    alpha:  float,
    beta:   float,
    coeff:  float,
):
    i = wp.tid()       # real-cell index  0 .. N-1
    if i >= N:
        return

    i_ext = i + ng     # position in extended (ghost-padded) array

    # F[i] = left-face flux,  F[i+1] = right-face flux
    for v in range(3):
        rhs = -(F[v, i + 1] - F[v, i]) / dx
        Q_out[v, i_ext] = alpha * Q0[v, i_ext] + beta * Q_in[v, i_ext] + coeff * dt * rhs
