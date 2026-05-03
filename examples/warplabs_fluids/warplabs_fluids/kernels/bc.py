import warp as wp


@wp.kernel
def bc_outflow_1d(Q: wp.array2d(dtype=float), ng: int, N: int, nvars: int):
    """Zeroth-order outflow: copy edge real cell into ghost cells."""
    var = wp.tid()
    if var >= nvars:
        return
    for g in range(ng):
        Q[var, ng - 1 - g] = Q[var, ng]
        Q[var, ng + N + g] = Q[var, ng + N - 1]


@wp.kernel
def bc_periodic_1d(Q: wp.array2d(dtype=float), ng: int, N: int, nvars: int):
    """Periodic: wrap real cells into ghost cells on both sides."""
    var = wp.tid()
    if var >= nvars:
        return
    for g in range(ng):
        Q[var, ng - 1 - g] = Q[var, ng + N - 1 - g]
        Q[var, ng + N + g] = Q[var, ng + g]
