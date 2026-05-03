import warp as wp


@wp.func
def cons_to_prim_1d(rho: float, rhou: float, E: float, gamma: float) -> wp.vec3f:
    """Conserved [rho, rhou, E] -> primitive [rho, u, p]."""
    u = rhou / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * u * u)
    return wp.vec3f(rho, u, p)


@wp.func
def prim_to_cons_1d(rho: float, u: float, p: float, gamma: float) -> wp.vec3f:
    """Primitive [rho, u, p] -> conserved [rho, rhou, E]."""
    E = p / (gamma - 1.0) + 0.5 * rho * u * u
    return wp.vec3f(rho, rho * u, E)


@wp.func
def sound_speed(rho: float, p: float, gamma: float) -> float:
    return wp.sqrt(gamma * p / rho)
