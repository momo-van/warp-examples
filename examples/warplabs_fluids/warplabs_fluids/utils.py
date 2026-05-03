import numpy as np


def prim_to_cons(rho: np.ndarray, u: np.ndarray, p: np.ndarray, gamma: float) -> np.ndarray:
    """Numpy helper: primitive (rho, u, p) -> conserved (3, N) array."""
    E = p / (gamma - 1.0) + 0.5 * rho * u ** 2
    return np.stack([rho, rho * u, E]).astype(np.float32)


def cons_to_prim(Q: np.ndarray, gamma: float):
    """Numpy helper: conserved (3, N) array -> (rho, u, p) tuple."""
    rho = Q[0]
    u   = Q[1] / rho
    p   = (gamma - 1.0) * (Q[2] - 0.5 * rho * u ** 2)
    return rho, u, p


def l1_error(q_num: np.ndarray, q_ref: np.ndarray, dx: float) -> float:
    return float(np.sum(np.abs(q_num - q_ref)) * dx)


def l2_error(q_num: np.ndarray, q_ref: np.ndarray, dx: float) -> float:
    return float(np.sqrt(np.sum((q_num - q_ref) ** 2) * dx))


def linf_error(q_num: np.ndarray, q_ref: np.ndarray) -> float:
    return float(np.max(np.abs(q_num - q_ref)))
