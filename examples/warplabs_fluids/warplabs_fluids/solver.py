import numpy as np
import warp as wp

from .kernels import (
    fused_rk_stage_1d_outflow,
    fused_rk_stage_1d_periodic,
    fused_rk_stage_1d_outflow_w5z,
    fused_rk_stage_1d_periodic_w5z,
    fused_rk_stage_1d_outflow_w5z_f64,
    fused_rk_stage_1d_periodic_w5z_f64,
)

_FUSED_KERNELS = {
    "outflow":  fused_rk_stage_1d_outflow,
    "periodic": fused_rk_stage_1d_periodic,
}

_FUSED_KERNELS_W5Z = {
    "outflow":  fused_rk_stage_1d_outflow_w5z,
    "periodic": fused_rk_stage_1d_periodic_w5z,
}

_FUSED_KERNELS_W5Z_F64 = {
    "outflow":  fused_rk_stage_1d_outflow_w5z_f64,
    "periodic": fused_rk_stage_1d_periodic_w5z_f64,
}

NVARS_1D = 3   # [rho, rho*u, E]

_SCHEME_NG = {
    "weno3-rk2":      2,
    "weno5z-rk3":     3,
    "weno5z-rk3-f64": 3,
}


class WarpEuler1D:
    """
    1-D compressible Euler solver on a uniform Cartesian grid.

    Two schemes available:
      "weno3-rk2"  — WENO3 + HLLC + SSP-RK2, ng=2 (default, legacy)
      "weno5z-rk3" — WENO5-Z + HLLC + SSP-RK3, ng=3 (matches JaxFluids accuracy)

    Ghost cells are embedded in the state array; real cells live at [ng : ng+N].

    Parameters
    ----------
    N      : number of real cells
    dx     : cell width (m)
    gamma  : ratio of specific heats (default 1.4)
    bc     : boundary condition — "outflow" or "periodic"
    device : Warp device string ("cuda" or "cpu")
    scheme : "weno3-rk2" or "weno5z-rk3"
    """

    def __init__(
        self,
        N: int,
        dx: float,
        gamma: float = 1.4,
        bc: str = "outflow",
        device: str = "cuda",
        scheme: str = "weno3-rk2",
    ):
        if bc not in _FUSED_KERNELS:
            raise ValueError(f"bc must be one of {list(_FUSED_KERNELS)}, got {bc!r}")
        if scheme not in _SCHEME_NG:
            raise ValueError(f"scheme must be one of {list(_SCHEME_NG)}, got {scheme!r}")

        self.N      = N
        self.dx     = float(dx)
        self.gamma  = float(gamma)
        self.bc     = bc
        self.device = device
        self.scheme = scheme
        self._ng    = _SCHEME_NG[scheme]
        self._N_ext = N + 2 * self._ng
        self._t     = 0.0
        self._f64   = (scheme == "weno5z-rk3-f64")

        wp.init()

        dtype = wp.float64 if self._f64 else float
        shape = (NVARS_1D, self._N_ext)
        self._Q        = wp.zeros(shape, dtype=dtype, device=device)
        self._Q_stage  = wp.zeros(shape, dtype=dtype, device=device)
        if scheme in ("weno5z-rk3", "weno5z-rk3-f64"):
            self._Q_stage2 = wp.zeros(shape, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self, Q0: np.ndarray) -> None:
        """Set initial condition from a (3, N) numpy array of conserved variables."""
        ng = self._ng
        assert Q0.shape == (NVARS_1D, self.N), \
            f"Expected shape ({NVARS_1D}, {self.N}), got {Q0.shape}"

        if self._f64:
            np_dtype, wp_dtype = np.float64, wp.float64
        else:
            np_dtype, wp_dtype = np.float32, float

        Q_ext = np.zeros((NVARS_1D, self._N_ext), dtype=np_dtype)
        Q_ext[:, ng : ng + self.N] = Q0.astype(np_dtype)
        self._Q = wp.from_numpy(Q_ext, dtype=wp_dtype, device=self.device)
        self._t = 0.0

    def step(self, dt: float) -> None:
        """Advance by one timestep (2 launches for RK2, 3 for RK3)."""
        if self.scheme == "weno5z-rk3-f64":
            self._step_rk3_f64(dt)
        elif self.scheme == "weno5z-rk3":
            self._step_rk3(dt)
        else:
            self._step_rk2(dt)
        self._t += dt

    def compute_dt(self, cfl: float = 0.4) -> float:
        """Return the largest stable dt satisfying the CFL condition."""
        ng = self._ng
        Q = self._Q.numpy()[:, ng : ng + self.N]
        rho = Q[0]
        u   = Q[1] / rho
        p   = (self.gamma - 1.0) * (Q[2] - 0.5 * rho * u ** 2)
        a   = np.sqrt(np.abs(self.gamma * p / rho))
        return cfl * self.dx / float(np.max(np.abs(u) + a))

    def run(self, t_end: float, cfl: float = 0.4, n_max: int = 10_000_000) -> int:
        """Run to t_end using adaptive CFL timestepping. Returns step count."""
        n = 0
        while self._t < t_end and n < n_max:
            dt = min(self.compute_dt(cfl), t_end - self._t)
            self.step(dt)
            n += 1
        return n

    def reset_state(self, Q0: np.ndarray) -> None:
        """Copy Q0 into the existing state buffer in-place (no reallocation).

        Unlike initialize(), this preserves the buffer pointer so any previously
        captured CUDA graph remains valid.
        """
        ng = self._ng
        np_dtype = np.float64 if self._f64 else np.float32
        Q_ext = np.zeros((NVARS_1D, self._N_ext), dtype=np_dtype)
        Q_ext[:, ng : ng + self.N] = Q0.astype(np_dtype)
        self._Q.assign(Q_ext)
        self._t = 0.0

    def capture_graph(self, dt: float, n_steps: int) -> "wp.Graph":
        """Capture n_steps fixed-dt timesteps as a CUDA graph.

        The graph captures only CUDA kernel launches; Python overhead (self._t
        updates) is not captured and should be handled by the caller.

        The captured graph is tied to the current buffer pointers.  Use
        reset_state() (not initialize()) to reset between replays.
        """
        if self.device == "cpu":
            raise ValueError("CUDA graph capture requires a CUDA device")
        wp.capture_begin(device=self.device)
        for _ in range(n_steps):
            if self.scheme == "weno5z-rk3-f64":
                self._step_rk3_f64(dt)
            elif self.scheme == "weno5z-rk3":
                self._step_rk3(dt)
            else:
                self._step_rk2(dt)
        return wp.capture_end(device=self.device)

    @property
    def state(self) -> np.ndarray:
        """Current conserved state as (3, N) numpy float32 array."""
        ng = self._ng
        return self._Q.numpy()[:, ng : ng + self.N]

    @property
    def time(self) -> float:
        return self._t

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _step_rk2(self, dt: float) -> None:
        """SSP-RK2: 2 kernel launches."""
        self._fused_stage(_FUSED_KERNELS[self.bc],
                          self._Q, self._Q, self._Q_stage, dt, 1.0, 0.0, 1.0)
        self._fused_stage(_FUSED_KERNELS[self.bc],
                          self._Q_stage, self._Q, self._Q, dt, 0.5, 0.5, 0.5)

    def _step_rk3(self, dt: float) -> None:
        """SSP-RK3 (Shu-Osher 1988): 3 kernel launches.
        Stage 1: Q1   = Q0 + dt*L(Q0)
        Stage 2: Q2   = 3/4*Q0 + 1/4*Q1 + 1/4*dt*L(Q1)
        Stage 3: Q_n1 = 1/3*Q0 + 2/3*Q2 + 2/3*dt*L(Q2)
        """
        k = _FUSED_KERNELS_W5Z[self.bc]
        self._fused_stage(k, self._Q, self._Q, self._Q_stage, dt, 1.0, 0.0, 1.0)
        self._fused_stage(k, self._Q_stage, self._Q, self._Q_stage2, dt, 0.75, 0.25, 0.25)
        self._fused_stage(k, self._Q_stage2, self._Q, self._Q, dt, 1.0/3.0, 2.0/3.0, 2.0/3.0)

    def _step_rk3_f64(self, dt: float) -> None:
        """SSP-RK3 float64 variant."""
        k = _FUSED_KERNELS_W5Z_F64[self.bc]
        self._fused_stage_f64(k, self._Q, self._Q, self._Q_stage, dt, 1.0, 0.0, 1.0)
        self._fused_stage_f64(k, self._Q_stage, self._Q, self._Q_stage2, dt, 0.75, 0.25, 0.25)
        self._fused_stage_f64(k, self._Q_stage2, self._Q, self._Q, dt, 1.0/3.0, 2.0/3.0, 2.0/3.0)

    def _fused_stage(
        self,
        kernel,
        Q_in:   wp.array,
        Q_ref:  wp.array,
        Q_out:  wp.array,
        dt:     float,
        alpha:  float,
        beta:   float,
        coeff:  float,
    ) -> None:
        wp.launch(
            kernel,
            dim=self.N,
            inputs=[Q_in, Q_ref, Q_out,
                    self._ng, self.N, self.gamma,
                    float(dt), self.dx,
                    float(alpha), float(beta), float(coeff)],
            device=self.device,
        )

    def _fused_stage_f64(
        self,
        kernel,
        Q_in:   wp.array,
        Q_ref:  wp.array,
        Q_out:  wp.array,
        dt:     float,
        alpha:  float,
        beta:   float,
        coeff:  float,
    ) -> None:
        wp.launch(
            kernel,
            dim=self.N,
            inputs=[Q_in, Q_ref, Q_out,
                    self._ng, self.N, wp.float64(self.gamma),
                    wp.float64(dt), wp.float64(self.dx),
                    wp.float64(alpha), wp.float64(beta), wp.float64(coeff)],
            device=self.device,
        )
