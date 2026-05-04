"""
Shu-Osher shock-density interaction animation.

Animates density, velocity, and pressure from t=0 to t=1.8.
The Mach-3 shock sweeps rightward through a sinusoidal density field,
generating complex post-shock density oscillations.

Saves shu_osher_animation.gif to benchmarks/shu_osher/.

Run from examples/warplabs_fluids/:
  python benchmarks/shu_osher/animate_shu_osher.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warp as wp

ROOT = Path(__file__).parent.parent.parent
OUT  = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from warplabs_fluids import WarpEuler1D, cons_to_prim
from cases.shu_osher import ic as shu_ic, L, T_END, GAMMA

N        = 512
DX       = L / N
CFL      = 0.4
N_FRAMES = 90
FPS      = 25

wp.init()
try:
    wp.zeros(1, dtype=float, device="cuda"); device = "cuda"
except Exception:
    device = "cpu"
print(f"device: {device}")

Q0, x = shu_ic(N, GAMMA)
solver = WarpEuler1D(N, DX, gamma=GAMMA, bc="outflow", device=device)
solver.initialize(Q0)
rho0, u0, p0 = cons_to_prim(Q0, GAMMA)

t_targets = np.linspace(0.0, T_END, N_FRAMES)
frames = []

for t_target in t_targets:
    if t_target == 0.0:
        frames.append((0.0, rho0.copy(), u0.copy(), p0.copy()))
        continue
    while solver.time < t_target - 1e-14:
        dt = min(solver.compute_dt(CFL), t_target - solver.time)
        solver.step(dt)
    wp.synchronize()
    rho, u, p = cons_to_prim(solver.state, GAMMA)
    frames.append((solver.time, rho.copy(), u.copy(), p.copy()))
    print(f"  t={solver.time:.4f}  {len(frames)}/{N_FRAMES}", flush=True)

# ── build figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
field_names = ["density  ρ", "velocity  u", "pressure  p"]

# set y-limits from data range with a small margin
def ylim_from_frames(idx):
    vals = np.concatenate([f[idx + 1] for f in frames])
    lo, hi = vals.min(), vals.max()
    pad = 0.08 * (hi - lo) if hi > lo else 0.1
    return lo - pad, hi + pad

ylims = [ylim_from_frames(i) for i in range(3)]

ic_lines, warp_lines = [], []
for ax, fname, ylim in zip(axes, field_names, ylims):
    li, = ax.plot(x, [np.nan]*N, color="0.72", lw=1.0, ls=":", label="t=0 (IC)", zorder=1)
    lw, = ax.plot(x, [np.nan]*N, color="#009e73", lw=1.4, ls="-", label="Warp CUDA", alpha=0.9)
    ax.set_xlim(0, L); ax.set_ylim(*ylim)
    ax.set_xlabel("x", fontsize=10); ax.set_title(fname, fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, lw=0.4, alpha=0.5)
    ic_lines.append(li); warp_lines.append(lw)

suptitle = fig.suptitle(
    f"Shu-Osher  |  N={N}  |  t=0.0000",
    fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])

for li, v in zip(ic_lines, [frames[0][1], frames[0][2], frames[0][3]]):
    li.set_ydata(v)


def update(i):
    t, rho, u, p = frames[i]
    for lw, val in zip(warp_lines, [rho, u, p]):
        lw.set_ydata(val)
    suptitle.set_text(f"Shu-Osher  |  N={N}  |  t={t:.4f}")
    return warp_lines + [suptitle]


ani = animation.FuncAnimation(
    fig, update, frames=N_FRAMES, interval=int(1000 / FPS), blit=True)
out = OUT / "shu_osher_animation.gif"
print(f"\nSaving {N_FRAMES}-frame GIF ...", flush=True)
ani.save(str(out), writer="pillow", fps=FPS, dpi=130)
print(f"Saved -> {out}")
