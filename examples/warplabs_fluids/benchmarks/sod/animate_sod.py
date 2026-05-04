"""
Sod shock tube animation — density / velocity / pressure evolving t=0→0.2.
Saves sod_animation.gif to benchmarks/sod/.

Run from examples/warplabs_fluids/:
  python benchmarks/sod/animate_sod.py
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
from cases.sod import ic as sod_ic, exact as sod_exact

N=512; GAMMA=1.4; DX=1.0/N; T_END=0.2; CFL=0.4; N_FRAMES=80; FPS=25

wp.init()
try:
    wp.zeros(1, dtype=float, device="cuda"); device="cuda"
except Exception:
    device="cpu"
print(f"device: {device}")

Q0, x = sod_ic(N, GAMMA)
solver = WarpEuler1D(N, DX, gamma=GAMMA, bc="outflow", device=device)
solver.initialize(Q0)
rho0,u0,p0 = cons_to_prim(Q0, GAMMA)

t_targets = np.linspace(0.0, T_END, N_FRAMES)
frames = []
for t_target in t_targets:
    if t_target == 0.0:
        frames.append((0.0,rho0.copy(),u0.copy(),p0.copy(),rho0.copy(),u0.copy(),p0.copy()))
        continue
    while solver.time < t_target - 1e-14:
        dt = min(solver.compute_dt(CFL), t_target - solver.time)
        solver.step(dt)
    wp.synchronize()
    rho,u,p = cons_to_prim(solver.state, GAMMA)
    rho_ex,u_ex,p_ex = sod_exact(solver.time, x, GAMMA)
    frames.append((solver.time,rho.copy(),u.copy(),p.copy(),rho_ex,u_ex,p_ex))
    print(f"  t={solver.time:.4f}  {len(frames)}/{N_FRAMES}", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
field_names=["density  ρ","velocity  u","pressure  p"]
ylims=[(0.0,1.18),(-0.05,1.0),(0.0,1.18)]
ic_lines,ex_lines,warp_lines=[],[],[]
for ax,fname,ylim in zip(axes,field_names,ylims):
    li,=ax.plot(x,[np.nan]*N,color="0.72",lw=1.2,ls=":",label="t=0 (IC)")
    le,=ax.plot(x,[np.nan]*N,color="#111",lw=1.8,ls="-",label="exact",zorder=5)
    lw,=ax.plot(x,[np.nan]*N,color="#009e73",lw=1.4,ls="-",label="Warp CUDA",alpha=0.9)
    ax.set_xlim(0,1); ax.set_ylim(*ylim)
    ax.set_xlabel("x",fontsize=10); ax.set_title(fname,fontsize=11)
    ax.legend(fontsize=8); ax.grid(True,lw=0.4,alpha=0.5)
    ic_lines.append(li); ex_lines.append(le); warp_lines.append(lw)

suptitle=fig.suptitle("Sod shock tube  |  N=512  |  t=0.0000",fontsize=12,fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.95])
for li,v in zip(ic_lines,[frames[0][1],frames[0][2],frames[0][3]]): li.set_ydata(v)

def update(i):
    t,rho,u,p,rho_ex,u_ex,p_ex=frames[i]
    for lw,le,wd,ed in zip(warp_lines,ex_lines,[rho,u,p],[rho_ex,u_ex,p_ex]):
        lw.set_ydata(wd); le.set_ydata(ed)
    suptitle.set_text(f"Sod shock tube  |  N=512  |  t={t:.4f}")
    return warp_lines+ex_lines+[suptitle]

ani=animation.FuncAnimation(fig,update,frames=N_FRAMES,interval=int(1000/FPS),blit=True)
out=OUT/"sod_animation.gif"
print(f"\nSaving {N_FRAMES}-frame GIF ...", flush=True)
ani.save(str(out),writer="pillow",fps=FPS,dpi=130)
print(f"Saved -> {out}")
