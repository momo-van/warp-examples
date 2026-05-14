# Warp Projects

A collection of GPU-accelerated projects built on [NVIDIA Warp](https://github.com/NVIDIA/warp) — exploring **computational physics** and **geometry processing** through custom CUDA kernels authored in Python.

Each project lives in its own subfolder under `projects/` with its own code, assets, and tests.

---

## Focus Areas

- **Computational physics** — fluid dynamics, continuum mechanics, thermodynamics, elasticity, wave propagation
- **Geometry processing** — mesh generation, remeshing, signed distance fields, Boolean operations, parametric surfaces
- **Simulation kernels** — custom Warp kernels for solvers that go beyond off-the-shelf physics engines

---

## Prerequisites

### System

- **OS**: Windows 10/11 or Linux
- **GPU**: NVIDIA GPU with CUDA Compute Capability ≥ 7.5 (Turing or newer)
- **CUDA Toolkit**: 12.x
- **Driver**: ≥ 525.60 (Windows) / ≥ 520.61 (Linux)
- **Python**: 3.10 or newer

### Install Warp

```powershell
pip install warp-lang numpy
```

Additional per-project dependencies are listed in each project's own `README.md`.

---

## Projects

| Project | Domain | Description |
|---|---|---|
| [`warplabs_fluids`](projects/warplabs_fluids/) | Compressible CFD | 1-D Euler solver (WENO5-Z + HLLC + SSP-RK3), validated against JaxFluids. Phase 1 complete. |

---

## Running tests

Each project has its own test suite. To run all tests across every project:

```powershell
python -m pytest projects/ -v
```

Or for a specific project:

```powershell
python -m pytest projects/<project_name>/tests/ -v
```

---

## Contributing a new project

1. Create a subfolder: `projects/<your_project>/`
2. Add your Python files and a `tests/` directory
3. Include a `README.md` inside the subfolder describing what the project demonstrates and how to run it
4. Keep all units SI (metres, radians, seconds) unless the problem domain dictates otherwise

---

## License

Apache 2.0 — see individual file headers.
