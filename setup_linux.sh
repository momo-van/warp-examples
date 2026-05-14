#!/usr/bin/env bash
# setup_linux.sh
# Configure Ubuntu / WSL2 for warplabs-fluids GPU benchmarks.
#
# Run from the repo root:
#   bash setup_linux.sh
#
# WSL2 note: the NVIDIA GPU driver is provided by the Windows host — do NOT
# install the NVIDIA GPU driver inside WSL2. The Windows driver exposes CUDA
# through /usr/lib/wsl/lib/libcuda.so.1 automatically.
#
# If Warp CUDA fails with "NVRTC not found" after this script, see the
# "Fallback: CUDA Toolkit" section at the bottom of this file.

set -euo pipefail

VENV_DIR="$HOME/venv-warplabs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== warplabs-fluids Linux / WSL2 setup ==="
echo "    Repo: $SCRIPT_DIR"
echo ""

# ── 0. Detect environment ────────────────────────────────────────────────────
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "[info] WSL2 detected — GPU via Windows driver passthrough"
else
    echo "[info] Native Linux"
fi

# ── 1. System packages ────────────────────────────────────────────────────────
echo ""
echo "[1/5] System packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq git curl wget software-properties-common

# Python 3.11 via deadsnakes PPA (Ubuntu 22.04 ships 3.10; 24.04 ships 3.12)
UBUNTU_VER=$(. /etc/os-release && echo "$VERSION_CODENAME")
if [[ "$UBUNTU_VER" == "jammy" ]]; then
    if ! command -v python3.11 &>/dev/null; then
        echo "    Adding deadsnakes PPA..."
        sudo add-apt-repository -y ppa:deadsnakes/ppa >/dev/null
        sudo apt-get update -qq
    fi
    sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
    PYTHON=python3.11
else
    # Ubuntu 24.04+ ships Python 3.12 by default
    sudo apt-get install -y -qq python3 python3-venv python3-dev
    PYTHON=python3
fi

echo "    $($PYTHON --version)"

# ── 2. GPU check ──────────────────────────────────────────────────────────────
echo ""
echo "[2/5] GPU check..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
        | sed 's/^/    /'
else
    echo "    nvidia-smi not found — GPU passthrough may not be active."
    echo "    In WSL2: ensure Windows NVIDIA driver >= 530 is installed."
fi

# ── 3. Virtualenv ─────────────────────────────────────────────────────────────
echo ""
echo "[3/5] Virtualenv at $VENV_DIR..."
$PYTHON -m venv "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel -q

# ── 4. Python packages ────────────────────────────────────────────────────────
echo ""
echo "[4/5] Python packages (JAX CUDA download ~1 GB, please wait)..."
pip install -r "$SCRIPT_DIR/requirements-linux.txt"

# ── 5. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Verifying..."

python - <<'PYEOF'
import sys
print(f"  Python:   {sys.version.split()[0]}")

import numpy as np
print(f"  numpy:    {np.__version__}")

import warp as wp
wp.init()
wp_devs = [str(d) for d in wp.get_devices()]
print(f"  warp:     {wp.__version__}  devices={wp_devs}")

import jax
jax_devs = [str(d) for d in jax.devices()]
print(f"  jax:      {jax.__version__}  devices={jax_devs}")

try:
    gpus = jax.devices('gpu')
    print(f"  JAX GPU:  {gpus[0]}  [OK]")
except Exception as e:
    print(f"  JAX GPU:  not available — {e}")

# Quick Warp CUDA smoke test
import warp as wp
if any('cuda' in str(d) for d in wp.get_devices()):
    import numpy as np_
    @wp.kernel
    def _add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
        i = wp.tid()
        c[i] = a[i] + b[i]
    a = wp.array([1.0, 2.0, 3.0], dtype=float, device="cuda")
    b = wp.array([4.0, 5.0, 6.0], dtype=float, device="cuda")
    c = wp.zeros(3, dtype=float, device="cuda")
    wp.launch(_add, dim=3, inputs=[a, b, c], device="cuda")
    wp.synchronize()
    print(f"  Warp GPU: smoke test passed  {c.numpy()}")
else:
    print("  Warp GPU: no CUDA device found")
PYEOF

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Activate in future sessions:"
echo "  source ~/venv-warplabs/bin/activate"
echo ""
echo "Run benchmarks:"
echo "  cd $SCRIPT_DIR/projects/warplabs_fluids"
echo "  python -m pytest tests/ -v"
echo "  python benchmarks/compare_sod.py"
echo "  python benchmarks/scaling_benchmark.py"
echo ""

# ── Fallback instructions (not run automatically) ────────────────────────────
cat <<'FALLBACK'
─────────────────────────────────────────────────────────────────────
If Warp CUDA fails with "NVRTC not found", install the CUDA toolkit:

    UBUNTU_VER=$(. /etc/os-release && echo "${VERSION_ID//./}")
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VER}/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install -y cuda-nvrtc-12-6 cuda-nvrtc-dev-12-6
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

Then re-run the verify step (step 5 above).
─────────────────────────────────────────────────────────────────────
FALLBACK
