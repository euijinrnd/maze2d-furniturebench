#!/bin/bash
# Build the `diffuser` conda env.
#
# Python 3.8 is a hard floor AND ceiling:
#   - ceiling: IsaacGym Preview 4 ships only gym_36/37/38.so and isaacgym/gymapi.py picks
#     the binary by sys.version_info, so py>=3.9 cannot import it at all.
#   - floor:   furniture_bench's setup.py declares python_requires=">3.7,<3.9".
#
# The install ORDER matters and the pins are not cosmetic. Read the comments before editing.
# Run from the repo root:  bash env/create_env.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_HOME="${CONDA_HOME:-$HOME/anaconda3}"
ENV_NAME="${ENV_NAME:-diffuser}"
FB="$REPO/implement_resources/furniture-bench"
ISAACGYM_TARBALL="${ISAACGYM_TARBALL:-/home/whwjdqls99/IsaacGym_Preview_4_Package.tar.gz}"

step() { echo; echo "=== [$1] $2"; }

step 0 "create env $ENV_NAME (python 3.8)"
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda env list | grep -qE "^${ENV_NAME}\s" || conda create -n "$ENV_NAME" python=3.8 -y
conda activate "$ENV_NAME"
python --version
# mujoco-py resolves MUJOCO_PY_MUJOCO_PATH inside its own setup.py, so these must be exported
# BEFORE step 5, not just at runtime.
source "$REPO/env/activate.sh"

step 1 "old setuptools/pip FIRST -- gym 0.21.0's sdist will not build otherwise"
# setuptools>=66 rejects gym 0.21's legacy version metadata; pip>=24 drops the
# --use-deprecated=legacy-resolver path some of these old sdists rely on.
python -m pip install "setuptools==65.5.0" "wheel==0.38.4"
python -m pip install "pip<24"

step 2 "gym 0.21.0 -- pinned by furniture_bench/setup.py, and d4rl needs the old 4-tuple API"
python -m pip install "gym==0.21.0"

step 3 "numpy<1.24 (MANDATORY) + small pure-python deps"
# isaacgym/torch_utils.py:135 uses `np.float` as a DEFAULT ARGUMENT, so it is evaluated
# at import time and hard-fails on numpy>=1.24. cython<3 is required by mujoco-py's
# build-at-import-time extension.
python -m pip install "numpy==1.23.5" "cython<3" einops tqdm h5py matplotlib pyyaml ninja pytest wandb "timm<1.0"

step 4 "torch 2.4.1+cu121 (last release with cp38 wheels)"
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

step 5 "mujoco-py + d4rl (maze2d)"
# 2.1.2.14, NOT 2.0.2.13: the mujoco-py version has to match the MuJoCo binary on disk, and this
# machine has ~/.mujoco/mujoco210. mujoco-py 2.0.2.x calls discover_mujoco() from its own setup.py
# and hard-fails with "We expected to find the file here: ~/.mujoco/mujoco200".
# 2.1+ is also license-free, so ~/.mujoco/mjkey.txt is vestigial.
# patchelf goes in the env because mujoco_py/builder.py shells out to it (fix_shared_library) and
# the copy in anaconda3/bin is not on PATH once this env is active.
python -m pip install patchelf
python -m pip install mujoco-py==2.1.2.14
python -m pip install "git+https://github.com/Farama-Foundation/d4rl@f2a05c0d66722499bf8031b094d9af3aea7c372b"
# mujoco-py cythonizes lazily on first import, so force that here -- it is the step that actually
# needs gcc, patchelf and LD_LIBRARY_PATH, and we want it to fail loudly now rather than inside a
# slurm job.
python -c "import mujoco_py; print('mujoco_py', mujoco_py.__version__, 'compiled ok')"

step 6 "furniture-bench --no-deps (its setup.py would re-pin gym and drag in extras)"
python -m pip install -e "$FB" --no-deps
# pyrealsense2 + dt_apriltags are NOT optional even for sim-only: furniture_bench/furniture/
# furniture.py:16 -> utils/detection.py -> perception/apriltag.py imports them at module load,
# so importing ANY furniture class needs them present.
python -m pip install opencv-python pynput joblib numba rich pyrealsense2
python -m pip install "$FB"/wheels/dt_apriltags-3.2.0-py3-none-manylinux2010_x86_64.whl

step 7 "IsaacGym Preview 4 (--no-deps: its setup.py pins python_requires<3.9 and re-pins torch)"
if [ ! -d "$HOME/isaacgym" ]; then
  [ -r "$ISAACGYM_TARBALL" ] || { echo "IsaacGym tarball not readable: $ISAACGYM_TARBALL"; exit 1; }
  tar -xzf "$ISAACGYM_TARBALL" -C "$HOME"
fi
python -m pip install -e "$HOME/isaacgym/python" --no-deps

step 8 "this repo"
python -m pip install -e "$REPO"

step 9 "summary"
python - <<'PY'
import importlib.util as u
for m in ["torch", "numpy", "gym", "mujoco_py", "d4rl", "furniture_bench", "isaacgym", "diffplan"]:
    print(f"  {m:16s} {'ok' if u.find_spec(m) else 'MISSING'}")
PY
echo
echo "Done. Activate with:  conda activate $ENV_NAME && source env/activate.sh"
