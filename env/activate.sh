# Source this AFTER `conda activate diffuser`. Also sourced by slurm/*.sbatch.
#   source env/activate.sh

# mujoco-py builds its extension against this at first import.
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"

# Build mujoco-py's extension WITHOUT GPU rendering support. builder.py:75 otherwise picks
# LinuxGPUExtensionBuilder (because /usr/lib/nvidia exists), which compiles gl/eglshim.c against
# GL/glew.h -- a system header that is not installed and needs root. We never render MuJoCo:
# maze2d eval only steps physics, and plot_plan() draws env.maze_arr with matplotlib. Setting this
# also short-circuits builder.py:116 _ensure_set_env_var, which would otherwise raise at IMPORT
# time demanding /usr/lib/nvidia on LD_LIBRARY_PATH even on a GPU-less login node.
export MUJOCO_PY_FORCE_CPU=1

# Three entries, all load-bearing:
#   mujoco210/bin     - libmujoco210.so for mujoco-py
#   $CONDA_PREFIX/lib - libpython3.8.so.1.0, which IsaacGym's prebuilt binaries need and which
#                       does not exist system-wide on this Rocky 9.4 / glibc 2.34 box
#   /usr/lib64        - libvulkan.so.1 + libnvidia-* for IsaacGym's camera rendering
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:${CONDA_PREFIX:-}/lib:/usr/lib64:${LD_LIBRARY_PATH:-}"

# d4rl imports flow/carla at module load and warns loudly; we only use maze2d.
export D4RL_SUPPRESS_IMPORT_ERROR=1

# Cap torch's intra-op thread pool. torch defaults it to the core count (52 on these nodes) and
# then spends all its time on thread synchronization, because a TemporalUnet activation is tiny
# ([32, 32, 8] for a smoke run). Measured on one CPU training step, batch 32, horizon 8:
#   52 threads (default) 30415 ms   |   8 threads 235 ms   |   4 threads 71 ms   |   1 thread 125 ms
# So this is a ~430x speedup, not a micro-optimisation, and it also stops us from taking over
# every core of a shared login node. torch reads OMP_NUM_THREADS at import, so one export covers
# scripts/train.py, scripts/eval.py, pytest and the slurm jobs without any code change.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# timm loads DINOv2 weights from the HF hub. Download once on the login node (any import with
# pretrained=True); compute nodes then read ~/.cache/huggingface without touching the network.
export HF_HUB_OFFLINE=1
