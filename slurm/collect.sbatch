#!/bin/bash
# Collect scripted demos, one seed per array task:  sbatch --array=1-4 slurm/collect.sbatch --num-demos 25
# Warm the IsaacGym SDF cache (~/.isaacgym/sdf_V100, 5 files) with ONE job before fanning out, or the
# concurrent first-touch jobs race on it and segfault.
#SBATCH --job-name=diffplan-collect
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm-%j.out
#SBATCH --exclude=cs-gpu-01,node23,node14
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/guard.sh && _guard_sbatch_args "$@"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate diffuser
source env/activate.sh
export PYTHONUNBUFFERED=1
python scripts/collect_furniture_demos.py --seed "${SLURM_ARRAY_TASK_ID:-0}" "$@"
