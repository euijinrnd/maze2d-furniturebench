#!/bin/bash
# Evaluate:  sbatch --array=0-4 slurm/eval.sbatch --checkpoint logs/one_leg/H32_T20_s0/state_latest.pt
# Array tasks each take an --episode-shard; merge with scripts/aggregate_eval.py.
# one_leg contact physics is GPU-dependent (scripted expert 5/10 on a 4090 vs 2/10 on a 3090), so
# furniture evals stay on suma_rtx4090; maze2d can go anywhere (sbatch -p ...).
#SBATCH --job-name=diffplan-eval
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
SHARD=()
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] && SHARD=(--episode-shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}")
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ${SHARD[*]}"
export PYTHONUNBUFFERED=1
python scripts/eval.py "${SHARD[@]}" "$@"
