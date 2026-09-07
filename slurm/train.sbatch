#!/bin/bash
# Train:  sbatch slurm/train.sbatch --config one_leg [--logbase /scratch2/...]
# Fastest-first fallback if suma_rtx4090 is full: suma_a6000, gigabyte_a6000 > *_rtx3090 > *_a5000.
# Never asus_pro6000 / gigabyte_pro6000 (Blackwell sm_120; torch 2.4.1+cu121 has no kernels) and
# never cs-gpu-01, node23 or node14 (CUDA init fails there although slurm reports them healthy).
#SBATCH --job-name=diffplan-train
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
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
export PYTHONUNBUFFERED=1
python scripts/train.py "$@"
