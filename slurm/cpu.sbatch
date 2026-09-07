#!/bin/bash
# Run any repo script on a CPU node:  sbatch slurm/cpu.sbatch scripts/build_furniture_dataset.py ...
#SBATCH --job-name=diffplan-cpu
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm-%j.out
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate diffuser
source env/activate.sh
export PYTHONUNBUFFERED=1
python "$@"
