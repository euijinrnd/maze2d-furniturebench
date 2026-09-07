#!/bin/bash
# Abort if an sbatch-only option was passed AFTER the script name.
#
# sbatch stops parsing its own options at the script path, so everything past it becomes the
# script's argv. This typo therefore sets NO dependency:
#
#   sbatch --array=0-4 slurm/eval.sbatch --checkpoint X --dependency=afterok:123
#
# it hands `--dependency=...` to python, which dies in argparse -- but the job still "runs" and
# exits, so anything waiting on it with afterok proceeds on nothing. That silently wasted 55 eval
# jobs once. argparse would catch it too, but only ~40 s in, after conda and isaacgym import.
#
# Only flags that sbatch owns and no script here accepts are listed; --output is deliberately
# ABSENT, since scripts/eval.py has its own.
_guard_sbatch_args() {
  local a
  for a in "$@"; do
    case "$a" in
      --dependency|--dependency=*|--array|--array=*|--partition|--partition=*|-p|\
      --gres|--gres=*|--qos|--qos=*|--job-name|--job-name=*|-J|\
      --nodelist|--nodelist=*|-w|--exclude|--exclude=*|-x|\
      --nodes|--nodes=*|-N|--ntasks|--ntasks=*|-n|\
      --cpus-per-task|--cpus-per-task=*|-c|--mem|--mem=*|\
      --time|--time=*|-t|--begin|--begin=*|--account|--account=*|-A|\
      --constraint|--constraint=*|-C)
        cat >&2 <<EOF

=========================== SUBMISSION ERROR ===========================
  '$a' is an sbatch option, but it appeared AFTER the script name, so
  slurm never saw it -- it was forwarded to the python script instead.

  Put every sbatch option BEFORE the script path:

      sbatch $a ... $0 --checkpoint ...
             ^^^^^^^^^^ here            not here ^^^^^^^^^^^^^^^^^^^^^^

  Nothing ran. Resubmit with the options reordered.
========================================================================

EOF
        exit 64  # EX_USAGE
        ;;
    esac
  done
}
