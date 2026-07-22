#!/bin/bash -l
#SBATCH --job-name=probing_merge
#SBATCH --output=/home/woody/vlbi/vlbi108v/BIMAP-FEATURE/slurm/logs/%x_%j.out
#SBATCH --error=/home/woody/vlbi/vlbi108v/BIMAP-FEATURE/slurm/logs/%x_%j.err
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mhaiderzaidi21@fau.de

# Combine the 3 per-model probing_worker.sh outputs into the final figures/CSVs.
# Submitted with --dependency=afterok:<3 worker job ids> by slurm/probing.sh.
#   Usage: sbatch slurm/probing_merge.sh ALIGNMENT PARTS_DIR OUT_DIR

set -eo pipefail

ALIGNMENT="${1:-forced}"
PARTS_DIR="${2:-probing_results_${ALIGNMENT}/_parts}"
OUT_DIR="${3:-probing_results_${ALIGNMENT}}"

echo "Job started at : $(date)"
echo "SLURM_JOBID    : ${SLURM_JOBID}"
echo "PARTS_DIR      : ${PARTS_DIR}"
echo "OUT_DIR        : ${OUT_DIR}"

cd "$WORK/BIMAP-FEATURE"

module purge
source "$HOME/miniconda3/bin/activate" feature

python src/merge_probing_results.py \
    --parts-dir "${PARTS_DIR}" \
    --output-dir "${OUT_DIR}" \
    --models base xlsr53 xlsr300m

echo "Job finished at: $(date)"
