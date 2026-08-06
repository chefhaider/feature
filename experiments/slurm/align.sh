#!/bin/bash -l
#SBATCH --job-name=align
#SBATCH --output=/home/woody/vlbi/vlbi108v/BIMAP-FEATURE/slurm/logs/%x_%j.out
#SBATCH --error=/home/woody/vlbi/vlbi108v/BIMAP-FEATURE/slurm/logs/%x_%j.err
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mhaiderzaidi21@fau.de

# Build the MMS forced-alignment cache (artifacts/alignment_cache.pkl).
#   Usage:  sbatch slurm/align.sh [MAX_SAMPLES]      (default: 100)
#
# Needs the FLEURS audio (already cached in $WORK) + the MMS model (~1.18 GB,
# cached in torch hub after first run). GPU makes alignment much faster than CPU.
# Resumable: re-running continues from the existing cache.

set -eo pipefail

MAX_SAMPLES="${1:-100}"

echo "Job started at : $(date)"
echo "Host           : $(hostname)"
echo "SLURM_JOBID    : ${SLURM_JOBID}"
echo "Max samples    : ${MAX_SAMPLES}"

cd "$WORK/BIMAP-FEATURE"
mkdir -p slurm/logs artifacts

module purge
source "$HOME/miniconda3/bin/activate" feature
# No `module load cuda` — the conda torch ships bundled CUDA.

export HF_HOME=$WORK/.cache/huggingface
export HF_DATASETS_CACHE=$WORK/.cache/huggingface/datasets
export HF_HUB_DISABLE_XET=1

nvidia-smi

python src/precompute_alignments.py \
    --languages en_us de_de es_419 \
    --max-samples "${MAX_SAMPLES}" \
    --device cuda \
    --output artifacts/alignment_cache.pkl

echo "Job finished at: $(date)"
