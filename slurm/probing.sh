#!/bin/bash -l
#SBATCH --job-name=probing
#SBATCH --output=/home/woody/vlbi/vlbi108v/BIMAP-FEATURE/logs/%x_%j.out
#SBATCH --error=/home/woody/vlbi/vlbi108v/BIMAP-FEATURE/logs/%x_%j.err
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mhaiderzaidi21@fau.de

# Segment-level phonological probing (notebook Steps 2-4) as a batch job.
#   Usage:  sbatch slurm/probing.sh [N_PROBE]        (default: 100)
#
# NOTE: this step is CPU-only (loads .pkl embeddings + sklearn logistic
#       regression). It does NOT use the GPU — we request one only because the
#       v100 partition is the known-good path on TinyGPU and gives us a large
#       memory share (the xls-r .pkl files are ~5 GB/language). If a CPU-only
#       partition is available to you, switch --partition/--gres to that and
#       add e.g. `--cpus-per-task=8 --mem=64G`.

set -eo pipefail

N_PROBE="${1:-100}"

echo "Job started at : $(date)"
echo "Host           : $(hostname)"
echo "SLURM_JOBID    : ${SLURM_JOBID}"
echo "N_PROBE        : ${N_PROBE}"

cd "$WORK/BIMAP-FEATURE"
mkdir -p logs

# ---- environment --------------------------------------------------------
module purge
source "$HOME/miniconda3/bin/activate" feature
# NOTE: no `module load cuda` — probing is CPU-only anyway.

export HF_HOME=$WORK/.cache/huggingface
export HF_DATASETS_CACHE=$WORK/.cache/huggingface/datasets
export HF_HUB_DISABLE_XET=1

# Build the phoneme cache once if missing (gruut ~3s/utt; ~10-15 min for 300 utts).
if [ ! -f artifacts/phoneme_cache.pkl ]; then
    echo "Phoneme cache missing — building it now..."
    python src/precompute_phonemes.py
fi

python src/run_probing.py \
    --n-probe "${N_PROBE}" \
    --models base xlsr53 xlsr300m \
    --output-dir probing_results

echo "Job finished at: $(date)"
