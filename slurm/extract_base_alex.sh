#!/bin/bash -l
#SBATCH --job-name=extract_base_alex
#SBATCH --output=/home/woody/vlbi/vlbi108v/BIMAP-FEATURE/logs/%x_%j.out
#SBATCH --error=/home/woody/vlbi/vlbi108v/BIMAP-FEATURE/logs/%x_%j.err
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mhaiderzaidi21@fau.de

# TEST run of the base-model extraction on the Alex cluster (A40 partition).
# Differs from slurm/extract_base.sh only in the GPU resource request; $HOME and
# $WORK are shared across the FAU clusters, so the env + paths are identical.
#
#   Usage:  sbatch slurm/extract_base_alex.sh [MAX_SAMPLES]   (default: 10, just a smoke test)
#   Tip:    run `bash slurm/prefetch.sh` on a login node first to cache the model + dataset.

unset SLURM_EXPORT_ENV   # let job steps inherit the environment (NHR@FAU recommended with --export=NONE)
set -eo pipefail

# ---- per-model settings -------------------------------------------------
MODEL="facebook/wav2vec2-base"
MODEL_TAG="base"
# -------------------------------------------------------------------------

MAX_SAMPLES="${1:-10}"
LANGUAGES="en_us de_de es_419"

echo "Job started at : $(date)"
echo "Host           : $(hostname)"
echo "SLURM_JOBID    : ${SLURM_JOBID}"
echo "Cluster        : Alex (a40)"
echo "Model          : ${MODEL}"
echo "Max samples    : ${MAX_SAMPLES}"

cd "$WORK/BIMAP-FEATURE"

# ---- environment --------------------------------------------------------
module purge
source "$HOME/miniconda3/bin/activate" feature
# NOTE: no `module load cuda` — the conda env ships torch with bundled CUDA (cu124).

export HF_HOME=$WORK/.cache/huggingface
export HF_DATASETS_CACHE=$WORK/.cache/huggingface/datasets
export HF_HUB_DISABLE_XET=1          # Xet backend stalled in a previous run; use plain HTTPS
# export HF_TOKEN=hf_xxx             # optional: higher HF rate limits
# export HF_HUB_OFFLINE=1            # uncomment once you've run prefetch.sh

nvidia-smi

# Separate dir so this smoke test never overwrites the real TinyGPU outputs.
OUTPUT_DIR="/home/woody/vlbi/vlbi108v/BIMAP-FEATURE/embeddings/alex_test/${MODEL_TAG}"

python src/extract_features.py \
    --model "${MODEL}" \
    --languages ${LANGUAGES} \
    --max-samples "${MAX_SAMPLES}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --split train \
    --trust-remote-code

echo "Job finished at: $(date)"
