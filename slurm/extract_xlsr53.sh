#!/bin/bash -l
#SBATCH --job-name=extract_xlsr53
#SBATCH --output=/home/woody/vlbi/vlbi108v/BIMAP-FEATURE/logs/%x_%j.out
#SBATCH --error=/home/woody/vlbi/vlbi108v/BIMAP-FEATURE/logs/%x_%j.err
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mhaiderzaidi21@fau.de

# Extract Wav2Vec2 embeddings from FLEURS for ONE model.
#   Usage:  sbatch slurm/extract_xlsr53.sh [MAX_SAMPLES]   (default: 100)
#   Tip:    run `bash slurm/prefetch.sh` on a login node first to cache the
#           model + dataset, then this job just reads from $WORK cache.

set -eo pipefail

# ---- per-model settings -------------------------------------------------
MODEL="facebook/wav2vec2-large-xlsr-53"
MODEL_TAG="xlsr53"
# -------------------------------------------------------------------------

MAX_SAMPLES="${1:-100}"
LANGUAGES="en_us de_de es_419"

echo "Job started at : $(date)"
echo "Host           : $(hostname)"
echo "SLURM_JOBID    : ${SLURM_JOBID}"
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

OUTPUT_DIR="/home/woody/vlbi/vlbi108v/BIMAP-FEATURE/embeddings/${MODEL_TAG}"

python src/extract_features.py \
    --model "${MODEL}" \
    --languages ${LANGUAGES} \
    --max-samples "${MAX_SAMPLES}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --split train \
    --trust-remote-code

echo "Job finished at: $(date)"
