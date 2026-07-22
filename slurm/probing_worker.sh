#!/bin/bash -l
#SBATCH --job-name=probing_worker
#SBATCH --output=/home/woody/vlbi/vlbi108v/BIMAP-FEATURE/slurm/logs/%x_%j.out
#SBATCH --error=/home/woody/vlbi/vlbi108v/BIMAP-FEATURE/slurm/logs/%x_%j.err
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mhaiderzaidi21@fau.de

# Probe ONE model. Meant to be submitted 3x in parallel (base/xlsr53/xlsr300m)
# by slurm/probing.sh -- not usually run by hand.
#   Usage: sbatch slurm/probing_worker.sh MODEL [N_PROBE] [ALIGNMENT] [LAYER_STRIDE] [N_REPEATS] [PARTS_DIR]

set -eo pipefail

MODEL="$1"
N_PROBE="${2:-100}"
ALIGNMENT="${3:-forced}"
LAYER_STRIDE="${4:-2}"
N_REPEATS="${5:-5}"
PARTS_DIR="${6:-probing_results_${ALIGNMENT}/_parts}"

echo "Job started at : $(date)"
echo "Host           : $(hostname)"
echo "SLURM_JOBID    : ${SLURM_JOBID}"
echo "MODEL          : ${MODEL}"
echo "N_PROBE        : ${N_PROBE}"
echo "ALIGNMENT      : ${ALIGNMENT}"
echo "LAYER_STRIDE   : ${LAYER_STRIDE}"
echo "N_REPEATS      : ${N_REPEATS}"
echo "PARTS_DIR      : ${PARTS_DIR}"

cd "$WORK/BIMAP-FEATURE"
mkdir -p slurm/logs

module purge
source "$HOME/miniconda3/bin/activate" feature
# NOTE: no `module load cuda` — probing is CPU-only anyway.

export HF_HOME=$WORK/.cache/huggingface
export HF_DATASETS_CACHE=$WORK/.cache/huggingface/datasets
export HF_HUB_DISABLE_XET=1

if [ "${ALIGNMENT}" = "forced" ]; then
    if [ ! -f artifacts/alignment_cache.pkl ]; then
        echo "ERROR: artifacts/alignment_cache.pkl missing."
        echo "Run  sbatch slurm/align.sh ${N_PROBE}  first."
        exit 1
    fi
else
    if [ ! -f artifacts/phoneme_cache.pkl ]; then
        echo "Phoneme cache missing — building it now..."
        python src/precompute_phonemes.py
    fi
fi

python src/run_probing.py \
    --n-probe "${N_PROBE}" \
    --models "${MODEL}" \
    --alignment "${ALIGNMENT}" \
    --layer-stride "${LAYER_STRIDE}" \
    --n-repeats "${N_REPEATS}" \
    --output-dir "${PARTS_DIR}/${MODEL}"

echo "Job finished at: $(date)"
