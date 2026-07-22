#!/bin/bash
# Orchestrates the probing pipeline as 3 PARALLEL per-model jobs + 1 merge job.
# Each of base/xlsr53/xlsr300m runs as its own sbatch job at the same time
# (instead of one serial job looping over all 3 models); the merge job waits
# for all three (--dependency=afterok) and combines their CSVs into the final
# figures/tables.
#
#   Usage:  bash slurm/probing.sh [N_PROBE] [ALIGNMENT] [LAYER_STRIDE] [N_REPEATS]
#           N_PROBE       default 100
#           ALIGNMENT     forced (default) | uniform
#           LAYER_STRIDE  default 2   (probe every Nth layer; 1 = every layer)
#           N_REPEATS     default 5   (grouped-split repeats per within-lang probe)
#
#   forced  -> needs artifacts/alignment_cache.pkl  (run `sbatch slurm/align.sh` first)
#   uniform -> uses artifacts/phoneme_cache.pkl      (each worker builds it if missing)
#
# IMPORTANT: run this directly with `bash`, NOT `sbatch` — it calls `sbatch`
# itself for each stage (3 workers + 1 merge). Results land in
# probing_results_<alignment>/ once the merge job completes.
#
# Why split into 3 parallel jobs: a single serial job probing every layer for
# all 3 models timed out at the 24h SLURM walltime cap (job 1754119, killed at
# 92% through the 3rd model). Splitting per model lets each run independently
# and in parallel, and --layer-stride 2 roughly halves the per-job work.

set -eo pipefail

N_PROBE="${1:-100}"
ALIGNMENT="${2:-forced}"
LAYER_STRIDE="${3:-2}"
N_REPEATS="${4:-5}"

cd "$(dirname "$0")/.."
mkdir -p slurm/logs

PARTS_DIR="probing_results_${ALIGNMENT}/_parts"
OUT_DIR="probing_results_${ALIGNMENT}"

if [ "${ALIGNMENT}" = "forced" ] && [ ! -f artifacts/alignment_cache.pkl ]; then
    echo "ERROR: artifacts/alignment_cache.pkl missing."
    echo "Run  sbatch slurm/align.sh ${N_PROBE}  first, then re-run this script."
    exit 1
fi

echo "Submitting 3 parallel worker jobs (base, xlsr53, xlsr300m)..."
echo "  N_PROBE=${N_PROBE}  ALIGNMENT=${ALIGNMENT}  LAYER_STRIDE=${LAYER_STRIDE}  N_REPEATS=${N_REPEATS}"

JOB_IDS=()
for MODEL in base xlsr53 xlsr300m; do
    JID=$(sbatch --parsable slurm/probing_worker.sh \
        "${MODEL}" "${N_PROBE}" "${ALIGNMENT}" "${LAYER_STRIDE}" "${N_REPEATS}" "${PARTS_DIR}")
    echo "  ${MODEL} -> job ${JID}"
    JOB_IDS+=("${JID}")
done

DEP="afterok:$(IFS=:; echo "${JOB_IDS[*]}")"
echo "Submitting merge job with dependency ${DEP}..."
MERGE_JID=$(sbatch --parsable --dependency="${DEP}" slurm/probing_merge.sh \
    "${ALIGNMENT}" "${PARTS_DIR}" "${OUT_DIR}")
echo "  merge -> job ${MERGE_JID}"

echo ""
echo "Pipeline submitted. Track with:  squeue -u \$USER"
echo "Final results will land in:      ${OUT_DIR}/  (after the merge job completes)"
