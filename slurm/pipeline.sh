#!/bin/bash
# Orchestrates the probing pipeline as 3 PARALLEL per-model jobs + 1 merge job.
# Each of base/xlsr53/xlsr300m runs as its own sbatch job at the same time
# (instead of one serial job looping over all 3 models); the merge job waits
# for all three (--dependency=afterok) and combines their CSVs into the final
# figures/tables.
#
#   Usage:  bash slurm/pipeline.sh [N_PROBE] [LAYER_STRIDE] [N_REPEATS]
#           N_PROBE       default 100
#           LAYER_STRIDE  default 2   (probe every Nth layer; 1 = every layer)
#           N_REPEATS     default 5   (grouped-split repeats per within-lang probe)
#
#   Requires artifacts/alignment_cache.pkl -- run `sbatch slurm/align.sh` first.
#   Frame->phoneme mapping is always MMS forced alignment; the old "uniform"
#   even-split mode was removed (it roughly halved probe scores). Its baseline
#   run is preserved at results/20260706_21094441/ and in git history.
#
# IMPORTANT: run this directly with `bash`, NOT `sbatch` — it calls `sbatch`
# itself for each stage (3 workers + 1 merge).
#
# OUTPUT: every invocation writes to its own timestamped run directory
#   results/YYYYMMDD_HHMMSSss/
# so runs are never overwritten. Per-model intermediates live in
# results/<ts>/_parts/<model>/ and the merged final results in results/<ts>/.
# A run_config.txt records the settings, since the folder name is only a timestamp.
#
# Why split into 3 parallel jobs: a single serial job probing every layer for
# all 3 models timed out at the 24h SLURM walltime cap (job 1754119, killed at
# 92% through the 3rd model). Splitting per model lets each run independently
# and in parallel.

set -eo pipefail

N_PROBE="${1:-100}"
LAYER_STRIDE="${2:-2}"
N_REPEATS="${3:-5}"

cd "$(dirname "$0")/.."
mkdir -p slurm/logs

# One timestamp shared by all 4 jobs of this run: YYYYMMDD_HHMMSS + centiseconds
RUN_TS="$(date +%Y%m%d_%H%M%S%2N)"
OUT_DIR="results/${RUN_TS}"
PARTS_DIR="${OUT_DIR}/_parts"
mkdir -p "${PARTS_DIR}"

if [ ! -f artifacts/alignment_cache.pkl ]; then
    echo "ERROR: artifacts/alignment_cache.pkl missing."
    echo "Run  sbatch slurm/align.sh ${N_PROBE}  first, then re-run this script."
    exit 1
fi

# Provenance: the directory name is just a timestamp, so record the settings.
cat > "${OUT_DIR}/run_config.txt" <<EOF
run_timestamp : ${RUN_TS}
submitted_at  : $(date)
n_probe       : ${N_PROBE}
alignment     : forced (MMS)
layer_stride  : ${LAYER_STRIDE}
n_repeats     : ${N_REPEATS}
models        : base xlsr53 xlsr300m
git_commit    : $(git rev-parse --short HEAD 2>/dev/null || echo "n/a")
EOF

echo "Run directory : ${OUT_DIR}"
echo "Submitting 3 parallel worker jobs (base, xlsr53, xlsr300m)..."
echo "  N_PROBE=${N_PROBE}  LAYER_STRIDE=${LAYER_STRIDE}  N_REPEATS=${N_REPEATS}"

JOB_IDS=()
for MODEL in base xlsr53 xlsr300m; do
    JID=$(sbatch --parsable slurm/probing_worker.sh \
        "${MODEL}" "${N_PROBE}" "${LAYER_STRIDE}" "${N_REPEATS}" "${PARTS_DIR}")
    echo "  ${MODEL} -> job ${JID}"
    JOB_IDS+=("${JID}")
done

DEP="afterok:$(IFS=:; echo "${JOB_IDS[*]}")"
echo "Submitting merge job with dependency ${DEP}..."
MERGE_JID=$(sbatch --parsable --dependency="${DEP}" slurm/probing_merge.sh \
    "${PARTS_DIR}" "${OUT_DIR}")
echo "  merge -> job ${MERGE_JID}"

echo ""
echo "Pipeline submitted. Track with:  squeue -u \$USER"
echo "Final results will land in:      ${OUT_DIR}/  (after the merge job completes)"
