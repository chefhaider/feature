#!/bin/bash
# Runs the probing pipeline as 3 parallel per-model jobs plus a merge job.
# base/xlsr53/xlsr300m each run as their own sbatch job; the merge job waits on all
# three (--dependency=afterok) and combines their CSVs into the final figures/tables.
#
#   Usage:  bash slurm/pipeline.sh [N_PROBE] [LAYER_STRIDE] [N_REPEATS] [KFOLD] [KFOLD_REPEATS]
#           N_PROBE        default 100
#           LAYER_STRIDE   default 2   probe every Nth layer; 1 = every layer
#           N_REPEATS      default 5   grouped-split repeats per within-lang probe
#           KFOLD          default 5   folds for the H3 gap test
#           KFOLD_REPEATS  default 3   repeats of the H3 gap test
#
#   Faster, coarser:  bash slurm/pipeline.sh 100 4 3 5 2
#
#   Requires artifacts/alignment_cache.pkl -- run `sbatch slurm/align.sh` first.
#
# Run with `bash`, not `sbatch`: this script submits the sbatch jobs itself.
#
# Each invocation writes to its own timestamped directory results/YYYYMMDD_HHMMSSss/,
# so runs are never overwritten. Per-model intermediates go to _parts/<model>/ and the
# merged results to the run directory; run_config.txt records the settings.

set -eo pipefail

N_PROBE="${1:-100}"
LAYER_STRIDE="${2:-2}"
N_REPEATS="${3:-5}"
KFOLD="${4:-5}"
KFOLD_REPEATS="${5:-3}"

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
kfold         : ${KFOLD}
kfold_repeats : ${KFOLD_REPEATS}
models        : base xlsr53 xlsr300m
git_commit    : $(git rev-parse --short HEAD 2>/dev/null || echo "n/a")
EOF

echo "Run directory : ${OUT_DIR}"
echo "Submitting 3 parallel worker jobs (base, xlsr53, xlsr300m)..."
echo "  N_PROBE=${N_PROBE}  LAYER_STRIDE=${LAYER_STRIDE}  N_REPEATS=${N_REPEATS}  KFOLD=${KFOLD}x${KFOLD_REPEATS}"

JOB_IDS=()
for MODEL in base xlsr53 xlsr300m; do
    JID=$(sbatch --parsable slurm/probing_worker.sh \
        "${MODEL}" "${N_PROBE}" "${LAYER_STRIDE}" "${N_REPEATS}" "${PARTS_DIR}" "${KFOLD}" "${KFOLD_REPEATS}")
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
