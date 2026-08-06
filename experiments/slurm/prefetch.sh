#!/bin/bash -l
# Pre-download all models + FLEURS splits into the shared HF cache.
#
# Run this on a LOGIN node (reliable internet) BEFORE submitting the GPU jobs:
#     bash slurm/prefetch.sh
#
# The last GPU run timed out after 4h stuck downloading the model on the
# compute node (Xet backend stalled). Caching here once avoids that for all
# three extraction jobs.

set -eo pipefail

cd "$WORK/BIMAP-FEATURE"

source "$HOME/miniconda3/bin/activate" feature

export HF_HOME=$WORK/.cache/huggingface
export HF_DATASETS_CACHE=$WORK/.cache/huggingface/datasets
export HF_HUB_DISABLE_XET=1          # the Xet backend hung last time; use plain HTTPS
# export HF_TOKEN=hf_xxx             # optional: higher HF rate limits

python - <<'PY'
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from datasets import load_dataset

MODELS = [
    "facebook/wav2vec2-base",
    "facebook/wav2vec2-large-xlsr-53",
    "facebook/wav2vec2-xls-r-300m",
]
LANGUAGES = ["en_us", "de_de", "es_419"]

for m in MODELS:
    print(f"Caching model: {m}", flush=True)
    Wav2Vec2FeatureExtractor.from_pretrained(m)
    Wav2Vec2Model.from_pretrained(m)

for lang in LANGUAGES:
    print(f"Caching FLEURS split: {lang}", flush=True)
    load_dataset("google/fleurs", lang, split="train", trust_remote_code=True)

print("Prefetch complete.", flush=True)
PY

echo "Done. You can now submit the extraction jobs (optionally with HF_HUB_OFFLINE=1)."
