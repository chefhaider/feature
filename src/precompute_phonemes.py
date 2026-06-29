"""
Pre-compute gruut+panphon phoneme features for all utterances in the embeddings.
Saves artifacts/phoneme_cache.pkl: dict mapping (lang, utt_id) -> list[phoneme_dicts].

Run once from the project root before running the probing notebook:
    python src/precompute_phonemes.py

Gruut takes ~2-3s per utterance, so 100 utts × 3 langs ≈ 10-15 min.
"""
import os, pickle, time, sys
from pathlib import Path

os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, ".")

from src.phonology import phonological_features

LANGUAGES = ["en_us", "de_de", "es_419"]
# Use base (smallest pkl) to get transcriptions — all models share the same audio/text
PKL_DIR = "embeddings/base"
OUTPUT = "artifacts/phoneme_cache.pkl"

cache = {}
total = 0

for lang in LANGUAGES:
    pkl_path = f"{PKL_DIR}/{lang}_features.pkl"
    print(f"\n=== {lang} ===")
    with open(pkl_path, "rb") as f:
        utts = pickle.load(f)
    print(f"  {len(utts)} utterances")
    for i, s in enumerate(utts):
        utt_id = s.get("id")
        key = (lang, utt_id)
        if key in cache:
            continue
        t0 = time.time()
        segs = phonological_features(s.get("transcription", ""), lang)
        elapsed = time.time() - t0
        cache[key] = segs
        total += 1
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(utts)}] last={elapsed:.1f}s | total cached={total}")
    print(f"  done ({len(utts)} utterances)")

os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
with open(OUTPUT, "wb") as f:
    pickle.dump(cache, f)

print(f"\nSaved {len(cache)} entries to {OUTPUT}")
