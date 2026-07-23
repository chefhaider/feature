#!/usr/bin/env python3
"""
Pre-compute MMS forced-alignment phoneme spans for FLEURS utterances.

Forced alignment needs the raw audio, which the embedding .pkl files do not store,
so this reads FLEURS, aligns each utterance, and saves per-recording phoneme time
spans to artifacts/alignment_cache.pkl for run_probing.py to consume.

Cache key = (lang, id, round(audio_length, 2)): recording-unique, since FLEURS
reuses `id` across speakers.

Usage (needs the FLEURS audio and the MMS model; GPU optional but faster):
    python src/precompute_alignments.py --max-samples 100
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src import align

LANGUAGES = ["en_us", "de_de", "es_419"]
OUTPUT = "artifacts/alignment_cache.pkl"


def main():
    ap = argparse.ArgumentParser(description="Pre-compute MMS forced-alignment spans.")
    ap.add_argument("--languages", nargs="+", default=LANGUAGES)
    ap.add_argument("--max-samples", type=int, default=100,
                    help="Utterances per language (match your extraction).")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--output", default=OUTPUT)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    print(f"Device: {device}", flush=True)

    print("Loading MMS aligner (~1.18 GB first time)...", flush=True)
    bundle = align.get_aligner(device)

    # resume from an existing cache if present
    cache = {}
    if os.path.exists(args.output):
        with open(args.output, "rb") as f:
            cache = pickle.load(f)
        print(f"Resuming: {len(cache)} spans already cached.", flush=True)

    for lang in args.languages:
        print(f"\n=== {lang} ===", flush=True)
        # Non-streaming read from the $WORK FLEURS cache — same utterances/order
        # as extract_features.py, and works offline on compute nodes.
        ds = load_dataset("google/fleurs", lang, split="train",
                          trust_remote_code=True)
        if len(ds) > args.max_samples:
            ds = ds.select(range(args.max_samples))
        t0 = time.time()
        for done, s in enumerate(ds, 1):
            audio = np.asarray(s["audio"]["array"], dtype=np.float32)
            sr = s["audio"]["sampling_rate"]
            alen = len(audio) / sr
            key = (lang, s.get("id"), round(alen, 2))
            if key in cache:
                continue
            try:
                word_spans = align.align_words(audio, sr, s["transcription"], bundle, device)
                spans = align.phoneme_spans(word_spans, lang)
                cache[key] = spans
            except Exception as e:
                print(f"  WARN align failed for id {s.get('id')}: {e}", flush=True)
                cache[key] = []
            if done % 10 == 0:
                print(f"  [{done}/{len(ds)}] {time.time()-t0:.0f}s", flush=True)

        # checkpoint after each language
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "wb") as f:
            pickle.dump(cache, f)
        print(f"  saved ({len(cache)} total spans) -> {args.output}", flush=True)

    print(f"\nDone. {len(cache)} alignment entries in {args.output}", flush=True)


if __name__ == "__main__":
    main()
