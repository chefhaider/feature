#!/usr/bin/env python3
"""
Segment-level phonological probing — batch version of notebook/segment_probing copy.ipynb.

Runs Steps 2-4 of the notebook (Step 1 label distributions are optional here):
  - loads the extracted wav2vec2 embeddings (embeddings/<tag>/<lang>_features.pkl)
  - uniform-aligns frames to phonemes (via the phoneme cache; gruut fallback)
  - trains linear probes per model × feature × layer
  - writes H1/H2/H3 tables (CSV), two figures (PNG), and a text summary.

CPU-only: no GPU needed (loads pickles, runs sklearn logistic regression).

Usage:
    python src/run_probing.py                       # N_PROBE=100, all 3 models
    python src/run_probing.py --n-probe 60
    python src/run_probing.py --models base xlsr53  # subset
    python src/run_probing.py --output-dir probing_results
"""
import argparse
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.phonology import phonological_features, keep_for, FEATURES
from src.probing import evaluate_probe, cross_lingual_probe

LANGUAGES = ["en_us", "de_de", "es_419"]
ALL_MODELS = {
    "base":     "embeddings/base",
    "xlsr53":   "embeddings/xlsr53",
    "xlsr300m": "embeddings/xlsr300m",
}
PHONEME_CACHE_PATH = "artifacts/phoneme_cache.pkl"


# ── phoneme labels ────────────────────────────────────────────────────────────
def load_phoneme_cache():
    if os.path.exists(PHONEME_CACHE_PATH):
        with open(PHONEME_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded phoneme cache: {len(cache)} entries from {PHONEME_CACHE_PATH}")
        return cache
    print(f"WARNING: {PHONEME_CACHE_PATH} not found — falling back to live gruut (~3s/utt).")
    print("Run `python src/precompute_phonemes.py` first to avoid this.")
    return {}


# ── data assembly ─────────────────────────────────────────────────────────────
def load_utts(model_dir, lang, n):
    with open(f"{model_dir}/{lang}_features.pkl", "rb") as f:
        return pickle.load(f)[:n]


def utt_phonemes(sample, lang, phon_cache):
    key = (lang, sample.get("id"))
    if key not in phon_cache:
        phon_cache[key] = phonological_features(sample.get("transcription", ""), lang)
    return phon_cache[key]


def segment_xy(utts_by_lang, layer, feature, phon_cache):
    """Uniform alignment: split each utterance's frames evenly across its phonemes,
    mean-pool each phoneme's slice at `layer`, keep only feature-relevant segments."""
    X, y = [], []
    for lang, utts in utts_by_lang.items():
        for s in utts:
            segs = utt_phonemes(s, lang, phon_cache)
            if not segs:
                continue
            h = s["hidden_states"][layer][0]          # (T, D)
            T, n = len(h), len(segs)
            for i, seg in enumerate(segs):
                if not keep_for(feature, seg):
                    continue
                f0 = int(i / n * T)
                f1 = max(f0 + 1, int((i + 1) / n * T))
                X.append(h[f0:f1].mean(axis=0))
                y.append(seg[feature])
    return np.array(X), np.array(y)


def safe_evaluate(X, y):
    if len(y) < 12 or len(set(y)) < 2 or min(Counter(y).values()) < 2:
        return None
    try:
        return evaluate_probe(X, y)
    except Exception:
        return None


# ── main experiment loop (notebook Step 3) ────────────────────────────────────
def run_probes(models, n_probe, phon_cache):
    layer_rows, within_rows, xling_rows = [], [], []

    for tag, mdir in models.items():
        print(f"\n== {tag} ==", flush=True)
        utts = {lang: load_utts(mdir, lang, n_probe) for lang in LANGUAGES}
        n_layers = len(utts["en_us"][0]["hidden_states"])
        layers = sorted({int(r * (n_layers - 1)) for r in (0, 0.25, 0.5, 0.75, 1.0)})
        probe_layer = n_layers // 2
        print(f"   layers={layers}  probe_layer={probe_layer}", flush=True)

        for feat in FEATURES:
            print(f"   feature: {feat}", flush=True)
            # H2: layer analysis, within-language (pooled across languages)
            for L in layers:
                res = safe_evaluate(*segment_xy(utts, L, feat, phon_cache))
                if res:
                    layer_rows.append({"model": tag, "feature": feat, "layer": L, **res})

            # within-language reference at the probe layer (for H3)
            rw = safe_evaluate(*segment_xy(utts, probe_layer, feat, phon_cache))
            if rw:
                within_rows.append({"model": tag, "feature": feat, "layer": probe_layer, **rw})

            # H1/H3: cross-lingual transfer (train EN -> test DE/ES) at the probe layer
            Xen, yen = segment_xy({"en_us": utts["en_us"]}, probe_layer, feat, phon_cache)
            for test in ["de_de", "es_419"]:
                Xt, yt = segment_xy({test: utts[test]}, probe_layer, feat, phon_cache)
                if len(yen) >= 12 and len(yt) >= 6 and len(set(yen)) >= 2:
                    r = cross_lingual_probe(Xen, yen, Xt, yt)
                    xling_rows.append({"model": tag, "feature": feat, "test": test,
                                       "layer": probe_layer, **r})
        del utts

    return (pd.DataFrame(layer_rows),
            pd.DataFrame(within_rows),
            pd.DataFrame(xling_rows))


# ── figures & summary (notebook Step 4) ───────────────────────────────────────
def make_figures(layer_df, within_df, xling_df, models, out_dir):
    mods = list(models)

    # Fig 1 — H2: macro-F1 by layer, per model
    fig, axes = plt.subplots(1, len(mods), figsize=(5 * len(mods), 4),
                             squeeze=False, sharey=True)
    for ax, tag in zip(axes[0], mods):
        sub = layer_df[layer_df.model == tag]
        for feat in FEATURES:
            s = sub[sub.feature == feat].sort_values("layer")
            if len(s):
                ax.plot(s.layer, s.macro_f1, marker="o", label=feat)
        ax.set_title(tag)
        ax.set_xlabel("layer")
        ax.set_ylabel("macro-F1")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("H2: phonological feature decodability by layer", y=1.03)
    fig.tight_layout()
    p1 = out_dir / "fig_h2_layers.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p1}")

    # Fig 2 — H1: cross-lingual transfer macro-F1 per feature, grouped by model
    if not xling_df.empty:
        h1 = xling_df.groupby(["feature", "model"])["macro_f1"].mean().unstack()
        h1 = h1.reindex(index=[f for f in FEATURES if f in h1.index], columns=mods)
        ax = h1.plot(kind="bar", figsize=(8, 4))
        ax.set_ylabel("macro-F1")
        ax.set_title("H1: cross-lingual transfer (train EN → test DE+ES) by model")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)
        fig = ax.get_figure()
        fig.tight_layout()
        p2 = out_dir / "fig_h1_models.png"
        fig.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {p2}")


def write_outputs(layer_df, within_df, xling_df, out_dir):
    layer_df.to_csv(out_dir / "layer_df.csv", index=False)
    within_df.to_csv(out_dir / "within_df.csv", index=False)
    xling_df.to_csv(out_dir / "xling_df.csv", index=False)

    lines = ["RESULTS SUMMARY", "=" * 60, ""]

    lines += ["H2 — macro-F1 by model × layer × feature:",
              layer_df.pivot_table(index=["model", "layer"], columns="feature",
                                   values="macro_f1").round(3).to_string(), ""]

    if not xling_df.empty:
        h1 = xling_df.groupby(["feature", "model"]).agg(
            macro_f1=("macro_f1", "mean"), majority=("majority", "mean")).reset_index()
        lines += ["H1 — cross-lingual transfer macro-F1 (train EN → test DE & ES):",
                  h1.pivot(index="feature", columns="model", values="macro_f1")
                    .round(3).to_string(),
                  "",
                  "H1 — majority baseline:",
                  h1.pivot(index="feature", columns="model", values="majority")
                    .round(3).to_string(), ""]

    if not within_df.empty and not xling_df.empty:
        w = within_df.groupby("feature")["macro_f1"].mean()
        c = xling_df.groupby("feature")["macro_f1"].mean()
        h3 = pd.DataFrame({"within_lang": w, "cross_lang": c})
        h3["transfer_gap"] = h3.within_lang - h3.cross_lang
        lines += ["H3 — within-lang vs cross-lingual transfer gap per feature:",
                  h3.sort_values("transfer_gap", ascending=False).round(3).to_string(), ""]

    txt = "\n".join(lines)
    (out_dir / "results_summary.txt").write_text(txt)
    print("\n" + txt)
    print(f"\nSaved CSVs + summary to {out_dir}/")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Segment-level phonological probing (batch).")
    ap.add_argument("--n-probe", type=int, default=100,
                    help="Utterances per language to probe.")
    ap.add_argument("--models", nargs="+", default=list(ALL_MODELS),
                    choices=list(ALL_MODELS), help="Which model tags to run.")
    ap.add_argument("--output-dir", default="probing_results")
    args = ap.parse_args()

    models = {k: ALL_MODELS[k] for k in args.models if os.path.isdir(ALL_MODELS[k])}
    assert models, f"No embeddings found for {args.models}. Extract them first."
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Models   : {list(models)}")
    print(f"N_PROBE  : {args.n_probe}")
    print(f"Output   : {out_dir.resolve()}")

    t0 = time.time()
    phon_cache = load_phoneme_cache()
    layer_df, within_df, xling_df = run_probes(models, args.n_probe, phon_cache)
    print(f"\nlayer: {layer_df.shape} | within: {within_df.shape} | xling: {xling_df.shape}")

    make_figures(layer_df, within_df, xling_df, models, out_dir)
    write_outputs(layer_df, within_df, xling_df, out_dir)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
