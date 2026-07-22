#!/usr/bin/env python3
"""
Segment-level phonological probing — batch version of notebook/segment_probing copy.ipynb.

Pipeline:
  - load extracted wav2vec2 embeddings (artifacts/embeddings/<tag>/<lang>_features.pkl)
  - attach phoneme labels to their real frames via MMS forced alignment
  - train linear probes across model x feature x LAYER x language
  - train ALL-PAIRS cross-lingual probes (every train->test language pair)
  - write H1/H2/H3 tables (CSV), figures (PNG), and a text summary.

Methodology notes:
  * Frame->phoneme mapping uses FORCED ALIGNMENT (artifacts/alignment_cache.pkl,
    built by src/precompute_alignments.py). The earlier "uniform" mode -- which
    split each utterance's frames evenly across its phonemes -- was removed: it
    ignored leading silence and real phoneme durations, roughly halving probe
    scores (voicing EN 0.50 uniform vs 0.86 forced). The uniform baseline run is
    preserved at results/20260706_21094441/ and in git history.
  * Probes use GROUPED train/test splits (by utterance) so phonemes from one
    recording never straddle the split -- a random split inflates scores.
  * Every layer is probed (not a sample of 5), so H2 gets a real curve.
  * H1/H3 are reported at each model+feature's BEST layer (chosen by
    within-language macro-F1), not an arbitrary middle layer.
  * All metrics carry a `_std` (repeated splits / bootstrap).

CPU-only: no GPU needed.

Usage:
    python src/run_probing.py
    python src/run_probing.py --n-probe 60
    python src/run_probing.py --layer-stride 2      # faster: every other layer
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

from src.align import segment_dataset
from src.phonology import FEATURES
from src.probing import evaluate_probe, cross_lingual_probe

LANGUAGES = ["en_us", "de_de", "es_419"]
ALL_MODELS = {
    "base":     "artifacts/embeddings/base",
    "xlsr53":   "artifacts/embeddings/xlsr53",
    "xlsr300m": "artifacts/embeddings/xlsr300m",
}
ALIGN_CACHE_PATH = "artifacts/alignment_cache.pkl"

# Okabe-Ito colorblind-safe hues
C_WITHIN, C_CROSS, C_LINE = "#0072B2", "#E69F00", "#9AA0A6"
FEATURE_COLORS = {"voicing": "#0072B2", "nasal": "#E69F00",
                  "manner": "#009E73", "place": "#D55E00"}


# ── caches ────────────────────────────────────────────────────────────────────
def load_align_cache():
    if os.path.exists(ALIGN_CACHE_PATH):
        with open(ALIGN_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded alignment cache: {len(cache)} entries from {ALIGN_CACHE_PATH}")
        return cache
    return None


# ── data assembly ─────────────────────────────────────────────────────────────
def load_utts(model_dir, lang, n):
    with open(f"{model_dir}/{lang}_features.pkl", "rb") as f:
        return pickle.load(f)[:n]


def utt_key(lang, s):
    """Recording-unique id (FLEURS reuses `id` across speakers).
    Tuple form — matches the alignment-cache key written by precompute_alignments.py."""
    return (lang, s.get("id"), round(float(s.get("audio_length", 0.0)), 2))


def utt_gid(lang, s):
    """Same identity as utt_key but as a flat string, for use as a sklearn `groups`
    label. (A list of tuples would become a 2-D object array, which breaks
    GroupShuffleSplit — groups must be 1-D scalars.)"""
    return f"{lang}|{s.get('id')}|{round(float(s.get('audio_length', 0.0)), 2)}"


def segment_xy(utts_by_lang, layer, feature, align_cache):
    """Build (X, y, groups) for one layer+feature using MMS forced alignment.

    Each phoneme is mean-pooled over the frames it actually occupies (spans from
    the alignment cache), rather than an even split across the utterance.
    `groups` is the recording each phoneme came from, so train/test splits can be
    grouped by utterance.
    """
    X, y, g = [], [], []
    for lang, utts in utts_by_lang.items():
        for s in utts:
            key = utt_key(lang, s)
            spans = align_cache.get(key)
            if not spans:
                continue
            Xi, yi = segment_dataset(
                s["hidden_states"][layer], s["audio_length"], spans, feature
            )
            if len(yi):
                X.extend(Xi)
                y.extend(yi)
                g.extend([utt_gid(lang, s)] * len(yi))
    return np.array(X), np.array(y), np.array(g)


def safe_evaluate(X, y, groups=None, n_repeats=5):
    if len(y) < 12 or len(set(y)) < 2 or min(Counter(y).values()) < 2:
        return None
    try:
        return evaluate_probe(X, y, groups=groups, n_repeats=n_repeats)
    except Exception as e:
        print(f"      probe failed: {e}", flush=True)
        return None


# ── main experiment loop ──────────────────────────────────────────────────────
def run_probes(models, n_probe, align_cache, n_repeats=5, layer_stride=1):
    """Probe every model x feature x layer x language, plus all-pairs transfer.

    Returns (layer_df, xling_df):
      layer_df — within-language score per model x feature x layer x language (H2,
                 and the within-language reference for H3)
      xling_df — all-pairs transfer per model x feature x layer x train x test
    """
    layer_rows, xling_rows = [], []

    for tag, mdir in models.items():
        print(f"\n== {tag} ==", flush=True)
        utts = {lang: load_utts(mdir, lang, n_probe) for lang in LANGUAGES}
        n_layers = len(utts["en_us"][0]["hidden_states"])
        layers = sorted(set(list(range(0, n_layers, layer_stride)) + [n_layers - 1]))
        print(f"   {n_layers} layers, probing {len(layers)}: {layers}", flush=True)

        for L in layers:
            # memo the (X, y, groups) per (lang, feature) for THIS layer only
            memo = {}

            def seg(lang, feat, _L=L):
                k = (lang, feat)
                if k not in memo:
                    memo[k] = segment_xy({lang: utts[lang]}, _L, feat, align_cache)
                return memo[k]

            for feat in FEATURES:
                # within-language (H2 curve + H3 reference), one probe per language
                for lang in LANGUAGES:
                    X, y, g = seg(lang, feat)
                    res = safe_evaluate(X, y, g, n_repeats)
                    if res:
                        layer_rows.append({"model": tag, "feature": feat, "layer": L,
                                           "language": lang, **res})

                # all-pairs cross-lingual transfer (every ordered train->test pair)
                for tr_lang in LANGUAGES:
                    Xtr, ytr, _ = seg(tr_lang, feat)
                    if len(ytr) < 12 or len(set(ytr)) < 2:
                        continue
                    for te_lang in LANGUAGES:
                        if te_lang == tr_lang:
                            continue
                        Xte, yte, _ = seg(te_lang, feat)
                        if len(yte) < 6 or len(set(yte)) < 2:
                            continue
                        try:
                            r = cross_lingual_probe(Xtr, ytr, Xte, yte)
                        except Exception as e:
                            print(f"      xling failed {tr_lang}->{te_lang}: {e}", flush=True)
                            continue
                        xling_rows.append({"model": tag, "feature": feat, "layer": L,
                                           "train_lang": tr_lang, "test_lang": te_lang, **r})
            memo.clear()
            print(f"   layer {L} done", flush=True)
        del utts

    return pd.DataFrame(layer_rows), pd.DataFrame(xling_rows)


# ── best-layer selection & H1/H3 tables ───────────────────────────────────────
def best_layers(layer_df):
    """Best layer per (model, feature), by mean within-language macro-F1."""
    m = layer_df.groupby(["model", "feature", "layer"])["macro_f1"].mean().reset_index()
    idx = m.groupby(["model", "feature"])["macro_f1"].idxmax()
    return m.loc[idx].set_index(["model", "feature"])["layer"].to_dict()


def h3_table(layer_df, xling_df):
    """Per-model, per-feature transfer gap evaluated at that cell's best layer."""
    bl = best_layers(layer_df)
    rows = []
    for (model, feature), L in bl.items():
        w = layer_df[(layer_df.model == model) & (layer_df.feature == feature)
                     & (layer_df.layer == L)]
        c = xling_df[(xling_df.model == model) & (xling_df.feature == feature)
                     & (xling_df.layer == L)]
        if w.empty or c.empty:
            continue
        rows.append({
            "model": model, "feature": feature, "best_layer": int(L),
            "within_lang": float(w.macro_f1.mean()),
            "within_std": float(w.macro_f1.std(ddof=0)),
            "cross_lang": float(c.macro_f1.mean()),
            "cross_std": float(c.macro_f1.std(ddof=0)),
            "majority": float(c.majority.mean()),
        })
    h3 = pd.DataFrame(rows)
    if not h3.empty:
        h3["transfer_gap"] = h3.within_lang - h3.cross_lang
    return h3


def language_matrix(layer_df, xling_df, model):
    """train x test macro-F1 matrix for one model (diagonal = within-language),
    averaged over features at each feature's best layer."""
    bl = best_layers(layer_df)
    mat = pd.DataFrame(index=LANGUAGES, columns=LANGUAGES, dtype=float)
    for tr in LANGUAGES:
        for te in LANGUAGES:
            vals = []
            for feat in FEATURES:
                L = bl.get((model, feat))
                if L is None:
                    continue
                if tr == te:
                    sub = layer_df[(layer_df.model == model) & (layer_df.feature == feat)
                                   & (layer_df.layer == L) & (layer_df.language == tr)]
                else:
                    sub = xling_df[(xling_df.model == model) & (xling_df.feature == feat)
                                   & (xling_df.layer == L) & (xling_df.train_lang == tr)
                                   & (xling_df.test_lang == te)]
                if not sub.empty:
                    vals.append(sub.macro_f1.mean())
            mat.loc[tr, te] = float(np.mean(vals)) if vals else np.nan
    return mat


# ── figures ───────────────────────────────────────────────────────────────────
def make_figures(layer_df, xling_df, models, out_dir):
    mods = [m for m in models if m in set(layer_df.model)]
    if not mods:
        print("No data to plot.")
        return
    h3 = h3_table(layer_df, xling_df)

    # Fig 1 — H2: macro-F1 vs layer, per model (mean over languages, ±1 std band)
    fig, axes = plt.subplots(1, len(mods), figsize=(5 * len(mods), 4),
                             squeeze=False, sharey=True)
    for ax, tag in zip(axes[0], mods):
        sub = layer_df[layer_df.model == tag]
        for feat in FEATURES:
            s = (sub[sub.feature == feat].groupby("layer")["macro_f1"]
                 .agg(["mean", "std"]).reset_index().sort_values("layer"))
            if s.empty:
                continue
            col = FEATURE_COLORS.get(feat)
            ax.plot(s.layer, s["mean"], marker="o", ms=4, lw=2, color=col, label=feat)
            ax.fill_between(s.layer, s["mean"] - s["std"].fillna(0),
                            s["mean"] + s["std"].fillna(0), color=col, alpha=0.15, lw=0)
        ax.set_title(tag)
        ax.set_xlabel("layer")
        ax.set_ylabel("macro-F1")
        ax.grid(alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("H2: phonological feature decodability by layer "
                 "(mean over languages, ±1 sd)", y=1.03)
    fig.tight_layout()
    p = out_dir / "fig_h2_layers.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"Saved {p}")

    # Fig 2 — H1: cross-lingual transfer at each cell's best layer, by model
    if not h3.empty:
        piv = h3.pivot(index="feature", columns="model", values="cross_lang")
        err = h3.pivot(index="feature", columns="model", values="cross_std")
        piv = piv.reindex(index=[f for f in FEATURES if f in piv.index], columns=mods)
        err = err.reindex_like(piv)
        ax = piv.plot(kind="bar", figsize=(8, 4), yerr=err, capsize=3)
        ax.set_ylabel("macro-F1")
        ax.set_title("H1: cross-lingual transfer (all language pairs) at best layer")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig = ax.get_figure(); fig.tight_layout()
        p = out_dir / "fig_h1_models.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"Saved {p}")

    # Fig 3 — H1 by layer: does any model pull ahead at any depth?
    fig, axes = plt.subplots(1, len(mods), figsize=(5 * len(mods), 4),
                             squeeze=False, sharey=True)
    for ax, tag in zip(axes[0], mods):
        sub = xling_df[xling_df.model == tag]
        for feat in FEATURES:
            s = (sub[sub.feature == feat].groupby("layer")["macro_f1"]
                 .mean().reset_index().sort_values("layer"))
            if not s.empty:
                ax.plot(s.layer, s.macro_f1, marker="o", ms=4, lw=2,
                        color=FEATURE_COLORS.get(feat), label=feat)
        ax.set_title(tag); ax.set_xlabel("layer"); ax.set_ylabel("cross-lingual macro-F1")
        ax.grid(alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("H1: cross-lingual transfer by layer (all pairs)", y=1.03)
    fig.tight_layout()
    p = out_dir / "fig_h1_transfer_by_layer.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"Saved {p}")

    # Fig 4 — H3: transfer-gap dumbbell, ONE PANEL PER MODEL (not aggregated)
    if not h3.empty:
        fig, axes = plt.subplots(1, len(mods), figsize=(5.5 * len(mods), 4.2),
                                 squeeze=False, sharex=True)
        for ax, tag in zip(axes[0], mods):
            sub = h3[h3.model == tag].sort_values("transfer_gap")
            ys = range(len(sub))
            for yi, (_, r) in zip(ys, sub.iterrows()):
                ax.plot([r.cross_lang, r.within_lang], [yi, yi], color=C_LINE, lw=2,
                        zorder=1, solid_capstyle="round")
                ax.text((r.within_lang + r.cross_lang) / 2, yi + 0.16,
                        f"{r.transfer_gap:.2f}", ha="center", va="bottom",
                        fontsize=8, color="#5F6368")
            ax.scatter(sub.cross_lang, ys, s=90, color=C_CROSS, zorder=3,
                       edgecolor="white", linewidth=1.4, label="cross-lingual")
            ax.scatter(sub.within_lang, ys, s=90, color=C_WITHIN, zorder=3,
                       edgecolor="white", linewidth=1.4, label="within-language")
            ax.set_yticks(list(ys))
            ax.set_yticklabels([f.capitalize() for f in sub.feature], fontsize=10)
            ax.set_xlim(0.3, 1.0)
            ax.set_ylim(-0.5, len(sub) - 0.3)
            ax.set_xlabel("macro-F1")
            ax.set_title(tag)
            ax.grid(axis="x", alpha=0.3)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(left=False)
        axes[0][0].legend(loc="lower right", frameon=True, fontsize=8)
        fig.suptitle("H3: transfer gap per feature, per model "
                     "(short bar = universal · long bar = language-specific)", y=1.04)
        fig.tight_layout()
        p = out_dir / "fig_h3_transfer_gap.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"Saved {p}")

    # Fig 5 — all-pairs language transfer matrix, per model
    fig, axes = plt.subplots(1, len(mods), figsize=(4.4 * len(mods), 4),
                             squeeze=False)
    for ax, tag in zip(axes[0], mods):
        mat = language_matrix(layer_df, xling_df, tag)
        im = ax.imshow(mat.values.astype(float), cmap="Blues", vmin=0.3, vmax=1.0)
        ax.set_xticks(range(len(LANGUAGES))); ax.set_xticklabels(LANGUAGES, rotation=45, ha="right")
        ax.set_yticks(range(len(LANGUAGES))); ax.set_yticklabels(LANGUAGES)
        ax.set_xlabel("test language"); ax.set_ylabel("train language")
        ax.set_title(tag)
        for i in range(len(LANGUAGES)):
            for j in range(len(LANGUAGES)):
                v = mat.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if v > 0.7 else "#202124")
        fig.colorbar(im, ax=ax, fraction=0.046, label="macro-F1")
    fig.suptitle("All-pairs transfer (diagonal = within-language), mean over features "
                 "at best layer", y=1.04)
    fig.tight_layout()
    p = out_dir / "fig_language_matrix.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"Saved {p}")


# ── outputs ───────────────────────────────────────────────────────────────────
def write_outputs(layer_df, xling_df, out_dir):
    layer_df.to_csv(out_dir / "layer_df.csv", index=False)
    xling_df.to_csv(out_dir / "xling_df.csv", index=False)
    h3 = h3_table(layer_df, xling_df)
    if not h3.empty:
        h3.to_csv(out_dir / "h3_table.csv", index=False)

    lines = ["RESULTS SUMMARY", "=" * 70, "",
             "Probes use GROUPED splits (by utterance); all metrics are mean over "
             "repeated splits / bootstraps, with _std available in the CSVs.", ""]

    lines += ["H2 — within-language macro-F1 by model x layer (mean over languages "
              "& features):",
              layer_df.pivot_table(index=["model", "layer"], columns="feature",
                                   values="macro_f1").round(3).to_string(), ""]

    bl = best_layers(layer_df)
    lines += ["Best layer per model x feature (by within-language macro-F1):",
              pd.Series(bl).unstack().to_string(), ""]

    if not h3.empty:
        lines += ["H1 — cross-lingual transfer (all pairs) at best layer, by model:",
                  h3.pivot(index="feature", columns="model",
                           values="cross_lang").round(3).to_string(), "",
                  "H1 — bootstrap sd of those values:",
                  h3.pivot(index="feature", columns="model",
                           values="cross_std").round(3).to_string(), "",
                  "H1 — majority baseline (macro-F1):",
                  h3.pivot(index="feature", columns="model",
                           values="majority").round(3).to_string(), "",
                  "H3 — transfer gap per feature, PER MODEL (not aggregated):",
                  h3.sort_values(["model", "transfer_gap"], ascending=[True, False])[
                      ["model", "feature", "best_layer", "within_lang", "cross_lang",
                       "transfer_gap"]].round(3).to_string(index=False), ""]

    for model in sorted(set(layer_df.model)):
        lines += [f"All-pairs transfer matrix — {model} (rows=train, cols=test):",
                  language_matrix(layer_df, xling_df, model).astype(float).round(3).to_string(), ""]

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
    ap.add_argument("--n-repeats", type=int, default=5,
                    help="Repeated grouped splits per within-language probe (error bars).")
    ap.add_argument("--layer-stride", type=int, default=1,
                    help="1 = probe every layer; 2 = every other (faster).")
    args = ap.parse_args()

    models = {k: ALL_MODELS[k] for k in args.models if os.path.isdir(ALL_MODELS[k])}
    assert models, f"No embeddings found for {args.models}. Extract them first."
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Models      : {list(models)}")
    print(f"N_PROBE     : {args.n_probe}")
    print(f"Alignment   : forced (MMS)")
    print(f"Repeats     : {args.n_repeats}   Layer stride: {args.layer_stride}")
    print(f"Output      : {out_dir.resolve()}")

    align_cache = load_align_cache()
    assert align_cache, (
        f"{ALIGN_CACHE_PATH} not found. Run `sbatch slurm/align.sh` "
        "first (needs FLEURS audio + the MMS model)."
    )

    t0 = time.time()
    layer_df, xling_df = run_probes(models, args.n_probe, align_cache,
                                    n_repeats=args.n_repeats,
                                    layer_stride=args.layer_stride)
    print(f"\nlayer_df: {layer_df.shape} | xling_df: {xling_df.shape}")

    make_figures(layer_df, xling_df, models, out_dir)
    write_outputs(layer_df, xling_df, out_dir)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
