#!/usr/bin/env python3


import argparse, os, sys, pickle, time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.phonology import phonological_features, keep_for, FEATURES, SEGMENT_FILTERS
from src.probing import evaluate_probe, cross_lingual_probe

LANGUAGES = ["en_us", "de_de", "es_419"]
MODEL_ID  = "facebook/wav2vec2-base"   # smallest model; runs on CPU in <15 min
HF_DATASET = "google/fleurs"


def extract_features(n_samples: int, emb_dir: Path, device: str) -> None:
    import torch
    from datasets import load_dataset
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    from tqdm import tqdm

    print(f"\n{'='*60}")
    print(f"Step 1 — Extracting features ({MODEL_ID}, {n_samples} samples/lang, {device})")
    print(f"{'='*60}")

    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID)
    model.to(device).eval()

    emb_dir.mkdir(parents=True, exist_ok=True)

    for lang in LANGUAGES:
        out_path = emb_dir / f"{lang}_features.pkl"
        if out_path.exists():
            print(f"  {lang}: already exists, skipping.")
            continue

        print(f"\n  Loading FLEURS {lang} (streaming)...")
        ds = load_dataset(HF_DATASET, lang, split="train",
                          streaming=True, trust_remote_code=True)

        records = []
        for sample in tqdm(ds.take(n_samples), total=n_samples, desc=f"  {lang}"):
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr    = sample["audio"]["sampling_rate"]

            inputs = fe(audio, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            hidden = [h.cpu().numpy() for h in out.hidden_states]

            records.append({
                "id":              sample.get("id", len(records)),
                "language":        lang,
                "transcription":   sample.get("transcription", ""),
                "raw_transcription": sample.get("raw_transcription", ""),
                "hidden_states":   hidden,
                "audio_length":    len(audio) / sr,
                "sampling_rate":   sr,
            })

        with open(out_path, "wb") as f:
            pickle.dump(records, f)
        print(f"  Saved {len(records)} utterances → {out_path}")


def build_phoneme_table(emb_dir: Path) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("Step 2 — Phonemizing transcriptions (gruut + panphon)")
    print(f"{'='*60}")

    rows = []
    for lang in LANGUAGES:
        with open(emb_dir / f"{lang}_features.pkl", "rb") as f:
            utts = pickle.load(f)
        print(f"  {lang}: {len(utts)} utterances", end="", flush=True)
        t0 = time.time()
        for s in utts:
            for idx, seg in enumerate(phonological_features(s["transcription"], lang)):
                rows.append({"language": lang, "utt_id": s["id"], "idx": idx, **seg})
        print(f"  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    print(f"  Total phonemes: {len(df)}")
    return df



def load_utts(emb_dir: Path, lang: str) -> list:
    with open(emb_dir / f"{lang}_features.pkl", "rb") as f:
        return pickle.load(f)


def segment_xy(utts_by_lang: dict, layer: int, feature: str,
               phon_cache: dict) -> tuple:
    X, y = [], []
    for lang, utts in utts_by_lang.items():
        for s in utts:
            segs = phon_cache[(lang, s["id"])]
            if not segs:
                continue
            h = s["hidden_states"][layer][0]       # (T, D)
            T, n = len(h), len(segs)
            for i, seg in enumerate(segs):
                if not keep_for(feature, seg):
                    continue
                f0 = int(i / n * T)
                f1 = max(f0 + 1, int((i + 1) / n * T))
                X.append(h[f0:f1].mean(axis=0))
                y.append(seg[feature])
    return np.array(X), np.array(y)


def safe_eval(X, y):
    if len(y) < 12 or len(set(y)) < 2 or min(Counter(y).values()) < 2:
        return None
    try:
        return evaluate_probe(X, y)
    except Exception:
        return None


def run_probes(emb_dir: Path, phon_df: pd.DataFrame) -> tuple:
    print(f"\n{'='*60}")
    print("Step 3 — Running linear probes")
    print(f"{'='*60}")

    # Build cache: (lang, utt_id) -> [seg_dicts]
    phon_cache = {}
    for (lang, utt_id), grp in phon_df.groupby(["language", "utt_id"]):
        phon_cache[(lang, utt_id)] = grp.to_dict("records")

    utts = {lang: load_utts(emb_dir, lang) for lang in LANGUAGES}
    n_layers   = len(utts["en_us"][0]["hidden_states"])
    probe_layer = n_layers // 2
    layers = sorted({int(r * (n_layers - 1)) for r in (0, 0.25, 0.5, 0.75, 1.0)})
    print(f"  Layers probed: {layers}  |  probe layer for xling: {probe_layer}")

    layer_rows, within_rows, xling_rows = [], [], []

    for feat in FEATURES:
        print(f"  Feature: {feat}")

        # H2 — layer analysis (all langs pooled)
        for L in layers:
            res = safe_eval(*segment_xy(utts, L, feat, phon_cache))
            if res:
                layer_rows.append({"feature": feat, "layer": L, **res})

        # within-language at probe layer
        rw = safe_eval(*segment_xy(utts, probe_layer, feat, phon_cache))
        if rw:
            within_rows.append({"feature": feat, **rw})

        # H1 / H3 — cross-lingual (train EN → test DE, ES)
        Xen, yen = segment_xy({"en_us": utts["en_us"]}, probe_layer, feat, phon_cache)
        for test_lang in ["de_de", "es_419"]:
            Xt, yt = segment_xy({test_lang: utts[test_lang]}, probe_layer,
                                 feat, phon_cache)
            if len(yen) >= 12 and len(yt) >= 6 and len(set(yen)) >= 2:
                r = cross_lingual_probe(Xen, yen, Xt, yt)
                xling_rows.append({"feature": feat, "test_lang": test_lang, **r})

    layer_df  = pd.DataFrame(layer_rows)
    within_df = pd.DataFrame(within_rows)
    xling_df  = pd.DataFrame(xling_rows)
    print(f"  layer_df: {layer_df.shape}  within_df: {within_df.shape}  "
          f"xling_df: {xling_df.shape}")
    return layer_df, within_df, xling_df



def make_figures(layer_df, within_df, xling_df, out_dir: Path) -> None:
    print(f"\n{'='*60}")
    print("Step 4 — Generating figures")
    print(f"{'='*60}")

    # ── Figure 1: H2 — macro-F1 by layer per feature ──────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for feat in FEATURES:
        s = layer_df[layer_df.feature == feat].sort_values("layer")
        if len(s):
            ax.plot(s.layer, s.macro_f1, marker="o", label=feat)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"H2: Phonological feature decodability by layer\n"
                 f"(wav2vec2-base, within-language, uniform alignment)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = out_dir / "fig1_h2_layers.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  Saved {p1}")

    # ── Figure 2: H1+H3 — within vs cross-lingual per feature ─────────────
    if xling_df.empty or within_df.empty:
        print("  Skipping Fig 2 — not enough cross-lingual data.")
        return

    w = within_df.groupby("feature")["macro_f1"].mean()
    c = xling_df.groupby("feature")["macro_f1"].mean()
    maj = xling_df.groupby("feature")["majority"].mean()
    feats = [f for f in FEATURES if f in w.index and f in c.index]

    x = np.arange(len(feats))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width, [w[f] for f in feats], width, label="Within-lang (EN+DE+ES)", color="steelblue")
    ax.bar(x,         [c[f] for f in feats], width, label="Cross-lingual (EN→DE+ES)", color="darkorange")
    ax.bar(x + width, [maj[f] for f in feats], width, label="Majority baseline", color="lightgray")
    ax.set_xticks(x)
    ax.set_xticklabels(feats)
    ax.set_ylabel("Macro-F1")
    ax.set_title("H1+H3: Within-language vs cross-lingual transfer per feature\n"
                 "(wav2vec2-base, middle layer)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = out_dir / "fig2_h1h3.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  Saved {p2}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary text
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(layer_df, within_df, xling_df, out_dir: Path) -> None:
    lines = ["RESULTS SUMMARY", "=" * 60, ""]

    lines += ["H2 — macro-F1 by layer (all features, within-language):",
              layer_df.pivot_table(index="layer", columns="feature",
                                   values="macro_f1").round(3).to_string(), ""]

    if not within_df.empty:
        lines += ["H3 — within-lang vs cross-lingual macro-F1:"]
        w = within_df.groupby("feature")["macro_f1"].mean().rename("within_lang")
        c = xling_df.groupby("feature")["macro_f1"].mean().rename("cross_lang") \
            if not xling_df.empty else pd.Series(dtype=float)
        h3 = pd.DataFrame({"within_lang": w, "cross_lang": c})
        h3["transfer_gap"] = h3.within_lang - h3.cross_lang
        lines += [h3.sort_values("transfer_gap", ascending=False).round(3).to_string(), ""]

    if not xling_df.empty:
        lines += ["H1 — cross-lingual transfer by feature:",
                  xling_df.groupby("feature")[["macro_f1", "majority"]]
                           .mean().round(3).to_string(), ""]

    txt = "\n".join(lines)
    p = out_dir / "results_summary.txt"
    p.write_text(txt)
    print(f"\n{txt}")
    print(f"\nSaved summary → {p}")

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce phonological probing results (end-to-end, CPU-friendly).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Utterances per language (10 ≈ 15 min on CPU)")
    parser.add_argument("--output-dir", default="reproduce_output",
                        help="Where to save embeddings and figures")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    emb_dir = out_dir / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    extract_features(args.n_samples, emb_dir, device)
    phon_df = build_phoneme_table(emb_dir)
    layer_df, within_df, xling_df = run_probes(emb_dir, phon_df)
    make_figures(layer_df, within_df, xling_df, out_dir)
    write_summary(layer_df, within_df, xling_df, out_dir)

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
    print(f"Figures saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
