#!/usr/bin/env python3
"""
Merge per-model probing outputs (from separate parallel slurm/probing_worker.sh
jobs) into one combined set of figures/tables.

Each part directory (e.g. probing_results_forced/_parts/base/) holds that
model's layer_df.csv + xling_df.csv, produced by src/run_probing.py --models
<one model>. This script concatenates them across models and re-runs the same
figure/table generation used for a single combined job.

Usage:
    python src/merge_probing_results.py \
        --parts-dir probing_results_forced/_parts \
        --output-dir probing_results_forced \
        --models base xlsr53 xlsr300m
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.run_probing import ALL_MODELS, make_figures, write_outputs


def main():
    ap = argparse.ArgumentParser(description="Merge per-model probing_results parts.")
    ap.add_argument("--parts-dir", required=True,
                    help="Directory containing one subdir per model "
                         "(each with layer_df.csv + xling_df.csv).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--models", nargs="+", default=list(ALL_MODELS))
    args = ap.parse_args()

    parts_dir = Path(args.parts_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_dfs, xling_dfs, present = [], [], []
    for m in args.models:
        mdir = parts_dir / m
        lf, xf = mdir / "layer_df.csv", mdir / "xling_df.csv"
        if not lf.exists() or not xf.exists():
            print(f"WARNING: missing results for '{m}' in {mdir} — skipping.")
            continue
        layer_dfs.append(pd.read_csv(lf))
        xling_dfs.append(pd.read_csv(xf))
        present.append(m)

    assert layer_dfs, f"No per-model results found under {parts_dir}."
    layer_df = pd.concat(layer_dfs, ignore_index=True)
    xling_df = pd.concat(xling_dfs, ignore_index=True)
    print(f"Merged models: {present}")
    print(f"layer_df: {layer_df.shape}   xling_df: {xling_df.shape}")

    models = {m: ALL_MODELS[m] for m in present}
    make_figures(layer_df, xling_df, models, out_dir)
    write_outputs(layer_df, xling_df, out_dir)
    print(f"\nMerged results written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
