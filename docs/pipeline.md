# Pipeline — execution order (fresh run)

Which shell script runs when, which Python file it calls, and what it produces.

```mermaid
flowchart TD
    %% ---------------- Stage 1 ----------------
    subgraph S1["STAGE 1 · Prefetch — login node"]
        A1["slurm/prefetch.sh<br/><i>inline python heredoc</i>"]
        A2[("$WORK/.cache/huggingface<br/>models + FLEURS audio")]
        A1 --> A2
    end

    %% ---------------- Stage 2 ----------------
    subgraph S2["STAGE 2 · Extract embeddings — 3 GPU jobs, parallel"]
        B1["slurm/extract_base.sh<br/>src/extract_features.py"]
        B2["slurm/extract_xlsr53.sh<br/>src/extract_features.py"]
        B3["slurm/extract_xlsr300m.sh<br/>src/extract_features.py"]
        B4[("artifacts/embeddings/base|xlsr53|xlsr300m/<br/>&lt;lang&gt;_features.pkl · hidden states")]
        B1 --> B4
        B2 --> B4
        B3 --> B4
    end

    %% ---------------- Stage 3 ----------------
    subgraph S3["STAGE 3 · Forced alignment — 1 GPU job"]
        C1["slurm/align.sh<br/>src/precompute_alignments.py<br/>→ src/align.py → src/phonology.py"]
        C2[("artifacts/alignment_cache.pkl<br/>phoneme time-spans per recording")]
        C1 --> C2
    end

    %% ---------------- Stage 4 ----------------
    subgraph S4["STAGE 4 · Probing — orchestrator, run with bash"]
        D0["slurm/pipeline.sh<br/><i>no python · submits 4 sbatch jobs</i>"]
        D1["slurm/probing_worker.sh base<br/>src/run_probing.py"]
        D2["slurm/probing_worker.sh xlsr53<br/>src/run_probing.py"]
        D3["slurm/probing_worker.sh xlsr300m<br/>src/run_probing.py"]
        D4[("results/&lt;TS&gt;/_parts/&lt;model&gt;/<br/>layer_df.csv · xling_df.csv")]
        D5["slurm/probing_merge.sh<br/>src/merge_probing_results.py"]
        D6[("results/&lt;TS&gt;/<br/>figures · CSVs · results_summary.txt")]

        D0 --> D1
        D0 --> D2
        D0 --> D3
        D1 --> D4
        D2 --> D4
        D3 --> D4
        D4 -->|"SLURM --dependency=afterok"| D5
        D5 --> D6
    end

    A2 --> B1
    A2 --> B2
    A2 --> B3
    A2 --> C1
    B4 --> D1
    B4 --> D2
    B4 --> D3
    C2 --> D1
    C2 --> D2
    C2 --> D3
```

## Commands, in order

```bash
bash   slurm/prefetch.sh              # 1. login node — warm the HF cache
sbatch slurm/extract_base.sh     100  # 2. these three can run
sbatch slurm/extract_xlsr53.sh   100  #    at the same time
sbatch slurm/extract_xlsr300m.sh 100
sbatch slurm/align.sh            100  # 3. needs FLEURS audio only
bash   slurm/pipeline.sh         100  # 4. needs stages 2 AND 3
```

> `pipeline.sh` is run with **`bash`, not `sbatch`** — it is an orchestrator that
> calls `sbatch` itself for the 3 workers plus the dependent merge job.

## Shell → Python → output

| Shell script | Python called | Output |
|---|---|---|
| `prefetch.sh` | *(inline heredoc)* | `$WORK/.cache/huggingface/` |
| `extract_base.sh`<br/>`extract_xlsr53.sh`<br/>`extract_xlsr300m.sh` | `src/extract_features.py` | `artifacts/embeddings/<tag>/<lang>_features.pkl` |
| `align.sh` | `src/precompute_alignments.py`<br/>→ `src/align.py` → `src/phonology.py` | `artifacts/alignment_cache.pkl` |
| `pipeline.sh` | *(none — submits jobs)* | `results/<TS>/run_config.txt` |
| `probing_worker.sh` ×3 | `src/run_probing.py`<br/>→ `align.py`, `probing.py`, `phonology.py` | `results/<TS>/_parts/<model>/*.csv` |
| `probing_merge.sh` | `src/merge_probing_results.py`<br/>→ reuses `run_probing.py` figure/table code | `results/<TS>/` final figures + CSVs |

## Notes

- **Stages 2 and 3 are independent** — `align.sh` reads FLEURS audio directly, not the
  embeddings, so it can run at the same time as the extraction jobs. Only stage 4 needs both.
- **Frame→phoneme mapping is always MMS forced alignment.** The old `uniform` mode (split
  each utterance's frames evenly across its phonemes) was removed along with
  `src/precompute_phonemes.py` and `artifacts/phoneme_cache.pkl`: it ignored leading silence
  and real phoneme durations, roughly halving probe scores (voicing EN 0.50 vs 0.86). The
  uniform baseline run is preserved at `results/20260706_21094441/` and in git history.
- **If a worker fails**, `--dependency=afterok` means the merge job never starts (it sits as
  `DependencyNeverSatisfied`). The surviving `_parts/` are reusable — rerun the failed model,
  then submit `probing_merge.sh` by hand.
- **Re-extracting with a different `--max-samples` invalidates the alignment cache** — it is
  keyed per recording, so new utterances have no spans and would be silently skipped. Rerun
  `align.sh` after any change to the extraction sample count.
