# FEATURE

Do speech models like wav2vec 2.0 encode the physical properties of speech sounds as
language-independent concepts, rather than memorising each language separately?

## Folder structure

```
src/          pipeline code
  extract_features.py       audio -> wav2vec2 hidden states
  phonology.py              text -> IPA -> phonological features
  align.py                  MMS forced alignment, phonemes -> frames
  precompute_alignments.py  builds the alignment cache
  probing.py                linear probes and metrics
  run_probing.py            runs the probes, writes figures and tables
  merge_probing_results.py  merges the per-model runs

slurm/        cluster job scripts
  prefetch.sh               cache models and FLEURS
  extract_*.sh              one extraction job per model
  align.sh                  build the alignment cache
  pipeline.sh               run the probing pipeline
  probing_worker.sh         probes one model
  probing_merge.sh          merges the results

artifacts/    embeddings and cached labels/alignments
results/      one timestamped folder per run (figures, tables, summary)
docs/         notes on the pipeline
notebook/     exploratory notebooks
```
