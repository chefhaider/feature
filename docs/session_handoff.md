# Session Handoff — FEATURE project

A self-contained summary of this working session so it can be exported into a new
session. Covers project context, decisions made, files created/changed, current
state, known issues, and next steps.

---

## 1. Project context

**FEATURE** — MSc AI thesis. Compares wav2vec / wav2vec 2.0 model variants on how
well they encode **phonological features** (voicing, nasality, place, manner)
**across languages**, and whether some languages share latent phonological
structure.

- **Models:** `facebook/wav2vec2-base` (mono, 768-d, 13 layers),
  `facebook/wav2vec2-large-xlsr-53` and `facebook/wav2vec2-xls-r-300m`
  (multilingual, 1024-d, 25 layers).
- **Dataset:** Google **FLEURS**. Languages so far: `en_us`, `de_de`, `es_419`.
- **Method:** extract hidden states → label phonemes → probe (logistic regression).

### Hypotheses (final, three)
- **H1** — multilingual models encode phonology more *language-independently* than
  monolingual base (better cross-lingual transfer); fine-tuning can improve it.
- **H2** — phonological information is concentrated in *specific layers* (not uniform).
- **H3** — some features/languages transfer well, others are language-specific
  (place expected to be the language-specific one); some languages share latent structure.

Mapping: **H1 = across models, H2 = across layers, H3 = across languages/features.**

---

## 2. Environment & workflow (IMPORTANT)

- User edits **locally on Windows**; Claude Code runs locally and **cannot run
  GPU/training jobs**. Work is **pushed to the cluster** and run there.
- Cluster: **NHR@FAU** (https://doc.nhr.fau.de). Two clusters share `$HOME`
  (`/home/hpc`) and `$WORK` (`/home/woody`):
  - **TinyGPU**: set `--partition=v100` + `--gres=gpu:v100:1`.
  - **Alex**: do **NOT** set `--partition` (causes "invalid partition name") — it's
    auto-selected from `--gres` (e.g. `--gres=gpu:a40:1`). Use `--export=NONE` +
    `unset SLURM_EXPORT_ENV`.
- Conda env `feature` (torch 2.6.0+cu124 — **bundles CUDA, so NO `module load cuda`**;
  `cuda/12.2` does not exist on TinyGPU).
- Project dir on cluster: `$WORK/BIMAP-FEATURE`. HF cache: `$WORK/.cache/huggingface`.
- A prior extraction job **timed out 4h downloading via HF Xet** → always set
  `HF_HUB_DISABLE_XET=1` and prefetch on a login node.

### ⚠️ Recurring problem to watch
Multiple times this session, **local edits were lost on the cluster** because the
working tree was reset (a `git checkout`/`reset`/`clean`, or auto-commits only
staging some files). Symptom: cluster ran old code (e.g. espeak error after the
gruut rewrite). **Always commit explicitly and confirm the cluster pulled it**
(`grep` for a known string in the file).

---

## 3. Repository structure (after this session)

```
src/
  extract_features.py   # GPU: FLEURS audio -> wav2vec2 hidden states (.pkl). CLI: --model, --max-samples, ...
  phonology.py          # text -> IPA (gruut) -> distinctive features (panphon) -> per-phoneme labels
  align.py              # MMS forced aligner: phoneme -> audio frames (accurate path; needs audio; not validated yet)
  probing.py            # linear probe: evaluate_probe, cross_lingual_probe (+ majority baseline, macro-F1)
slurm/
  prefetch.sh           # run on LOGIN node first: caches all models + FLEURS into $WORK
  extract_base.sh       # TinyGPU, v100, per-model output dir embeddings/base
  extract_xlsr53.sh     # -> embeddings/xlsr53
  extract_xlsr300m.sh   # -> embeddings/xlsr300m
  extract_base_alex.sh  # Alex (a40) test variant -> embeddings/alex_test/base
  get_embeddings.sh     # original generic script (superseded)
notebook/
  segment_probing copy.ipynb   # THE main notebook (steps 1-4); user runs this
  visualize_embeddings.ipynb   # orientation: PCA/t-SNE/silhouette of mean-pooled embeddings
  evaluation.ipynb / pipeline.ipynb / plot.ipynb   # original MVP (utterance-level; superseded)
docs/
  probing_explained.md  # plain-language explainer (feature extraction + probing)
  session_handoff.md    # this file
artifacts/              # user-created; holds proj_description.md, experiments_phase1/2.csv, flo.md, phon_features.pkl
requirements.txt        # added: gruut[de,es], panphon, torchaudio
```

Structural refactor done early: `scripts/` → `src/` (+ `__init__.py`), notebooks →
`notebook/`, and each notebook has a setup cell that `chdir`s to project root so
`from src... import` works.

---

## 4. Key methodology decisions

### Phonological features (replacing the broken grapheme pseudo-G2P)
Four features, from real IPA + panphon distinctive features, each with a **segment filter**:

| feature | classes | probe on |
|---|---|---|
| voicing | voiced / voiceless | **obstruents only** |
| nasal | nasal / oral | all segments |
| manner | plosive / fricative / affricate / nasal / approximant | consonants |
| place | labial / coronal / dorsal / laryngeal | consonants |

**Why the filter matters:** sonorants/vowels are ~all voiced, so probing voicing on
everything gives a near-constant label (the MVP's failure → stuck at chance).
Restricting to obstruents made it 76/24 → **63/37 balanced**.

### G2P backend = gruut (NOT phonemizer/espeak)
espeak-ng is a **system binary** the cluster's conda mirror doesn't have (no root to
install). Switched to **gruut** — pure-Python, pip-installable, multilingual:
```
pip install "gruut[de,es]" panphon
```
Only `text_to_ipa` in `phonology.py` changed; panphon and everything downstream are
unchanged.

### Probe = logistic regression (correct default)
Probing tests whether info is *linearly* accessible — an MLP would conflate
"present" with "extractable". Report accuracy + **macro-F1** (labels imbalanced) +
**majority baseline** always.

### Alignment
- **Accurate:** `src/align.py` uses torchaudio **MMS forced aligner** (word spans →
  split into phonemes → slice frames). Needs raw audio; **not validated yet**.
- **In the notebook (what runs now):** **uniform alignment** — split each
  utterance's frames evenly across its phonemes. Approximate but needs only the
  saved `.pkl`s (no audio/download).

---

## 5. The pipeline (how it all fits)

```
AUDIO  --extract_features.py (GPU)-->  hidden states .pkl  (list of L layers, each (1, T, D))
TEXT   --phonology.py (gruut+panphon)-->  per-phoneme labels (voicing/nasal/manner/place)
                 |
   segment_xy (uniform align): split T frames across phonemes, mean-pool each phoneme's
   slice at a chosen layer, keep only feature-relevant segments
                 v
   X = (all phonemes, D),  y = (all phonemes,)   <- pooled over many sentences
                 |
   probing.py: StandardScaler -> LogisticRegression -> accuracy / macro-F1 / majority
                 v
   H1 (cross-lingual transfer) | H2 (per layer) | H3 (transfer gap per feature)
```

Old MVP vs now: **one mean-pooled vector + one majority label per sentence** →
**one vector + one label per phoneme** (5 sentences × 3 langs = 1,642 phonemes vs 15).

---

## 6. Current state

- **Embeddings:** extracted (user "has all the embeddings"). Per-model dirs under
  `embeddings/<tag>/<lang>_features.pkl`.
- **`segment_probing copy.ipynb`** — **step 1 ran successfully on 5 samples/language**
  (gruut + panphon working; 1,642 phonemes; voicing-on-obstruents balanced 63/37).
  Steps 2–4 (uniform-align → probe → evaluate H1/H2/H3) have been **added but not yet
  run** by the user.
- Saved `artifacts/phon_features.pkl` (the step-1 per-phoneme table).

### Notebook step layout (`segment_probing copy.ipynb`)
1. **Step 1 (done):** load transcriptions, phonemize, show label distributions + segment-filter effect.
2. **Step 2:** `segment_xy` helpers (uniform alignment); `MODELS` points at `embeddings/<tag>`; `N_PROBE=30`.
3. **Step 3:** experiment loop → `layer_df`, `within_df`, `xling_df`.
4. **Step 4a (H2):** macro-F1 vs layer plot. **4b (H1):** cross-lingual transfer table + majority. **4c (H3):** `transfer_gap` per feature.

---

## 7. Known issues / gotchas

- **Code sync to cluster keeps failing** (see §2). Commit + verify before running.
- **FLEURS re-download hang:** the notebook kernel must use `$WORK` HF cache, set
  `HF_HOME`/`HF_DATASETS_CACHE` before importing `datasets`, and restart the kernel.
  Alternative: read transcriptions from the saved `.pkl`s (no FLEURS at all) — this
  is what step-1b was switched to.
- **Step 1 is CPU-only** (gruut/panphon) — GPU does not speed it up. GPU only helps
  MMS alignment (step 2 accurate path) and extraction.
- **Memory:** loading large-model `.pkl`s (all layers) is heavy; lower `N_PROBE` or
  run `base` first if OOM.
- Small N + uniform alignment = **noisy, slightly pessimistic numbers** — a working
  pipeline, not final results.

---

## 8. Next steps (suggested)

1. **Run steps 2–4** of `segment_probing copy.ipynb` (set `MODELS` paths, maybe raise
   `N_PROBE` to 60–100). Read H1/H2/H3 tables.
2. **Validate `src/align.py`** (MMS) on the cluster and swap it in for `segment_xy`'s
   uniform split → accurate phoneme boundaries (needs FLEURS audio cached).
3. Scale up sample size once the pipeline is trusted.
4. (Optional / in description but not built) **fine-tuning** experiments for H1.

---

## 9. Pending commit

End-of-session, these were **not yet committed** and should be (then pulled on the
cluster):
- `src/phonology.py` (gruut backend), `requirements.txt`
- `notebook/segment_probing copy.ipynb` (steps 2–4)
- `docs/probing_explained.md`, `docs/session_handoff.md`
