# How Phonological Feature Extraction & Probing Work

A plain-language explanation of the segment-level pipeline: how phonological
features are extracted, how the wav2vec2 embeddings are used, and how probing is
implemented.

---

## 1. Phonological feature extraction — from spelling to sound

### Before (the MVP)
- Walked through each sentence **letter by letter** (a fake "G2P" on spelling).
- Looked features up in a small hand-coded letter dictionary (buggy, only voicing + manner).
- Took a **majority vote** over the whole sentence → **one label per sentence**.

Problems:
- **Spelling ≠ sound.** "the" is pronounced `ð ə` (no t, no h); "quick" starts with a `k`. Worse across languages (German "v" → `f`, Spanish "z" → `θ`).
- **One label per sentence** — but phonological features are properties of *individual sounds*, so this measured the wrong unit (voicing was stuck at chance).

### Now
Pipeline: `transcription → IPA phonemes (gruut) → distinctive features (panphon) → per-phoneme labels`

- **gruut** converts text into real **IPA phonemes**, language-aware (en-us, de-de, es-es).
- **panphon** gives each phoneme its articulatory **distinctive features**, from which we derive four labels: **voicing, nasal, manner, place** (+ flags `is_vowel`, `is_obstruent`).
- Output is **one row per phoneme**, not one per sentence.

Example — *"the quick brown fox jumps"* → 20 analyzed sounds:

| phoneme | voicing | nasal | manner | place |
|---|---|---|---|---|
| ð | voiced | oral | fricative | coronal |
| k | voiceless | oral | plosive | dorsal |
| dʒ | voiced | oral | affricate | coronal |
| … | | | | |

### Before → After

| | MVP | Now |
|---|---|---|
| Unit | letters (spelling) | IPA phonemes (real sounds) |
| Labels | 1 per **sentence** (majority vote) | 1 per **phoneme** |
| Features | voicing, manner | voicing, nasal, **place**, manner |
| Feature source | hand-coded letter dict (buggy) | panphon distinctive features |
| Cross-lingual | ✗ spelling differs per language | ✓ IPA is universal |

### The key fix: segment filters
Even with real phonemes, probing *voicing* on every sound fails — vowels, nasals
and approximants are all voiced, so the label is nearly constant. So each feature
is probed only on the segments where it is a real contrast (`keep_for` /
`SEGMENT_FILTERS`):

- **voicing → obstruents only** (stops/fricatives/affricates)
- **manner / place → consonants**
- **nasal → all segments**

Effect on voicing: 76/24 (degenerate) → **63/37 (balanced, learnable)**.

### Impact
5 sentences × 3 languages → **1,642 labeled phonemes** (vs. 15 sentence-labels
before) → enough real, contrastive data to probe.

---

## 2. The embeddings — what shape they come in

When wav2vec2 runs on a clip, it turns each **20 ms slice of audio** into a vector
of numbers (the embedding = the model's "understanding" of that moment). These are
saved in the `.pkl` files.

Per sentence, the saved `hidden_states` is:

```
a list of L layers, each of shape (1, T, D)
```

| symbol | meaning | base | large (xlsr) |
|---|---|---|---|
| L | number of layers | 13 | 25 |
| 1 | batch (one clip) | 1 | 1 |
| T | frames (~50 per second of audio) | varies | varies |
| D | embedding size | 768 | 1024 |

Pick one layer and drop the batch dim → `(T, 768)`: T frames, each a 768-vector.

- **Old way:** average over all T frames → `(768,)`, **one vector per sentence**.
- **New way:** average each phoneme's *slice* of frames → `(768,)`, **one vector per phoneme**.

---

## 3. How probing is implemented

### Per sentence (extracting examples)
For *"the cat"* (phonemes `ð ə k æ t`), with the sentence's layer embeddings `(T=100, 768)`:

1. **Uniform align** — 100 frames ÷ 5 phonemes = 20 frames each:
   `ð→0–20, ə→20–40, k→40–60, æ→60–80, t→80–100`.
   *(This is the simple/approximate method. The accurate upgrade is forced
   alignment via `src/align.py` (MMS), which first splits the sentence into words,
   then phonemes — it needs the raw audio.)*
2. **Pool each phoneme's frames → one vector** `(768,)`, attach its label.
3. **Filter for the feature.** For voicing, keep obstruents only (vowels drop out):

```
X (this sentence) = [vec_ð, vec_k, vec_t]   shape (3, 768)
y (this sentence) = [voiced, voiceless, voiceless]
```

This is the inner loop of `segment_xy()`.

### Across all sentences (the actual probe)
We do **not** probe a single sentence. We repeat the above for **every sentence in
every language** and pile all phoneme rows into one big table:

```
X = (all phonemes, 768)   ← thousands of rows
y = (all phonemes,)
```

Then run the probe **once** on that pile:

```
evaluate_probe(X, y)  →  StandardScaler → LogisticRegression → accuracy / macro-F1
```

- The classifier is still simple **logistic regression** (probing tests whether the
  feature is *linearly* readable from the embedding). What changed is that `(X, y)`
  is now built per-phoneme instead of per-sentence — far more, far better examples.
- Each feature (voicing, nasal, manner, place) gets its own pile and its own probe.
- We repeat **per layer** (which layer encodes it best → H2) and **per language
  pair** for transfer (train EN → test DE/ES → H1/H3).

### One-line summary
> Old = one label per sentence. New = one label per **sound**: we cut each
> sentence's embeddings into per-phoneme slices, average each into a vector, pool
> all phonemes from all sentences into one big `(X, y)`, and train a single linear
> probe on the pile.

---

## 4. How it maps to the hypotheses

| Hypothesis | What it predicts | How it's tested |
|---|---|---|
| **H1** | multilingual models encode phonology more language-independently | cross-lingual transfer (train EN → test DE/ES), base vs xlsr |
| **H2** | phonology concentrated in specific layers | probe accuracy vs layer |
| **H3** | some features/languages transfer, others are language-specific | `transfer_gap` per feature (place = language-specific) |
