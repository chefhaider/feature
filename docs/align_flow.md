# Stage 3 deep dive — `slurm/align.sh` (forced alignment)

What runs, in what order, and the **type + shape** of every value passed between
functions. All shapes below were verified empirically against the real pipeline.

**Goal of this stage:** answer *"which audio frames does each phoneme occupy?"* and
persist the answer, so probing never needs the raw audio.

---

## Function-call flow

```mermaid
flowchart TD
    SH["<b>slurm/align.sh</b><br/>sbatch · v100 · 4h"]
    --> MAIN["<b>precompute_alignments.py :: main</b>"]

    MAIN --> GA["<b>align.get_aligner</b>(device)"]
    GA --> GAO["<i>returns</i> bundle = 3-tuple<br/>model · tokenizer · aligner<br/>MMS_FA, ~1.18 GB"]

    MAIN --> LD["<b>datasets.load_dataset</b><br/>'google/fleurs', lang, split='train'<br/>.select(range(max_samples))"]
    LD --> LDO["<i>returns</i> Dataset · N rows<br/>each: id int · transcription str<br/>audio.array float64 · audio.sampling_rate int"]

    LDO --> LOOP{{"for each utterance"}}

    LOOP --> PREP["audio = np.asarray(..., float32) → <b>(n_samples,)</b><br/>sr = 16000<br/>alen = n_samples/sr → float sec<br/>key = (lang, id, round(alen,2))"]

    PREP --> AW["<b>align.align_words</b><br/>(audio, sr, transcript, bundle, device)"]

    subgraph AWSUB["inside align_words"]
        W1["<b>_words</b>(transcript)<br/>→ list[str] · diacritics KEPT<br/>['über','die']"]
        W2["<b>_romanize</b>(w) per word<br/>→ str · a-z only<br/>'über' → 'uber'"]
        W3["wav = torch tensor <b>(1, n_samples)</b>"]
        W4["<b>model</b>(wav)<br/>→ emission <b>(1, n_frames, 28)</b><br/>n_frames ≈ 49.8 · seconds"]
        W5["<b>tokenizer</b>(roman_words)<br/>→ list[list[int]] char ids<br/>'tornado' → [7,5,9,4,1,13,5]"]
        W6["<b>aligner</b>(emission[0], tokens)<br/>→ list[list[TokenSpan]]<br/>one list per word"]
        W7["ratio = n_samples / n_frames / sr<br/>≈ 0.0201 s per frame"]
        W1 --> W2 --> W3 --> W4 --> W5 --> W6 --> W7
    end

    AW --> AWSUB
    AWSUB --> AWO["<i>returns</i> word_spans<br/><b>list[(str, float, float)]</b><br/>('tornado', 0.60, 1.02)"]

    AWO --> PS["<b>align.phoneme_spans</b>(word_spans, lang)"]

    subgraph PSSUB["inside phoneme_spans · per word"]
        P1["<b>phonology.phonological_features</b>(word, lang)"]
        P2["<b>text_to_ipa</b> · gruut<br/>'tornado' → 'tɔɹnˈeɪdoʊ' str"]
        P3["<b>ipa_to_features</b> · panphon<br/>word_to_vector_list → list[list[int]] M×24<br/>ipa_segs → list[str] M"]
        P4["→ <b>list[dict] · M entries</b><br/>phoneme·voicing·nasal·manner<br/>place·is_vowel·is_obstruent"]
        P5["step = (t1-t0)/M<br/>split the word's span evenly<br/>across ITS OWN M phonemes"]
        P1 --> P2 --> P3 --> P4 --> P5
    end

    PS --> PSSUB
    PSSUB --> PSO["<i>returns</i> spans<br/><b>list[(dict, float, float)]</b><br/>({'phoneme':'ə',...}, 0.522, 0.542)"]

    PSO --> STORE["cache[key] = spans"]
    STORE --> LOOP

    LOOP -->|"after each language"| CKPT["<b>pickle.dump</b> · checkpoint"]
    CKPT --> OUT[("<b>artifacts/alignment_cache.pkl</b><br/>dict · 300 entries · 32,282 spans<br/>(lang,id,alen) → list[(dict,t0,t1)]")]
```

---

## Types and shapes at each hop

| # | Value | Type | Shape / example |
|---|---|---|---|
| 1 | `audio` | `np.float32` | `(n_samples,)` — 6.8 s → `(108800,)` |
| 2 | `sr` | `int` | `16000` |
| 3 | `alen` | `float` | `6.8` seconds |
| 4 | `key` | `tuple[str,int,float]` | `('en_us', 903, 6.8)` |
| 5 | `_words(...)` | `list[str]` | `['a','tornado','is']` — diacritics kept |
| 6 | `_romanize(w)` | `str` | `'über' → 'uber'` — a–z only |
| 7 | `wav` | `torch.float32` | `(1, n_samples)` |
| 8 | `emission` | `torch.float32` | `(1, n_frames, 28)` — 4 s → `(1, 199, 28)` |
| 9 | `tokenizer(words)` | `list[list[int]]` | `'tornado' → [7,5,9,4,1,13,5]` |
| 10 | `token_spans` | `list[list[TokenSpan]]` | one inner list per word |
| 11 | `ratio` | `float` | `≈0.0201` s/frame (**49.8 fps**) |
| 12 | `word_spans` | `list[(str,float,float)]` | `('tornado', 0.60, 1.02)` |
| 13 | `phonological_features` | `list[dict]` | M dicts, 7 keys each |
| 14 | `spans` | `list[(dict,float,float)]` | `({'phoneme':'ə',…}, 0.522, 0.542)` |
| 15 | **cache** | `dict` | 300 entries → 32,282 spans (~108/utterance) |

---

## Three things worth understanding

**1. Two text forms travel in parallel.** MMS's dictionary is **29 symbols, `a–z` + `'` `-` `*`**
— German/Spanish diacritics raise `KeyError`. But gruut *needs* the original spelling to
phonemize correctly. So `align_words` keeps them paired: the **romanized** form is fed to the
tokenizer, the **original** form is returned with the timestamps and later handed to gruut.

**2. Uniform splitting still happens — but only *within* a word.** `phoneme_spans` divides
each word's aligned span evenly across that word's phonemes. Since a word is short
(~0.4 s / ~5 phonemes) the error is small, unlike the old utterance-level uniform split
(~7 s / ~68 phonemes) which ignored silence entirely.

**3. `alen` is load-bearing, not decoration.** It is part of the cache key *and* the divisor
that converts seconds → frame index later in
`align.segment_dataset`: `f0 = int(t0 / audio_length * T)`. FLEURS reuses `id` across
speakers, so `(lang, id)` alone would collide between two different recordings of the same
sentence — `alen` makes the key recording-unique.

---

## Failure handling

- Per-utterance `try/except`: a failed alignment stores `cache[key] = []` and continues, so one
  bad clip can't kill a multi-hour job. Empty span lists are skipped later by `segment_xy`.
- **Resumable**: an existing cache is loaded at startup and keys already present are skipped.
- **Checkpointed per language**, so a walltime kill loses at most one language's work.

## Verified state

`artifacts/alignment_cache.pkl` — 300 entries (100 × 3 languages), 32,282 phoneme spans,
0 out-of-bounds (`f0 >= T`) cases, 0 empty span-lists.
