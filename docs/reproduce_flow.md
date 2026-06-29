# How `reproduce.py` works

```mermaid
flowchart TD
    A([python reproduce.py]) --> B

    B["Step 1 — Extract features
    wav2vec2-base on CPU
    streams 10 utterances × 3 languages from FLEURS
    saves hidden states to reproduce_output/embeddings/"]

    B --> C["Step 2 — Phonemize
    gruut converts transcriptions → IPA
    panphon maps IPA → phonological features
    voicing · nasal · manner · place"]

    C --> D["Step 3 — Probe
    uniform alignment: split frames evenly across phonemes
    logistic regression probe at each layer
    within-language + cross-lingual EN → DE/ES"]

    D --> E["Step 4 — Figures"]

    E --> F["fig1_h2_layers.png
    H2: macro-F1 by layer
    which layers encode phonology best?"]

    E --> G["fig2_h1h3.png
    H1+H3: within-lang vs cross-lingual
    do multilingual features transfer?"]

    E --> H["results_summary.txt
    numeric tables for H1 / H2 / H3"]
```

## Run it

```bash
git clone <repo>
cd BIMAP-FEATURE
pip install -r requirements.txt

python reproduce.py              # ~15 min on CPU
```

## Output

```
reproduce_output/
├── embeddings/
│   ├── en_us_features.pkl
│   ├── de_de_features.pkl
│   └── es_419_features.pkl
├── fig1_h2_layers.png
├── fig2_h1h3.png
└── results_summary.txt
```

## Notes

- No GPU needed — wav2vec2-base runs on CPU in under 15 min with 10 samples
- Uses HF streaming — no 2 GB dataset download, only the audio files needed
- Results at N=10 are directionally correct but noisy; final thesis uses N=100 with forced alignment
