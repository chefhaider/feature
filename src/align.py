"""
Forced alignment bridge: map phonemes to wav2vec2 frames so the phonological
labels from `phonology.py` can be attached to embeddings for segment-level probing.

Approach (kept deliberately simple):
    1. torchaudio's MMS forced aligner gives WORD time spans for an utterance.
    2. Each word span is split uniformly across that word's phonemes
       (phonemizer/panphon via phonology.py) -> per-phoneme time spans.
    3. For a saved hidden-state layer, slice the frames inside each phoneme span
       and mean-pool -> one embedding per phoneme, with its feature label.

The raw FLEURS audio is needed for step 1 (the saved .pkl holds only hidden
states), so run this where the FLEURS audio is available/cached.

NOTE: the MMS calls in get_aligner()/align_words() follow the torchaudio
"Forced alignment for multilingual data" tutorial API -- validate once on the
cluster (torchaudio version, dictionary coverage for de/es diacritics).
"""
import re

import numpy as np
import torch

from src.phonology import phonological_features, keep_for


def get_aligner(device="cpu"):
    """Load the MMS forced-alignment model + tokenizer + aligner."""
    from torchaudio.pipelines import MMS_FA as bundle

    model = bundle.get_model(with_star=False).to(device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    return model, tokenizer, aligner


def _normalize(text):
    """Lowercase and keep letters/spaces -- MMS expects clean word tokens."""
    text = text.lower()
    text = re.sub(r"[^a-zäöüßàáéíóúñ ]", " ", text)
    return text.split()


def align_words(waveform, sample_rate, transcript, aligner_bundle, device="cpu"):
    """Return [(word, start_s, end_s), ...] for one utterance.

    waveform: 1-D float tensor/array of audio at `sample_rate`.
    """
    model, tokenizer, aligner = aligner_bundle
    words = _normalize(transcript)
    if not words:
        return []

    wav = torch.as_tensor(waveform, dtype=torch.float32).reshape(1, -1).to(device)
    with torch.inference_mode():
        emission, _ = model(wav)
        token_spans = aligner(emission[0], tokenizer(words))

    num_frames = emission.size(1)
    ratio = wav.size(1) / num_frames / sample_rate  # seconds per emission frame
    out = []
    for spans, word in zip(token_spans, words):
        out.append((word, spans[0].start * ratio, spans[-1].end * ratio))
    return out


def phoneme_spans(word_spans, lang):
    """Split each word span across its phonemes -> [(label_dict, start_s, end_s)]."""
    spans = []
    for word, t0, t1 in word_spans:
        feats = phonological_features(word, lang)
        if not feats:
            continue
        step = (t1 - t0) / len(feats)
        for i, f in enumerate(feats):
            spans.append((f, t0 + i * step, t0 + (i + 1) * step))
    return spans


def segment_dataset(hidden_layer, audio_length, spans, feature):
    """Build (X, y) for one feature from one utterance.

    hidden_layer: array (1, T, D) for the chosen layer of this utterance.
    audio_length: utterance duration in seconds (saved alongside the embeddings).
    spans:        output of phoneme_spans().
    feature:      one of phonology.FEATURES; the segment filter is applied.
    """
    h = hidden_layer[0]  # (T, D)
    T = h.shape[0]
    X, y = [], []
    for seg, t0, t1 in spans:
        if not keep_for(feature, seg):
            continue
        f0 = int(t0 / audio_length * T)
        f1 = max(f0 + 1, int(t1 / audio_length * T))
        X.append(h[f0:f1].mean(axis=0))
        y.append(seg[feature])
    return np.array(X), np.array(y)
