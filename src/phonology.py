
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

LANG_CODE: Dict[str, str] = {
    "en_us": "en-us",
    "de_de": "de-de",
    "es_419": "es-es",
}

FEATURES = ["voicing", "nasal", "manner", "place"]

SEGMENT_FILTERS = {
    "voicing": "obstruent",
    "nasal": "all",
    "manner": "consonant",
    "place": "consonant",
}


# --------------------------------------------------------------------------- #
# text -> IPA
# --------------------------------------------------------------------------- #
def text_to_ipa(text: str, lang: str) -> str:
    """Phonemize text into an IPA string using gruut (pure-Python, no system deps)."""
    from gruut import sentences

    code = LANG_CODE.get(lang, lang)
    phones: List[str] = []
    for sent in sentences(text, lang=code):
        for word in sent:
            if word.phonemes:
                phones.extend(word.phonemes)
    return "".join(phones)


@lru_cache(maxsize=1)
def _feature_table():
    import panphon

    return panphon.FeatureTable()


def ipa_to_features(ipa: str) -> List[Dict[str, Optional[str]]]:
    """Segment an IPA string and return one label dict per phoneme.

    Each dict has: phoneme (the IPA symbol), voicing, nasal, manner, place (any
    label may be None where undefined, e.g. manner/place for a vowel) plus the
    boolean flags is_vowel / is_obstruent used by the segment filters.
    """
    ft = _feature_table()
    ipa = ipa.replace(" ", "")
    vectors = ft.word_to_vector_list(ipa, numeric=True)
    names = ft.names
    segs = ft.ipa_segs(ipa)  # IPA symbols, aligned with `vectors`

    segments: List[Dict[str, Optional[str]]] = []
    for i, vec in enumerate(vectors):
        f = dict(zip(names, vec))
        segments.append(
            {
                "phoneme": segs[i] if i < len(segs) else "",
                "voicing": _voicing(f),
                "nasal": _nasal(f),
                "manner": _manner(f),
                "place": _place(f),
                "is_vowel": _is_vowel(f),
                "is_obstruent": _is_obstruent(f),
            }
        )
    return segments


def phonological_features(text: str, lang: str) -> List[Dict[str, Optional[str]]]:
    """Convenience: transcription -> per-phoneme phonological label dicts."""
    return ipa_to_features(text_to_ipa(text, lang))


def keep_for(feature: str, seg: Dict[str, Optional[str]]) -> bool:
    """Whether `seg` should be included when probing `feature` (applies the
    recommended segment filter and drops segments with an undefined label)."""
    if seg.get(feature) is None:
        return False
    filt = SEGMENT_FILTERS[feature]
    if filt == "obstruent":
        return bool(seg["is_obstruent"])
    if filt == "consonant":
        return not bool(seg["is_vowel"])
    return True  # "all"

def _is_vowel(f: Dict[str, int]) -> bool:
    return f.get("syl", -1) == 1


def _is_obstruent(f: Dict[str, int]) -> bool:
    return f.get("son", 0) == -1


def _voicing(f: Dict[str, int]) -> str:
    return "voiced" if f.get("voi", 0) == 1 else "voiceless"


def _nasal(f: Dict[str, int]) -> str:
    return "nasal" if f.get("nas", 0) == 1 else "oral"


def _manner(f: Dict[str, int]) -> Optional[str]:
    if _is_vowel(f):
        return None
    if f.get("nas", 0) == 1:
        return "nasal"
    if f.get("son", 0) == -1:  # obstruent
        if f.get("delrel", 0) == 1:
            return "affricate"
        if f.get("cont", 0) == -1:
            return "plosive"
        return "fricative"
    return "approximant"  # non-nasal sonorant consonant (liquids, glides, trills)


def _place(f: Dict[str, int]) -> Optional[str]:
    if _is_vowel(f):
        return None
    if f.get("lab", 0) == 1:
        return "labial"
    if f.get("cor", 0) == 1:
        return "coronal"
    if f.get("hi", 0) == 1 or f.get("back", 0) == 1:
        return "dorsal"  # velar/uvular/palatal: heuristic (panphon has no single 'dor')
    return "laryngeal"  # glottal/pharyngeal: no oral place of articulation
