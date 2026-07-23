"""
Linear-probe utilities.

Logistic regression tests whether a feature is linearly readable from a frozen
representation. Features are standardized; we report accuracy and macro-F1 (labels
are imbalanced) against a majority-class baseline.

Splits are grouped by utterance: phonemes from one recording are correlated, so a
plain random split would let the probe memorize the recording rather than the
phonology. Scores are repeated over several splits (within-language) or
bootstrapped over the test set (cross-lingual), giving each metric a `_std`.

Usage (X = phoneme embeddings, y = phonological labels, g = utterance ids):

    from src.probing import evaluate_probe, cross_lingual_probe
    print(evaluate_probe(X, y, groups=g))                 # within-language
    print(cross_lingual_probe(X_en, y_en, X_de, y_de))    # train EN, test DE
"""
from collections import Counter

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, KFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_METRICS = ("accuracy", "macro_f1", "majority", "majority_acc")


def make_probe():
    """StandardScaler -> logistic regression."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )


# --- H3 paired k-fold transfer-gap test ---
def _macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _majority_macro_f1(y_train, y_test):
    """macro-F1 of always predicting language A's majority class on the test set."""
    maj = Counter(y_train).most_common(1)[0][0]
    return _macro_f1(y_test, np.full(len(y_test), maj))


def _norm(f1, base):
    """Baseline-normalized score: 0 = majority baseline, 1 = perfect. Makes gaps
    comparable across features, which differ in class count and baseline."""
    denom = 1.0 - base
    return float((f1 - base) / denom) if denom > 1e-9 else np.nan


def paired_gap_kfold(Xa, ya, ga, others, k=5, repeats=5, random_state=0,
                     min_train=8, min_test=4):
    """Paired within-vs-cross-lingual transfer gap for one (feature, train-lang A).

    For each of k folds (split of A's utterances, repeated `repeats` times):
      within_i = macro-F1 on A's held-out utterances,
      cross_i  = mean macro-F1 on every other language in `others`,
      gap_i    = within_i - cross_i.

    Within and cross share the same probe and training set, so the gap is paired and
    comes from held-out folds.

    Xa, ya : (N, D), (N,) phonemes of language A at one layer/feature.
    ga     : (N,) utterance id per phoneme, used for fold grouping.
    others : list of (Xb, yb) for the test languages.
    Returns a list of per-fold dicts, empty if the data is too small.
    """
    Xa, ya, ga = np.asarray(Xa), np.asarray(ya), np.asarray(ga)
    if len(ya) < 2 * k or len(set(ya)) < 2 or not others:
        return []
    uniq = np.array(sorted(set(ga.tolist())))
    if len(uniq) < k:
        return []

    rng = np.random.RandomState(random_state)
    out = []
    for rep in range(repeats):
        kf = KFold(n_splits=k, shuffle=True, random_state=rng.randint(1 << 30))
        for fold, (tr_u, te_u) in enumerate(kf.split(uniq)):
            train_g, test_g = set(uniq[tr_u]), set(uniq[te_u])
            trm = np.fromiter((g in train_g for g in ga), bool, len(ga))
            tem = np.fromiter((g in test_g for g in ga), bool, len(ga))
            if trm.sum() < min_train or tem.sum() < min_test:
                continue
            if len(set(ya[trm])) < 2 or len(set(ya[tem])) < 2:
                continue

            probe = make_probe().fit(Xa[trm], ya[trm])
            within = _macro_f1(ya[tem], probe.predict(Xa[tem]))
            within_base = _majority_macro_f1(ya[trm], ya[tem])

            c_vals, c_bases = [], []
            for Xb, yb in others:
                Xb, yb = np.asarray(Xb), np.asarray(yb)
                if len(yb) < min_test or len(set(yb)) < 2:
                    continue
                c_vals.append(_macro_f1(yb, probe.predict(Xb)))
                c_bases.append(_majority_macro_f1(ya[trm], yb))
            if not c_vals:
                continue
            cross, cross_base = float(np.mean(c_vals)), float(np.mean(c_bases))

            out.append({
                "rep": rep, "fold": fold,
                "within": within, "cross": cross, "gap": within - cross,
                "within_norm": _norm(within, within_base),
                "cross_norm": _norm(cross, cross_base),
                "gap_norm": _norm(within, within_base) - _norm(cross, cross_base),
            })
    return out


def _split_scores(y_true, y_pred, y_majority):
    """Metrics for a single train/test split (or bootstrap resample)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        # Baseline reported as macro-F1 so it pairs with the headline macro_f1.
        # A most-frequent classifier scores 0 F1 on the minority class, so on
        # imbalanced labels its macro-F1 is far below its accuracy.
        "majority": float(f1_score(y_true, y_majority, average="macro", zero_division=0)),
        "majority_acc": float(accuracy_score(y_true, y_majority)),
    }


def _aggregate(per_split, n, n_classes):
    """mean + std across splits/bootstraps."""
    out = {}
    for k in _METRICS:
        vals = [d[k] for d in per_split]
        out[k] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    out["n"] = int(n)
    out["n_classes"] = int(n_classes)
    out["n_splits"] = len(per_split)
    return out


def evaluate_probe(X, y, groups=None, n_repeats=5, test_size=0.2, random_state=0):
    """Within-language probe over repeated splits. Returns mean +- std metrics.

    If `groups` is given (one id per sample, e.g. the utterance a phoneme came
    from), splits are GROUPED so no utterance appears in both train and test.
    """
    X, y = np.asarray(X), np.asarray(y)

    if groups is not None:
        groups = np.asarray(groups)
        splitter = GroupShuffleSplit(n_splits=n_repeats, test_size=test_size,
                                     random_state=random_state)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size,
                                          random_state=random_state)
        split_iter = splitter.split(X, y)

    per_split = []
    for tr, te in split_iter:
        y_tr, y_te = y[tr], y[te]
        # a grouped split can land all of one class on one side -- skip those
        if len(set(y_tr)) < 2 or len(y_te) == 0:
            continue
        probe = make_probe().fit(X[tr], y_tr)
        dummy = DummyClassifier(strategy="most_frequent").fit(X[tr], y_tr)
        per_split.append(_split_scores(y_te, probe.predict(X[te]), dummy.predict(X[te])))

    if not per_split:
        return None
    return _aggregate(per_split, len(y), len(set(y)))


def cross_lingual_probe(X_train, y_train, X_test, y_test, n_boot=200, random_state=0):
    """Zero-shot transfer: train on one language, test on another.

    Train and test are already disjoint (different languages), so no split is
    needed; error bars come from bootstrapping the test set. The reported point
    estimate is the full-test score, `_std` is the bootstrap spread.
    """
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_test, y_test = np.asarray(X_test), np.asarray(y_test)

    probe = make_probe().fit(X_train, y_train)
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    pred = probe.predict(X_test)
    maj = dummy.predict(X_test)

    out = _aggregate([_split_scores(y_test, pred, maj)], len(y_test), len(set(y_test)))

    rng = np.random.default_rng(random_state)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_test), len(y_test))
        if len(set(y_test[idx])) < 2:
            continue
        boots.append(_split_scores(y_test[idx], pred[idx], maj[idx]))
    if boots:
        for k in _METRICS:
            out[f"{k}_std"] = float(np.std([b[k] for b in boots]))
    return out
