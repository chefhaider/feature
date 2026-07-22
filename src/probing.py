"""
Simple linear-probe utilities.

Logistic regression is the right default for probing: it tests whether a feature
is *linearly* readable from a frozen representation (a non-linear probe can find
structure the model doesn't actually expose linearly). We standardize the
features, report accuracy + macro-F1 (the labels are imbalanced, so macro-F1
matters), and always compare against a majority-class baseline.

Two things matter for validity:

  * GROUPED splits. Phonemes from the same recording are highly correlated (same
    speaker, same channel). A plain random split puts phonemes from one utterance
    on both sides, so the probe can memorize the recording instead of the
    phonology -- which inflates within-language scores. Pass `groups` (one
    utterance id per phoneme) and the split is made at the utterance level.

  * ERROR BARS. Every score is repeated over several splits (within-language) or
    bootstrapped over the test set (cross-lingual), so each metric comes with a
    `_std`. Differences smaller than the spread are not real.

Usage (X = phoneme embeddings, y = phonological labels, g = utterance ids):

    from src.probing import evaluate_probe, cross_lingual_probe
    print(evaluate_probe(X, y, groups=g))                 # within-language
    print(cross_lingual_probe(X_en, y_en, X_de, y_de))    # train EN, test DE
"""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_METRICS = ("accuracy", "macro_f1", "majority", "majority_acc")


def make_probe():
    """StandardScaler -> logistic regression."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )


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
