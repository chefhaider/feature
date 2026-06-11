"""
Simple linear-probe utilities.

Logistic regression is the right default for probing: it tests whether a feature
is *linearly* readable from a frozen representation (a non-linear probe can find
structure the model doesn't actually expose linearly). We standardize the
features, report accuracy + macro-F1 (the labels are imbalanced, so macro-F1
matters), and always compare against a majority-class baseline.

Usage (X = phoneme embeddings, y = phonological labels):

    from src.probing import evaluate_probe, cross_lingual_probe
    print(evaluate_probe(X, y))                 # within-language
    print(cross_lingual_probe(X_en, y_en, X_de, y_de))   # train EN, test DE
"""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_probe():
    """StandardScaler -> logistic regression."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )


def _scores(y_true, y_pred, majority):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "majority": float(majority),
        "n": len(y_true),
        "n_classes": len(set(y_true)),
    }


def evaluate_probe(X, y, test_size=0.2, random_state=0):
    """Train/test a linear probe within one dataset. Returns a metrics dict."""
    X, y = np.asarray(X), np.asarray(y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    probe = make_probe().fit(X_tr, y_tr)
    majority = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr).score(X_te, y_te)
    return _scores(y_te, probe.predict(X_te), majority)


def cross_lingual_probe(X_train, y_train, X_test, y_test):
    """Train on one language, test on another (zero-shot transfer)."""
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_test, y_test = np.asarray(X_test), np.asarray(y_test)
    probe = make_probe().fit(X_train, y_train)
    majority = (
        DummyClassifier(strategy="most_frequent").fit(X_train, y_train).score(X_test, y_test)
    )
    return _scores(y_test, probe.predict(X_test), majority)
