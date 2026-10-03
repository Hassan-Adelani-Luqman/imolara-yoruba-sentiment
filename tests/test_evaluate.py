import numpy as np

from src.evaluate import _fast_score, compute_metrics


def test_fast_score_matches_sklearn_metrics():
    rng = np.random.default_rng(0)
    y_true, y_pred = rng.integers(0, 3, 500), rng.integers(0, 3, 500)
    y_pred[:5] = 0  # make sure no class is missing
    full = compute_metrics(y_true, y_pred)
    for metric in ("macro_f1", "weighted_f1", "accuracy"):
        assert abs(_fast_score(y_true, y_pred, metric) - full[metric]) < 1e-4


def test_fast_score_handles_absent_classes():
    y_true, y_pred = np.array([2, 2, 2, 2]), np.array([2, 2, 2, 2])
    assert abs(_fast_score(y_true, y_pred, "macro_f1") - compute_metrics(y_true, y_pred)["macro_f1"]) < 1e-4
