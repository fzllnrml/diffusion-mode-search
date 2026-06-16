from dataclasses import dataclass
from scipy.spatial.distance import cdist

import numpy as np


@dataclass
class ModeMetrics:
    recall: float
    precision: float
    f1: float
    soft_error: float
    n_true: int
    n_found: int
    n_hits: int
    n_correct: int
    per_mode_dist: np.ndarray


def _ensure_2d(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _pairwise_distances(a, b):
    return cdist(a, b, metric='euclidean')


def compute_recall(true_modes, found_modes, eps_hit):
    true_modes = _ensure_2d(np.asarray(true_modes, dtype=float))
    found_modes = _ensure_2d(np.asarray(found_modes, dtype=float))

    K = true_modes.shape[0]
    if found_modes.shape[0] == 0:
        return 0.0, np.zeros(K, dtype=bool), np.full(K, np.inf)

    dists = _pairwise_distances(true_modes, found_modes)
    min_dists = dists.min(axis=1)
    hits = min_dists <= eps_hit

    return float(hits.mean()), hits, min_dists


def compute_precision(true_modes, found_modes, eps_hit):
    true_modes = _ensure_2d(np.asarray(true_modes, dtype=float))
    found_modes = _ensure_2d(np.asarray(found_modes, dtype=float))

    M = found_modes.shape[0]
    if M == 0:
        return 0.0, np.zeros(0, dtype=bool)

    dists = _pairwise_distances(found_modes, true_modes)
    min_dists = dists.min(axis=1)
    correct = min_dists <= eps_hit

    return float(correct.mean()), correct


def compute_soft_error(true_modes, found_modes):
    true_modes = _ensure_2d(np.asarray(true_modes, dtype=float))
    found_modes = _ensure_2d(np.asarray(found_modes, dtype=float))

    K = true_modes.shape[0]
    if found_modes.shape[0] == 0:
        return float("inf"), np.full(K, np.inf)

    dists = _pairwise_distances(true_modes, found_modes)
    per_mode = dists.min(axis=1)

    return float(per_mode.mean()), per_mode


def evaluate_modes(true_modes, found_modes, eps_hit=1.0):
    recall, hits, min_dists = compute_recall(true_modes, found_modes, eps_hit)
    precision, correct = compute_precision(true_modes, found_modes, eps_hit)
    soft_err, per_mode = compute_soft_error(true_modes, found_modes)

    if recall + precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0.0

    return ModeMetrics(
        recall=recall,
        precision=precision,
        f1=f1,
        soft_error=soft_err,
        n_true=len(_ensure_2d(np.asarray(true_modes))),
        n_found=len(_ensure_2d(np.asarray(found_modes))),
        n_hits=int(hits.sum()),
        n_correct=int(correct.sum()) if len(correct) > 0 else 0,
        per_mode_dist=per_mode,
    )
