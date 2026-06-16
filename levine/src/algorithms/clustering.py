from __future__ import annotations

import numpy as np
from typing import List


def merge_close(points: np.ndarray, radius: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0,) if points.ndim == 1 else (0, points.shape[1]))

    points = np.asarray(points, dtype=float)

    if points.ndim == 1:
        return _merge_1d(points, radius)
    else:
        return _merge_nd(points, radius)


def _merge_1d(points: np.ndarray, radius: float) -> np.ndarray:
    pts = np.sort(points)
    merged: List[float] = [float(pts[0])]
    counts: List[int] = [1]

    for p in pts[1:]:
        if abs(p - merged[-1]) <= radius:
            n = counts[-1]
            merged[-1] = (merged[-1] * n + p) / (n + 1)
            counts[-1] = n + 1
        else:
            merged.append(float(p))
            counts.append(1)

    return np.array(merged)


def _merge_nd(points: np.ndarray, radius: float) -> np.ndarray:
    centers: List[np.ndarray] = []
    counts: List[int] = []

    for p in points:
        if not centers:
            centers.append(p.copy())
            counts.append(1)
            continue

        dists = np.linalg.norm(np.array(centers) - p, axis=1)
        min_idx = np.argmin(dists)

        if dists[min_idx] <= radius:
            n = counts[min_idx]
            centers[min_idx] = (centers[min_idx] * n + p) / (n + 1)
            counts[min_idx] = n + 1
        else:
            centers.append(p.copy())
            counts.append(1)

    return np.array(centers) if centers else np.empty((0, points.shape[1]))


def agglomerative_merge(points: np.ndarray, radius: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    if len(points) <= 1:
        return np.array(points)

    points = np.atleast_2d(points)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=radius,
        linkage="complete",
    )
    labels = clustering.fit_predict(points)

    centers = []
    for label in np.unique(labels):
        mask = labels == label
        centers.append(points[mask].mean(axis=0))

    return np.array(centers)
