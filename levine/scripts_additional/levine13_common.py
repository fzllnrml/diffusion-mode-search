#!/usr/bin/env python3
"""Shared utilities for Levine 13D evaluation."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class LevineData:
    X: np.ndarray
    y: np.ndarray
    label_names: np.ndarray
    population_by_label_index: Dict[int, str]
    unassigned_id: Optional[int]


def parse_csv_list(value: str, cast=float) -> List[Any]:
    return [cast(x.strip()) for x in str(value).split(",") if x.strip()]


def load_population_map(path: Path) -> Dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"Population names file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    lower = {str(c).lower(): c for c in df.columns}
    if "label" not in lower or "population" not in lower:
        raise ValueError(f"Unexpected population file columns: {df.columns.tolist()}")
    return {
        int(row[lower["label"]]): str(row[lower["population"]])
        for _, row in df.iterrows()
    }


def raw_label_to_population(raw_label: Any, pop_map: Mapping[int, str]) -> str:
    s = str(raw_label).strip()
    if s.lower() == "unassigned":
        return "unassigned"
    try:
        label_id = int(float(s))
    except (TypeError, ValueError):
        return f"bad_label_{s}"
    return str(pop_map.get(label_id, f"unknown_label_{label_id}"))


def load_levine_data(data_path: Path, pop_path: Path) -> LevineData:
    if not data_path.exists():
        raise FileNotFoundError(f"Processed Levine data not found: {data_path}")
    d = np.load(data_path, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    y = np.asarray(d["y"], dtype=np.int64)
    label_names = np.array([str(x) for x in d["label_names"]], dtype=object)
    pop_map = load_population_map(pop_path)
    population_by_label_index = {
        idx: raw_label_to_population(raw, pop_map)
        for idx, raw in enumerate(label_names)
    }
    unassigned = [idx for idx, name in population_by_label_index.items() if name == "unassigned"]
    return LevineData(
        X=X,
        y=y,
        label_names=label_names,
        population_by_label_index=population_by_label_index,
        unassigned_id=unassigned[0] if unassigned else None,
    )


def parse_run_metadata(path: Path) -> Dict[str, Any]:
    """Extract common metadata from names such as modes_labeled_v3f2_R0.8_seed0.npy."""
    stem = path.stem
    if stem.startswith("modes_"):
        stem = stem[len("modes_"):]
    meta: Dict[str, Any] = {"run_name": stem, "modes_path": str(path)}
    patterns = {
        "checkpoint": r"(?:^|_)(all|labeled)(?:_|$)",
        "method": r"(?:^|_)(v2|v3f1|v3f2|v3f3|b0|b10|data_mean_shift)(?:_|$)",
        "r_value": r"(?:^|_)R(-?\d+(?:\.\d+)?)",
        "seed": r"(?:^|_)seed(\d+)",
        "training_seed": r"(?:^|_)(?:all|labeled)_s(\d+)(?:_|$)",
        "n_samples": r"(?:^|_)n(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, stem)
        if match:
            value: Any = match.group(1)
            if key in {"r_value"}:
                value = float(value)
            elif key in {"seed", "training_seed", "n_samples"}:
                value = int(value)
            meta[key] = value
    return meta


def _majority(values: np.ndarray) -> Tuple[int, int]:
    labels, counts = np.unique(values, return_counts=True)
    idx = int(np.argmax(counts))
    return int(labels[idx]), int(counts[idx])


def annotate_modes_from_neighbor_arrays(
    modes: np.ndarray,
    data: LevineData,
    distances: np.ndarray,
    indices: np.ndarray,
    k: int,
) -> pd.DataFrame:
    modes = np.atleast_2d(np.asarray(modes, dtype=np.float32))
    rows: List[Dict[str, Any]] = []
    for mode_id in range(len(modes)):
        neigh_labels = data.y[indices[mode_id, :k]]
        majority_all, majority_count = _majority(neigh_labels)
        purity_all = majority_count / k

        if data.unassigned_id is None:
            labeled_mask = np.ones(k, dtype=bool)
        else:
            labeled_mask = neigh_labels != data.unassigned_id
        labeled_count = int(labeled_mask.sum())
        frac_unassigned = 1.0 - labeled_count / k

        if labeled_count:
            majority_labeled, majority_labeled_count = _majority(neigh_labels[labeled_mask])
            purity_labeled = majority_labeled_count / labeled_count
            predicted_population = data.population_by_label_index[majority_labeled]
        else:
            majority_labeled = -1
            majority_labeled_count = 0
            purity_labeled = np.nan
            predicted_population = "unassigned"

        row: Dict[str, Any] = {
            "mode_id": mode_id,
            "k_neighbors": int(k),
            "nearest_label_all_index": majority_all,
            "nearest_label_all_name": data.population_by_label_index[majority_all],
            "purity_all": float(purity_all),
            "nearest_label_labeled_index": majority_labeled,
            "nearest_label_labeled_only_name": predicted_population,
            "purity_labeled_only": float(purity_labeled),
            "labeled_neighbor_count": labeled_count,
            "majority_labeled_neighbor_count": majority_labeled_count,
            "frac_unassigned_neighbors": float(frac_unassigned),
            "mean_knn_dist": float(np.mean(distances[mode_id, :k])),
            "median_knn_dist": float(np.median(distances[mode_id, :k])),
            "max_knn_dist": float(np.max(distances[mode_id, :k])),
        }
        for dim, value in enumerate(modes[mode_id]):
            row[f"x_{dim}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def add_confidence_flags(
    eval_df: pd.DataFrame,
    purity_threshold: float,
    max_unassigned_fraction: float,
    distance_threshold_global: Optional[float] = None,
    distance_threshold_by_population: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    out = eval_df.copy()
    out["passes_label_rule"] = (
        (out["purity_labeled_only"] >= purity_threshold)
        & (out["frac_unassigned_neighbors"] <= max_unassigned_fraction)
        & (out["nearest_label_labeled_only_name"] != "unassigned")
    )
    if distance_threshold_global is None:
        out["passes_global_distance"] = True
    else:
        out["passes_global_distance"] = out["mean_knn_dist"] <= float(distance_threshold_global)

    if distance_threshold_by_population is None:
        out["population_distance_threshold"] = np.nan
        out["passes_population_distance"] = True
    else:
        out["population_distance_threshold"] = out["nearest_label_labeled_only_name"].map(
            distance_threshold_by_population
        )
        # If a population has too few calibration cells, fall back to the global criterion.
        out["passes_population_distance"] = np.where(
            out["population_distance_threshold"].notna(),
            out["mean_knn_dist"] <= out["population_distance_threshold"],
            out["passes_global_distance"],
        )
    out["confident_without_distance"] = out["passes_label_rule"]
    out["confident_global_distance"] = out["passes_label_rule"] & out["passes_global_distance"]
    out["confident_population_distance"] = out["passes_label_rule"] & out["passes_population_distance"]
    return out


def population_counts(data: LevineData) -> pd.DataFrame:
    rows = []
    for label_index, raw_label in enumerate(data.label_names):
        pop = data.population_by_label_index[label_index]
        count = int(np.sum(data.y == label_index))
        rows.append({
            "label_index": label_index,
            "raw_label": str(raw_label),
            "population": pop,
            "cell_count": count,
        })
    df = pd.DataFrame(rows)
    labeled_total = int(df.loc[df["population"] != "unassigned", "cell_count"].sum())
    all_total = int(df["cell_count"].sum())
    df["percent_of_labeled_cells"] = np.where(
        df["population"] == "unassigned", np.nan, 100.0 * df["cell_count"] / max(labeled_total, 1)
    )
    df["percent_of_all_cells"] = 100.0 * df["cell_count"] / max(all_total, 1)
    return df


def summarize_annotation(
    eval_df: pd.DataFrame,
    data: LevineData,
    flag_column: str,
) -> Dict[str, Any]:
    selected = eval_df[eval_df[flag_column]].copy()
    covered = sorted(selected["nearest_label_labeled_only_name"].dropna().unique().tolist())
    counts = population_counts(data)
    covered_counts = counts[counts["population"].isin(covered)]
    return {
        "annotation_flag": flag_column,
        "raw_modes": int(len(eval_df)),
        "annotated_modes": int(len(selected)),
        "covered_populations": int(len(covered)),
        "covered_population_names": ";".join(covered),
        "weighted_population_coverage_labeled_pct": float(covered_counts["percent_of_labeled_cells"].sum()),
        "weighted_population_coverage_all_pct": float(covered_counts["percent_of_all_cells"].sum()),
        "mean_purity": float(selected["purity_labeled_only"].mean()) if len(selected) else np.nan,
        "mean_unassigned_fraction": float(selected["frac_unassigned_neighbors"].mean()) if len(selected) else np.nan,
        "mean_knn_distance": float(selected["mean_knn_dist"].mean()) if len(selected) else np.nan,
    }


def calibrate_real_cell_knn_distances(
    data: LevineData,
    k_values: Sequence[int],
    quantile: float = 0.95,
    global_sample_size: int = 20_000,
    per_population_sample_size: int = 2_000,
    seed: int = 0,
    n_jobs: int = -1,
) -> Tuple[pd.DataFrame, Dict[int, float], Dict[int, Dict[str, float]]]:
    """Estimate reference kNN-distance distributions using real cells.

    Distances are measured to the full data cloud. Query cells are drawn without
    replacement, and the zero self-neighbor is removed.
    """
    rng = np.random.default_rng(seed)
    max_k = max(int(k) for k in k_values)
    tree = cKDTree(data.X)

    rows: List[Dict[str, Any]] = []
    global_thresholds: Dict[int, float] = {}
    population_thresholds: Dict[int, Dict[str, float]] = {int(k): {} for k in k_values}

    def query_indices(sample_idx: np.ndarray, group: str, label_index: Optional[int]) -> None:
        dist, _ = tree.query(data.X[sample_idx], k=max_k + 1, workers=-1)
        dist = np.atleast_2d(dist)[:, 1:]  # exact self-neighbor
        for k in k_values:
            means = dist[:, : int(k)].mean(axis=1)
            threshold = float(np.quantile(means, quantile))
            rows.append({
                "group": group,
                "label_index": label_index,
                "k_neighbors": int(k),
                "n_reference_cells": int(len(sample_idx)),
                "mean": float(np.mean(means)),
                "median": float(np.median(means)),
                "q90": float(np.quantile(means, 0.90)),
                "q95": float(np.quantile(means, 0.95)),
                "q99": float(np.quantile(means, 0.99)),
                "selected_quantile": float(quantile),
                "selected_threshold": threshold,
            })
            if group == "__global__":
                global_thresholds[int(k)] = threshold
            else:
                population_thresholds[int(k)][group] = threshold

    global_n = min(global_sample_size, len(data.X))
    global_idx = rng.choice(len(data.X), size=global_n, replace=False)
    query_indices(global_idx, "__global__", None)

    for label_index in range(len(data.label_names)):
        population = data.population_by_label_index[label_index]
        if population == "unassigned":
            continue
        idx = np.flatnonzero(data.y == label_index)
        # A q95 estimate from only a handful of cells is too unstable. Keep at least 25 when possible.
        if len(idx) < 5:
            continue
        n = min(per_population_sample_size, len(idx))
        sample_idx = rng.choice(idx, size=n, replace=False)
        query_indices(sample_idx, population, label_index)

    return pd.DataFrame(rows), global_thresholds, population_thresholds


def theoretical_minimum_majority_count(
    k: int, purity_threshold: float, max_unassigned_fraction: float
) -> int:
    min_labeled = math.ceil(k * (1.0 - max_unassigned_fraction) - 1e-12)
    return math.ceil(min_labeled * purity_threshold - 1e-12)


def cell_level_coverage(
    cells: np.ndarray,
    modes: np.ndarray,
    radii: Sequence[float],
    batch_size: int = 20_000,
) -> Dict[float, float]:
    modes = np.atleast_2d(np.asarray(modes, dtype=np.float32))
    if len(modes) == 0:
        return {float(r): 0.0 for r in radii}
    tree = cKDTree(modes)
    nearest_parts = []
    for start in range(0, len(cells), batch_size):
        d, _ = tree.query(cells[start:start + batch_size], k=1, workers=-1)
        nearest_parts.append(np.asarray(d).reshape(-1))
    nearest = np.concatenate(nearest_parts)
    return {float(r): float(np.mean(nearest <= float(r)) * 100.0) for r in radii}


def population_detection_table(
    eval_df: pd.DataFrame,
    data: LevineData,
    flag_column: str,
) -> pd.DataFrame:
    counts = population_counts(data)
    selected = eval_df[eval_df[flag_column]].copy()
    if len(selected):
        found = selected.groupby("nearest_label_labeled_only_name").agg(
            found_modes=("mode_id", "count"),
            best_purity=("purity_labeled_only", "max"),
            min_unassigned_fraction=("frac_unassigned_neighbors", "min"),
            min_mean_knn_distance=("mean_knn_dist", "min"),
        ).reset_index().rename(columns={"nearest_label_labeled_only_name": "population"})
    else:
        found = pd.DataFrame(columns=[
            "population", "found_modes", "best_purity",
            "min_unassigned_fraction", "min_mean_knn_distance"
        ])
    out = counts[counts["population"] != "unassigned"].merge(found, on="population", how="left")
    out["found_modes"] = out["found_modes"].fillna(0).astype(int)
    out["found"] = out["found_modes"] > 0
    return out.sort_values(["found", "cell_count"], ascending=[False, False])
