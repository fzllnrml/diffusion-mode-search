#!/usr/bin/env python3
"""Geometric cell-level coverage for saved Levine 13D candidate arrays.

Unlike weighted population coverage, this script checks every real cell.  It
reports (1) distance to any retained candidate and (2) label-consistent distance
to a retained candidate annotated as the same reference population.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scripts_additional.levine13_common import (
    add_confidence_flags,
    annotate_modes_from_neighbor_arrays,
    load_levine_data,
    parse_csv_list,
    parse_run_metadata,
)


def expand(patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern)]
        if matches:
            out.extend(matches)
        elif Path(pattern).exists():
            out.append(Path(pattern))
    return sorted(set(out))


def nearest_distances(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if len(centers) == 0:
        return np.full(len(points), np.inf, dtype=np.float32)
    tree = cKDTree(np.asarray(centers, dtype=np.float32))
    dist, _ = tree.query(points, k=1, workers=-1)
    return np.asarray(dist, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--modes", nargs="+", required=True, help="Paths or glob patterns")
    p.add_argument("--data", default="data/levine13/levine13_processed.npz")
    p.add_argument("--populations", default="data/population_names_Levine_13dim.txt")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--purity", type=float, default=0.90)
    p.add_argument("--max-unassigned", type=float, default=0.20)
    p.add_argument(
        "--flags",
        default="all_raw,confident_without_distance",
        help="all_raw plus confidence columns from annotation",
    )
    p.add_argument("--radii", default="0.25,0.5,0.8,1.0,1.5,2.0")
    p.add_argument("--out-dir", default="results_levine13/additional/cell_level_coverage")
    args = p.parse_args()

    paths = expand(args.modes)
    if not paths:
        raise FileNotFoundError("No mode files matched")
    data = load_levine_data(Path(args.data), Path(args.populations))
    tree = cKDTree(data.X)
    radii = parse_csv_list(args.radii, float)
    flags = [x.strip() for x in args.flags.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    population_rows = []
    labeled_mask = np.ones(len(data.y), dtype=bool)
    if data.unassigned_id is not None:
        labeled_mask = data.y != data.unassigned_id

    for path in paths:
        modes = np.atleast_2d(np.asarray(np.load(path), dtype=np.float32))
        dist, idx = tree.query(modes, k=args.k, workers=-1)
        eval_df = annotate_modes_from_neighbor_arrays(modes, data, dist, idx, args.k)
        eval_df = add_confidence_flags(eval_df, args.purity, args.max_unassigned)
        meta = parse_run_metadata(path)

        for flag in flags:
            if flag == "all_raw":
                selected = eval_df.copy()
            elif flag in eval_df.columns:
                selected = eval_df[eval_df[flag]].copy()
            else:
                raise ValueError(f"Unknown flag: {flag}")

            mode_ids = selected["mode_id"].to_numpy(dtype=int)
            selected_modes = modes[mode_ids] if len(mode_ids) else np.empty((0, modes.shape[1]))
            d_any = nearest_distances(data.X, selected_modes)

            # Label-consistent distance: each labeled cell must be close to a
            # retained candidate carrying exactly its own reference label.
            d_same = np.full(len(data.X), np.inf, dtype=np.float32)
            for label_index, population in data.population_by_label_index.items():
                if population == "unassigned":
                    continue
                cell_idx = np.flatnonzero(data.y == label_index)
                candidate_rows = selected[
                    selected["nearest_label_labeled_index"] == label_index
                ]
                if len(candidate_rows):
                    centers = modes[candidate_rows["mode_id"].to_numpy(dtype=int)]
                    d_same[cell_idx] = nearest_distances(data.X[cell_idx], centers)

            for radius in radii:
                row = dict(meta)
                row.update({
                    "selection_flag": flag,
                    "k_neighbors": args.k,
                    "purity_threshold": args.purity,
                    "max_unassigned_fraction": args.max_unassigned,
                    "retained_candidates": int(len(selected_modes)),
                    "radius": radius,
                    "all_cells_any_candidate_pct": float(np.mean(d_any <= radius) * 100),
                    "labeled_cells_any_candidate_pct": float(np.mean(d_any[labeled_mask] <= radius) * 100),
                    "labeled_cells_label_consistent_pct": float(np.mean(d_same[labeled_mask] <= radius) * 100),
                })
                summary_rows.append(row)

                for label_index, population in data.population_by_label_index.items():
                    if population == "unassigned":
                        continue
                    mask = data.y == label_index
                    population_rows.append({
                        **meta,
                        "selection_flag": flag,
                        "k_neighbors": args.k,
                        "radius": radius,
                        "label_index": label_index,
                        "population": population,
                        "cell_count": int(mask.sum()),
                        "covered_any_candidate_pct": float(np.mean(d_any[mask] <= radius) * 100),
                        "covered_label_consistent_pct": float(np.mean(d_same[mask] <= radius) * 100),
                    })

    summary = pd.DataFrame(summary_rows)
    per_population = pd.DataFrame(population_rows)
    summary.to_csv(out_dir / "cell_level_coverage_summary.csv", index=False)
    per_population.to_csv(out_dir / "cell_level_coverage_by_population.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved {out_dir}")


if __name__ == "__main__":
    main()
