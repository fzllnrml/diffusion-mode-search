#!/usr/bin/env python3
"""Evaluate saved Levine 13D candidates under several nearest-neighbor annotation settings."""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from levine13_common import (
    add_confidence_flags,
    annotate_modes_from_neighbor_arrays,
    calibrate_real_cell_knn_distances,
    cell_level_coverage,
    load_levine_data,
    parse_csv_list,
    parse_run_metadata,
    population_detection_table,
    summarize_annotation,
    theoretical_minimum_majority_count,
)


def collect_mode_paths(patterns: List[str]) -> List[Path]:
    found: Dict[str, Path] = {}
    for pattern in patterns:
        for item in glob.glob(pattern, recursive=True):
            path = Path(item)
            if path.is_file() and path.suffix == ".npy":
                found[str(path.resolve())] = path
    return sorted(found.values(), key=lambda p: str(p))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes-glob",
        action="append",
        required=True,
        help="Glob for modes_*.npy. May be supplied more than once.",
    )
    parser.add_argument("--data-path", default="data/levine13/levine13_processed.npz")
    parser.add_argument("--pop-path", default="data/population_names_Levine_13dim.txt")
    parser.add_argument("--k-values", default="25,50,100,200")
    parser.add_argument("--purity-threshold", type=float, default=0.90)
    parser.add_argument("--max-unassigned-fraction", type=float, default=0.20)
    parser.add_argument("--distance-quantile", type=float, default=0.95)
    parser.add_argument("--global-calibration-sample", type=int, default=20000)
    parser.add_argument("--per-population-calibration-sample", type=int, default=2000)
    parser.add_argument("--coverage-radii", default="0.25,0.5,0.75,1.0,1.5")
    parser.add_argument("--cell-coverage-k-values", default="100", help="Compute expensive cell-level coverage only for these k values.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out-dir", default="results_levine13/additional/annotation_sensitivity")
    args = parser.parse_args()

    k_values = parse_csv_list(args.k_values, int)
    radii = parse_csv_list(args.coverage_radii, float)
    cell_coverage_k_values = set(parse_csv_list(args.cell_coverage_k_values, int))
    if not k_values or min(k_values) <= 0:
        raise ValueError("k-values must contain positive integers")

    out_dir = Path(args.out_dir)
    detail_dir = out_dir / "candidate_details"
    population_dir = out_dir / "population_tables"
    detail_dir.mkdir(parents=True, exist_ok=True)
    population_dir.mkdir(parents=True, exist_ok=True)

    data = load_levine_data(Path(args.data_path), Path(args.pop_path))
    mode_paths = collect_mode_paths(args.modes_glob)
    if not mode_paths:
        raise FileNotFoundError(f"No modes files matched: {args.modes_glob}")

    print(f"Loaded Levine data: X={data.X.shape}; mode files={len(mode_paths)}")
    print("Calibrating real-cell kNN distances...")
    calibration_df, global_thresholds, population_thresholds = calibrate_real_cell_knn_distances(
        data=data,
        k_values=k_values,
        quantile=args.distance_quantile,
        global_sample_size=args.global_calibration_sample,
        per_population_sample_size=args.per_population_calibration_sample,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    calibration_df.to_csv(out_dir / "distance_calibration.csv", index=False)

    tree = cKDTree(data.X)
    labeled_mask = np.ones(len(data.X), dtype=bool)
    if data.unassigned_id is not None:
        labeled_mask = data.y != data.unassigned_id
    labeled_cells = data.X[labeled_mask]

    summary_rows = []
    cell_rows = []
    population_rows = []

    for path in mode_paths:
        metadata = parse_run_metadata(path)
        modes = np.atleast_2d(np.asarray(np.load(path), dtype=np.float32))
        if modes.shape[1] != data.X.shape[1]:
            raise ValueError(f"Dimension mismatch for {path}: modes={modes.shape}, X={data.X.shape}")
        print(f"Evaluating {path.name}: {len(modes)} candidates")
        distances, indices = tree.query(modes, k=max(k_values), workers=-1)
        distances = np.atleast_2d(distances)
        indices = np.atleast_2d(indices)

        for k in k_values:
            eval_df = annotate_modes_from_neighbor_arrays(modes, data, distances, indices, k)
            eval_df = add_confidence_flags(
                eval_df,
                purity_threshold=args.purity_threshold,
                max_unassigned_fraction=args.max_unassigned_fraction,
                distance_threshold_global=global_thresholds[k],
                distance_threshold_by_population=population_thresholds[k],
            )
            for key, value in metadata.items():
                eval_df[key] = value
            detail_name = f"{metadata['run_name']}_k{k}.csv"
            eval_df.to_csv(detail_dir / detail_name, index=False)

            theoretical_min = theoretical_minimum_majority_count(
                k, args.purity_threshold, args.max_unassigned_fraction
            )
            for flag in [
                "confident_without_distance",
                "confident_global_distance",
                "confident_population_distance",
            ]:
                row = summarize_annotation(eval_df, data, flag)
                row.update(metadata)
                row.update({
                    "k_neighbors": k,
                    "purity_threshold": args.purity_threshold,
                    "max_unassigned_fraction": args.max_unassigned_fraction,
                    "theoretical_minimum_majority_neighbors": theoretical_min,
                    "global_distance_threshold": global_thresholds[k],
                    "distance_quantile": args.distance_quantile,
                })
                summary_rows.append(row)

                pop_df = population_detection_table(eval_df, data, flag)
                pop_df["annotation_flag"] = flag
                pop_df["k_neighbors"] = k
                for key, value in metadata.items():
                    pop_df[key] = value
                population_rows.append(pop_df)
                pop_df.to_csv(
                    population_dir / f"{metadata['run_name']}_k{k}_{flag}.csv",
                    index=False,
                )

                if k in cell_coverage_k_values:
                    selected_modes = modes[eval_df[flag].to_numpy(dtype=bool)]
                    for population_scope, cells in [("labeled_cells", labeled_cells), ("all_cells", data.X)]:
                        coverage = cell_level_coverage(cells, selected_modes, radii)
                        cell_row = {
                            **metadata,
                            "k_neighbors": k,
                            "annotation_flag": flag,
                            "population_scope": population_scope,
                            "selected_modes": int(len(selected_modes)),
                        }
                        for radius, pct in coverage.items():
                            cell_row[f"cell_coverage_pct_r{radius:g}"] = pct
                        cell_rows.append(cell_row)

            # The raw-candidate coverage is independent of k; keep one copy at the smallest k.
            if cell_coverage_k_values and k == min(k_values):
                for population_scope, cells in [("labeled_cells", labeled_cells), ("all_cells", data.X)]:
                    coverage = cell_level_coverage(cells, modes, radii)
                    cell_row = {
                        **metadata,
                        "k_neighbors": np.nan,
                        "annotation_flag": "all_raw_modes",
                        "population_scope": population_scope,
                        "selected_modes": int(len(modes)),
                    }
                    for radius, pct in coverage.items():
                        cell_row[f"cell_coverage_pct_r{radius:g}"] = pct
                    cell_rows.append(cell_row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "annotation_sensitivity_summary.csv", index=False)
    cell_summary = pd.DataFrame(cell_rows)
    cell_summary.to_csv(out_dir / "cell_level_coverage_summary.csv", index=False)
    if population_rows:
        populations = pd.concat(population_rows, ignore_index=True)
        populations.to_csv(out_dir / "population_detection_all_runs.csv", index=False)

        # Frequency across seeds for each exact configuration.
        grouping = [
            c for c in ["checkpoint", "training_seed", "method", "r_value", "n_samples", "k_neighbors", "annotation_flag", "population"]
            if c in populations.columns
        ]
        frequency = populations.groupby(grouping, dropna=False).agg(
            seeds_or_runs=("run_name", "nunique"),
            times_found=("found", "sum"),
            detection_frequency=("found", "mean"),
            cell_count=("cell_count", "first"),
            percent_of_labeled_cells=("percent_of_labeled_cells", "first"),
        ).reset_index()
        frequency.to_csv(out_dir / "population_detection_frequency.csv", index=False)

    print(f"Saved: {out_dir / 'annotation_sensitivity_summary.csv'}")
    print(f"Saved: {out_dir / 'cell_level_coverage_summary.csv'}")
    print(f"Saved: {out_dir / 'distance_calibration.csv'}")


if __name__ == "__main__":
    main()
