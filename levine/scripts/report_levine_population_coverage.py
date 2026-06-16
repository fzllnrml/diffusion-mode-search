#!/usr/bin/env python3
"""Compute population coverage from saved Levine 13D candidate evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = Path("data/levine13/levine13_processed.npz")
DEFAULT_POP_PATH = Path("data/population_names_Levine_13dim.txt")


def load_population_map(pop_path: Path) -> Dict[int, str]:
    if not pop_path.exists():
        raise FileNotFoundError(f"Population names file not found: {pop_path}")

    df = pd.read_csv(pop_path, sep=r"\s+", engine="python")
    cols = [c.lower() for c in df.columns]
    if "label" not in cols or "population" not in cols:
        raise ValueError(f"Unexpected population file columns: {df.columns.tolist()}")

    label_col = df.columns[cols.index("label")]
    pop_col = df.columns[cols.index("population")]

    return {int(row[label_col]): str(row[pop_col]) for _, row in df.iterrows()}


def raw_label_to_population(raw: Any, pop_map: Dict[int, str]) -> str:
    s = str(raw).strip()
    if s.lower() == "unassigned":
        return "unassigned"
    try:
        label_id = int(float(s))
        return pop_map.get(label_id, f"unknown_label_{label_id}")
    except Exception:
        return f"bad_label_{s}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--pop-path", default=str(DEFAULT_POP_PATH))
    parser.add_argument("--eval-csv", required=True, help="Mode evaluation CSV, e.g. eval_labeled_v3f2_R0.8_seed0.csv")
    parser.add_argument("--out", default="results_levine13/population_coverage_report.csv")
    parser.add_argument("--purity-threshold", type=float, default=0.90)
    parser.add_argument("--max-unassigned-fraction", type=float, default=0.20)
    parser.add_argument("--sort-by", choices=["count", "population", "found_then_count"], default="count")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    pop_path = Path(args.pop_path)
    eval_path = Path(args.eval_csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Processed Levine data not found: {data_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval CSV not found: {eval_path}")

    d = np.load(data_path, allow_pickle=True)
    y = d["y"].astype(int)
    label_names = np.array([str(x) for x in d["label_names"]], dtype=object)

    pop_map = load_population_map(pop_path)

    label_records = []
    total_cells = int(len(y))

    for idx, raw_label in enumerate(label_names):
        population = raw_label_to_population(raw_label, pop_map)
        count = int((y == idx).sum())
        label_records.append(
            {
                "label_index": idx,
                "raw_label": raw_label,
                "population": population,
                "count_all_cells": count,
            }
        )

    count_df = pd.DataFrame(label_records)

    labeled_df = count_df[count_df["population"] != "unassigned"].copy()
    total_labeled = int(labeled_df["count_all_cells"].sum())

    count_df["percent_of_all_cells"] = 100.0 * count_df["count_all_cells"] / max(total_cells, 1)
    count_df["percent_of_labeled_cells"] = np.where(
        count_df["population"] != "unassigned",
        100.0 * count_df["count_all_cells"] / max(total_labeled, 1),
        np.nan,
    )

    eval_df = pd.read_csv(eval_path)

    required_cols = [
        "nearest_label_labeled_only_name",
        "purity_labeled_only",
        "frac_unassigned_neighbors",
        "mean_knn_dist",
    ]
    for col in required_cols:
        if col not in eval_df.columns:
            raise ValueError(f"Eval CSV missing required column '{col}'. Columns: {eval_df.columns.tolist()}")

    confident = eval_df[
        (eval_df["purity_labeled_only"] >= args.purity_threshold)
        & (eval_df["frac_unassigned_neighbors"] <= args.max_unassigned_fraction)
        & (eval_df["nearest_label_labeled_only_name"] != "unassigned")
    ].copy()

    if len(confident) > 0:
        agg = confident.groupby("nearest_label_labeled_only_name").agg(
            found_modes=("mode_id", "count"),
            best_purity=("purity_labeled_only", "max"),
            mean_purity=("purity_labeled_only", "mean"),
            min_frac_unassigned=("frac_unassigned_neighbors", "min"),
            mean_frac_unassigned=("frac_unassigned_neighbors", "mean"),
            best_mean_knn_dist=("mean_knn_dist", "min"),
            mean_knn_dist=("mean_knn_dist", "mean"),
        ).reset_index().rename(columns={"nearest_label_labeled_only_name": "population"})
    else:
        agg = pd.DataFrame(columns=[
            "population", "found_modes", "best_purity", "mean_purity",
            "min_frac_unassigned", "mean_frac_unassigned",
            "best_mean_knn_dist", "mean_knn_dist"
        ])

    report = count_df.merge(agg, on="population", how="left")
    report["found"] = report["found_modes"].fillna(0).astype(int) > 0
    report["found_modes"] = report["found_modes"].fillna(0).astype(int)

    # Put labeled populations first, unassigned last.
    report["is_unassigned"] = report["population"].eq("unassigned")

    if args.sort_by == "count":
        report = report.sort_values(["is_unassigned", "count_all_cells"], ascending=[True, False])
    elif args.sort_by == "population":
        report = report.sort_values(["is_unassigned", "population"], ascending=[True, True])
    else:
        report = report.sort_values(["is_unassigned", "found", "count_all_cells"], ascending=[True, False, False])

    report = report.drop(columns=["is_unassigned"])

    report.to_csv(out_path, index=False)

    found_labeled = report[(report["population"] != "unassigned") & (report["found"])]
    all_labeled = report[report["population"] != "unassigned"]

    print("\n=== Levine13 population coverage report ===")
    print(f"eval_csv: {eval_path}")
    print(f"total cells: {total_cells}")
    print(f"total labeled cells: {total_labeled}")
    print(f"found labeled populations: {len(found_labeled)} / {len(all_labeled)}")
    print(f"confident rule: purity_labeled_only >= {args.purity_threshold}, "
          f"frac_unassigned <= {args.max_unassigned_fraction}")
    print(f"saved: {out_path}\n")

    show_cols = [
        "raw_label",
        "population",
        "count_all_cells",
        "percent_of_labeled_cells",
        "percent_of_all_cells",
        "found",
        "found_modes",
        "best_purity",
        "min_frac_unassigned",
        "best_mean_knn_dist",
    ]

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 220)
    print(report[show_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Not found labeled populations ===")
    not_found = all_labeled[~all_labeled["found"]].copy()
    if len(not_found) == 0:
        print("All labeled populations were found by this rule.")
    else:
        print(not_found[[
            "raw_label", "population", "count_all_cells",
            "percent_of_labeled_cells", "percent_of_all_cells"
        ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
