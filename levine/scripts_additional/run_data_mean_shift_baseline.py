#!/usr/bin/env python3
"""Direct data-space adaptive kNN mean-shift baseline for Levine 13D.

The baseline never calls the diffusion model.  Starts are sampled from real
cells, each update replaces a point by the mean of its k nearest real cells,
and converged endpoints are merged by the same complete-linkage routine used by
the original diffusion baselines.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.algorithms.clustering import agglomerative_merge
from scripts_additional.levine13_common import load_levine_data, parse_csv_list


def adaptive_knn_mean_shift(
    reference: np.ndarray,
    starts: np.ndarray,
    k: int,
    max_iter: int,
    tol: float,
    damping: float,
) -> tuple[np.ndarray, int, float]:
    tree = cKDTree(reference)
    x = starts.astype(np.float32, copy=True)
    used = 0
    final_shift = np.inf
    for iteration in range(max_iter):
        _, idx = tree.query(x, k=min(k, len(reference)), workers=-1)
        idx = np.atleast_2d(idx)
        target = reference[idx].mean(axis=1)
        new_x = (1.0 - damping) * x + damping * target
        shifts = np.linalg.norm(new_x - x, axis=1)
        final_shift = float(shifts.max())
        x = new_x.astype(np.float32)
        used = iteration + 1
        if final_shift < tol:
            break
    return x, used, final_shift


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/levine13/levine13_processed.npz")
    p.add_argument("--populations", default="data/population_names_Levine_13dim.txt")
    p.add_argument("--subsets", default="labeled,all")
    p.add_argument("--n-starts", type=int, default=300)
    p.add_argument("--k-neighbors", type=int, default=200)
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--damping", type=float, default=1.0)
    p.add_argument("--r-values", default="0.8,1.5")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--out-dir", default="results_levine13/additional/data_mean_shift")
    args = p.parse_args()

    data = load_levine_data(Path(args.data), Path(args.populations))
    subsets = [x.strip() for x in args.subsets.split(",") if x.strip()]
    radii = parse_csv_list(args.r_values, float)
    seeds = parse_csv_list(args.seeds, int)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    endpoint_dir = out_dir / "endpoints"
    endpoint_dir.mkdir(exist_ok=True)
    rows = []

    for subset in subsets:
        if subset == "all":
            reference = data.X
        elif subset == "labeled":
            mask = np.ones(len(data.y), dtype=bool)
            if data.unassigned_id is not None:
                mask = data.y != data.unassigned_id
            reference = data.X[mask]
        else:
            raise ValueError(f"Unknown subset: {subset}")

        for seed in seeds:
            rng = np.random.default_rng(seed)
            start_idx = rng.choice(len(reference), size=min(args.n_starts, len(reference)), replace=False)
            starts = reference[start_idx]
            t0 = time.perf_counter()
            endpoints, iterations, final_shift = adaptive_knn_mean_shift(
                reference, starts, args.k_neighbors, args.max_iter, args.tol, args.damping
            )
            elapsed = time.perf_counter() - t0
            endpoint_path = endpoint_dir / f"endpoints_{subset}_data_mean_shift_k{args.k_neighbors}_n{len(starts)}_seed{seed}.npy"
            np.save(endpoint_path, endpoints)

            for radius in radii:
                modes = agglomerative_merge(endpoints, radius).astype(np.float32)
                run_name = f"{subset}_data_mean_shift_k{args.k_neighbors}_n{len(starts)}_R{radius:g}_seed{seed}"
                path = out_dir / f"modes_{run_name}.npy"
                np.save(path, modes)
                rows.append({
                    "run_name": run_name,
                    "checkpoint": subset,
                    "method": "data_mean_shift",
                    "r_value": radius,
                    "seed": seed,
                    "n_samples": len(starts),
                    "reference_cells": len(reference),
                    "k_mean_shift": args.k_neighbors,
                    "iterations": iterations,
                    "final_max_shift": final_shift,
                    "raw_modes": len(modes),
                    "nfe": 0,
                    "elapsed_sec": elapsed,
                    "modes_path": str(path),
                })
                print(f"{run_name}: modes={len(modes)}, iterations={iterations}, {elapsed:.2f}s")

    pd.DataFrame(rows).to_csv(out_dir / "run_summary.csv", index=False)
    print(f"Saved {out_dir}")


if __name__ == "__main__":
    main()
