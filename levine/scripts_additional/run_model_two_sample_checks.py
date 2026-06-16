#!/usr/bin/env python3
"""Quantitative real-vs-generated checks for Levine 13D checkpoints."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.algorithms.mode_finder_v2 import ddim_sample
from src.models.diffusion import DiffusionModel
from src.utils.device import resolve_device
from scripts_additional.levine13_common import load_levine_data


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_proj: int, rng: np.random.Generator) -> tuple[float, float]:
    d = x.shape[1]
    directions = rng.normal(size=(n_proj, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    px = np.sort(x @ directions.T, axis=0)
    py = np.sort(y @ directions.T, axis=0)
    vals = np.mean(np.abs(px - py), axis=0)
    return float(vals.mean()), float(np.quantile(vals, 0.95))


def classifier_auc(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    z = np.vstack([x, y])
    target = np.concatenate([np.zeros(len(x), dtype=int), np.ones(len(y), dtype=int)])
    xtr, xte, ytr, yte = train_test_split(
        z, target, test_size=0.30, random_state=seed, stratify=target
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    clf.fit(xtr, ytr)
    prob = clf.predict_proba(xte)[:, 1]
    auc = roc_auc_score(yte, prob)
    return float(max(auc, 1.0 - auc))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", default="all=checkpoints/levine13_all_100k.pt,labeled=checkpoints/levine13_labeled_100k.pt")
    p.add_argument("--data", default="data/levine13/levine13_processed.npz")
    p.add_argument("--populations", default="data/population_names_Levine_13dim.txt")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--projections", type=int, default=100)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--out-dir", default="results_levine13/additional/model_two_sample")
    args = p.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    data = load_levine_data(Path(args.data), Path(args.populations))
    checkpoints = {}
    for item in args.checkpoints.split(","):
        tag, path = item.split("=", 1)
        checkpoints[tag.strip()] = Path(path.strip())
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    rows = []

    for tag, checkpoint in checkpoints.items():
        if tag == "all":
            reference = data.X
        elif tag == "labeled":
            mask = np.ones(len(data.y), dtype=bool)
            if data.unassigned_id is not None:
                mask = data.y != data.unassigned_id
            reference = data.X[mask]
        else:
            raise ValueError(f"Checkpoint tag must be all or labeled, got {tag}")
        device = resolve_device(args.device)
        model = DiffusionModel.from_checkpoint(str(checkpoint), device=device)

        for seed in seeds:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(reference), size=min(args.n, len(reference)), replace=False)
            real = reference[idx].astype(np.float32)
            t0 = time.perf_counter()
            generated = np.asarray(
                ddim_sample(model, n=len(real), num_steps=args.ddim_steps, seed=seed),
                dtype=np.float32,
            )
            elapsed = time.perf_counter() - t0
            np.save(sample_dir / f"generated_{tag}_seed{seed}.npy", generated)

            sw_mean, sw_q95 = sliced_wasserstein(real, generated, args.projections, rng)
            mean_delta = float(np.linalg.norm(real.mean(axis=0) - generated.mean(axis=0)))
            cov_real = np.cov(real, rowvar=False)
            cov_gen = np.cov(generated, rowvar=False)
            cov_rel = float(np.linalg.norm(cov_real - cov_gen, ord="fro") / max(np.linalg.norm(cov_real, ord="fro"), 1e-12))
            marginal_w1 = np.mean(
                [np.mean(np.abs(np.sort(real[:, j]) - np.sort(generated[:, j]))) for j in range(real.shape[1])]
            )
            auc = classifier_auc(real, generated, seed)
            rows.append({
                "checkpoint": tag,
                "checkpoint_path": str(checkpoint),
                "seed": seed,
                "n_real": len(real),
                "n_generated": len(generated),
                "ddim_steps": args.ddim_steps,
                "sliced_wasserstein_mean": sw_mean,
                "sliced_wasserstein_q95": sw_q95,
                "marginal_wasserstein_mean": float(marginal_w1),
                "mean_vector_l2_error": mean_delta,
                "covariance_relative_frobenius_error": cov_rel,
                "linear_classifier_auc": auc,
                "generation_elapsed_sec": elapsed,
            })
            print(f"{tag} seed={seed}: SW={sw_mean:.4f}, AUC={auc:.4f}, {elapsed:.2f}s")

    per_run = pd.DataFrame(rows)
    per_run.to_csv(out_dir / "two_sample_metrics_per_run.csv", index=False)
    metric_cols = [
        "sliced_wasserstein_mean", "sliced_wasserstein_q95", "marginal_wasserstein_mean",
        "mean_vector_l2_error", "covariance_relative_frobenius_error", "linear_classifier_auc",
        "generation_elapsed_sec",
    ]
    agg = per_run.groupby("checkpoint")[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([str(x) for x in col if str(x)]) for col in agg.columns.to_flat_index()]
    agg.to_csv(out_dir / "two_sample_metrics_summary.csv", index=False)
    print(agg.to_string(index=False))
    print(f"Saved {out_dir}")


if __name__ == "__main__":
    main()
