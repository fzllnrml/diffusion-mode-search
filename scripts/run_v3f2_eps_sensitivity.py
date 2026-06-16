from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import load_config, setup_logging
from src.utils.device import resolve_device
from src.utils.distribution import GaussianMixture
from src.utils.metrics import evaluate_modes
from src.algorithms.mode_finder_v3_f2 import ModeFinderV3F2
from src.algorithms.baseline import BaselineModeFinder

try:
    from scripts.run_all_experiments import _train_or_load
except ModuleNotFoundError:
    from run_all_experiments import _train_or_load


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run_v3f2(model, cfg, seed: int):
    mf = cfg.mode_finder_v3_f2
    finder = ModeFinderV3F2(
        model=model,
        timesteps=mf.timesteps,
        step_size=mf.step_size,
        n_particles=mf.n_particles,
        merge_factor=mf.merge_factor,
        merge_radius_min=mf.merge_radius_min,
        ascent_steps=mf.ascent_steps,
        normalize_score=mf.normalize_score,
        refine_steps=mf.refine_steps,
        refine_step_scale=mf.refine_step_scale,
        ddim_steps=mf.ddim_steps,
        init_stop_t=mf.init_stop_t,
        x_min=cfg.distribution.x_min,
        x_max=cfg.distribution.x_max,
    )
    return finder.find_modes(seed=seed, verbose=False)


def run_baseline(model, cfg, seed: int, refine_steps: int, n_samples: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    bl = cfg.baseline
    finder = BaselineModeFinder(
        model=model,
        n_samples=n_samples,
        refine_steps=refine_steps,
        refine_alpha=bl.refine_alpha,
        merge_radius=bl.merge_radius,
    )
    return finder.find_modes()


def add_metric_rows(rows, method, K, dseed, init_seed, result, true_modes, eps_values):
    for eps in eps_values:
        m = evaluate_modes(true_modes, result.modes, eps_hit=eps)
        rows.append({
            "method": method,
            "K": K,
            "dseed": dseed,
            "init_seed": init_seed,
            "eps": eps,
            "recall": m.recall,
            "precision": m.precision,
            "f1": m.f1,
            "soft_error": m.soft_error,
            "n_found": m.n_found,
            "n_hits": m.n_hits,
            "n_correct": m.n_correct,
            "nfe": result.nfe,
        })


def summarize(df: pd.DataFrame, group_cols, out_path: Path):
    metrics = ["recall", "precision", "f1", "soft_error", "n_found", "nfe"]
    summary = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join([str(x) for x in col if x != ""])
        for col in summary.columns.to_flat_index()
    ]
    summary = summary.fillna(0.0)

    method_order = {"v3f2": 0, "b0": 1, "b10": 2}
    summary["_method_order"] = summary["method"].map(method_order).fillna(99)

    sort_cols = []
    if "K" in group_cols:
        sort_cols.append("K")
    sort_cols += ["eps", "_method_order"]

    summary = summary.sort_values(sort_cols).drop(columns=["_method_order"])
    summary.to_csv(out_path, index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eps", type=str, default="1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--k-values", type=str, default="2,3,4,5,6,7")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-init-seeds", type=int, default=3)
    parser.add_argument("--baseline-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging)

    cfg.checkpoint.checkpoint_dir = args.checkpoint_dir

    eps_values = parse_float_list(args.eps)
    k_values = parse_int_list(args.k_values)
    n_samples = args.baseline_samples or cfg.baseline.n_samples

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.device)

    print("=" * 80)
    print("v3f2 eps sensitivity")
    print("=" * 80)
    print(f"config:           {args.config}")
    print(f"checkpoint_dir:   {args.checkpoint_dir}")
    print(f"output_dir:       {out_dir}")
    print(f"K values:         {k_values}")
    print(f"eps values:       {eps_values}")
    print(f"n_seeds:          {args.n_seeds}")
    print(f"n_init_seeds:     {args.n_init_seeds}")
    print(f"baseline_samples: {n_samples}")
    print(f"dim:              {cfg.dim}")
    print()

    rows = []
    t_all = time.perf_counter()

    for K in k_values:
        print(f"\n{'=' * 80}\nK={K}\n{'=' * 80}")

        for dseed in range(args.n_seeds):
            mixture = GaussianMixture.random(
                K=K,
                dim=cfg.dim,
                sigma_range=(0.5, 1.2),
                min_sep=2.0,
                bounds=(cfg.distribution.x_min, cfg.distribution.x_max),
                seed=dseed,
            )

            true_modes = mixture.mode_locations
            tag = f"dim{cfg.dim}_K{K}_dseed{dseed}"

            print(f"\n--- K={K}, dseed={dseed}, tag={tag} ---")
            model = _train_or_load(
                mixture=mixture,
                cfg=cfg,
                device=device,
                checkpoint_dir=args.checkpoint_dir,
                tag=tag,
            )

            for init_seed in range(args.n_init_seeds):
                print(f"  init_seed={init_seed}: running v3f2...")
                r_v3f2 = run_v3f2(model, cfg, seed=init_seed)
                add_metric_rows(rows, "v3f2", K, dseed, init_seed, r_v3f2, true_modes, eps_values)

                print(f"  init_seed={init_seed}: running b0...")
                r_b0 = run_baseline(
                    model, cfg,
                    seed=init_seed,
                    refine_steps=0,
                    n_samples=n_samples,
                )
                add_metric_rows(rows, "b0", K, dseed, init_seed, r_b0, true_modes, eps_values)

                print(f"  init_seed={init_seed}: running b10...")
                r_b10 = run_baseline(
                    model, cfg,
                    seed=init_seed + 1000,
                    refine_steps=10,
                    n_samples=n_samples,
                )
                add_metric_rows(rows, "b10", K, dseed, init_seed, r_b10, true_modes, eps_values)

                raw_df = pd.DataFrame(rows)
                raw_df.to_csv(out_dir / "eps_sensitivity_raw.csv", index=False)

    df = pd.DataFrame(rows)
    raw_path = out_dir / "eps_sensitivity_raw.csv"
    by_k_path = out_dir / "eps_sensitivity_by_k.csv"
    overall_path = out_dir / "eps_sensitivity_overall.csv"

    df.to_csv(raw_path, index=False)

    by_k = summarize(df, ["K", "eps", "method"], by_k_path)
    overall = summarize(df, ["eps", "method"], overall_path)

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    cols = [
        "eps", "method",
        "recall_mean", "precision_mean", "f1_mean",
        "soft_error_mean", "n_found_mean", "nfe_mean",
    ]
    print(overall[cols].round(4).to_string(index=False))

    print("\nSaved:")
    print(f"  raw:     {raw_path}")
    print(f"  by K:    {by_k_path}")
    print(f"  overall: {overall_path}")
    print(f"\nDone in {(time.perf_counter() - t_all):.1f} sec")


if __name__ == "__main__":
    main()
