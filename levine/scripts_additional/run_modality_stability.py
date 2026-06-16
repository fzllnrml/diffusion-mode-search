#!/usr/bin/env python3
"""Evaluate local score-field stability of saved Levine 13D candidates."""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from src.models.diffusion import DiffusionModel
from src.utils.device import resolve_device
from scripts_additional.levine13_common import parse_csv_list, parse_run_metadata


def expand(patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern)]
        out.extend(matches if matches else ([Path(pattern)] if Path(pattern).exists() else []))
    return sorted(set(out))


def score_batch(model: DiffusionModel, x: np.ndarray, t: int = 0) -> np.ndarray:
    tx = torch.as_tensor(x, device=model.device, dtype=torch.float32)
    with torch.no_grad():
        s = model.score(tx, t=t)
    return s.detach().cpu().numpy().astype(np.float32)


def ascent(
    model: DiffusionModel,
    starts: np.ndarray,
    steps: int,
    alpha: float,
    x_min: float,
    x_max: float,
) -> np.ndarray:
    x = torch.as_tensor(starts, device=model.device, dtype=torch.float32).clone()
    with torch.no_grad():
        for _ in range(steps):
            x = torch.clamp(x + alpha * model.score(x, t=0), x_min, x_max)
    return x.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--modes", nargs="+", required=True)
    p.add_argument("--checkpoint", default="checkpoints/levine13_labeled_100k.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--ascent-steps", type=int, default=100)
    p.add_argument("--ascent-alpha", type=float, default=0.001)
    p.add_argument("--perturbations", type=int, default=8)
    p.add_argument("--perturb-sigma", type=float, default=0.05)
    p.add_argument("--attraction-radii", default="0.05,0.1,0.2")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--x-min", type=float, default=-10.0)
    p.add_argument("--x-max", type=float, default=10.0)
    p.add_argument("--out-dir", default="results_levine13/additional/modality_stability")
    args = p.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    paths = expand(args.modes)
    if not paths:
        raise FileNotFoundError("No mode files matched")
    device = resolve_device(args.device)
    model = DiffusionModel.from_checkpoint(args.checkpoint, device=device)
    radii = parse_csv_list(args.attraction_radii, float)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_frames = []
    summary_rows = []
    for path in paths:
        modes = np.atleast_2d(np.asarray(np.load(path), dtype=np.float32))
        started = time.perf_counter()
        initial_score = score_batch(model, modes)
        refined = ascent(model, modes, args.ascent_steps, args.ascent_alpha, args.x_min, args.x_max)
        final_score = score_batch(model, refined)

        repeated = np.repeat(modes, args.perturbations, axis=0)
        noise = rng.normal(0.0, args.perturb_sigma, size=repeated.shape).astype(np.float32)
        perturbed = np.clip(repeated + noise, args.x_min, args.x_max)
        endpoints = ascent(model, perturbed, args.ascent_steps, args.ascent_alpha, args.x_min, args.x_max)
        reference = np.repeat(refined, args.perturbations, axis=0)
        endpoint_distance = np.linalg.norm(endpoints - reference, axis=1).reshape(len(modes), args.perturbations)
        elapsed = time.perf_counter() - started

        df = pd.DataFrame({
            "mode_id": np.arange(len(modes)),
            "score_norm_before": np.linalg.norm(initial_score, axis=1),
            "score_norm_after": np.linalg.norm(final_score, axis=1),
            "extra_ascent_displacement": np.linalg.norm(refined - modes, axis=1),
            "perturb_endpoint_distance_mean": endpoint_distance.mean(axis=1),
            "perturb_endpoint_distance_max": endpoint_distance.max(axis=1),
        })
        for radius in radii:
            df[f"attraction_fraction_r{radius:g}"] = np.mean(endpoint_distance <= radius, axis=1)
        for j in range(modes.shape[1]):
            df[f"x_{j}"] = modes[:, j]
            df[f"refined_x_{j}"] = refined[:, j]
        meta = parse_run_metadata(path)
        for key, value in meta.items():
            df[key] = value
        detail_frames.append(df)

        row = {
            **meta,
            "raw_modes": len(modes),
            "median_score_norm_before": float(df["score_norm_before"].median()),
            "median_score_norm_after": float(df["score_norm_after"].median()),
            "median_extra_ascent_displacement": float(df["extra_ascent_displacement"].median()),
            "q95_extra_ascent_displacement": float(df["extra_ascent_displacement"].quantile(0.95)),
            "mean_perturb_endpoint_distance": float(df["perturb_endpoint_distance_mean"].mean()),
            "elapsed_sec": elapsed,
            "ascent_steps": args.ascent_steps,
            "ascent_alpha": args.ascent_alpha,
            "perturbations": args.perturbations,
            "perturb_sigma": args.perturb_sigma,
        }
        for radius in radii:
            col = f"attraction_fraction_r{radius:g}"
            row[f"mean_{col}"] = float(df[col].mean())
            row[f"fully_stable_candidates_r{radius:g}_pct"] = float(np.mean(df[col] == 1.0) * 100)
        summary_rows.append(row)
        df.to_csv(out_dir / f"candidate_stability_{path.stem}.csv", index=False)
        print(f"{path.name}: n={len(modes)}, elapsed={elapsed:.2f}s")

    pd.concat(detail_frames, ignore_index=True).to_csv(out_dir / "candidate_stability_all.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "modality_stability_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved {out_dir}")


if __name__ == "__main__":
    main()
