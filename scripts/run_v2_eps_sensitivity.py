from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from src.config import load_config, setup_logging
from src.utils.device import resolve_device
from src.utils.distribution import GaussianMixture
from src.utils.metrics import evaluate_modes
from src.algorithms.mode_finder_v2 import ModeFinderV2
from scripts.run_all_experiments import _train_or_load


logger = logging.getLogger(__name__)


def run_v2_once(model, true_modes, cfg, seed: int, eps_values: List[float]) -> Dict[str, Any]:
    mf = cfg.mode_finder_v2
    x_min, x_max = cfg.distribution.x_min, cfg.distribution.x_max

    finder = ModeFinderV2(
        model=model,
        timesteps=mf.timesteps,
        step_size=mf.step_size,
        split_method=mf.split_method,
        split_eps=mf.split_eps,
        split_threshold=mf.split_threshold,
        hessian_fd_eps=mf.hessian_fd_eps,
        hessian_split_eigenvalue_threshold=mf.hessian_split_eigenvalue_threshold,
        max_split_directions=mf.max_split_directions,
        merge_radius=mf.merge_radius,
        adaptive_merge=mf.adaptive_merge,
        ascent_steps=mf.ascent_steps,
        normalize_score=mf.normalize_score,
        refine_steps=mf.refine_steps,
        refine_step_scale=mf.refine_step_scale,
        n_starts=mf.n_starts,
        start_method=mf.start_method,
        ddim_steps=mf.ddim_steps,
        n_pilot_multiplier=mf.n_pilot_multiplier,
        pilot_cluster_radius=mf.pilot_cluster_radius,
        starts_min_sep=mf.starts_min_sep,
        max_active_per_start=mf.max_active_per_start,
        x_min=x_min,
        x_max=x_max,
    )

    result = finder.find_modes(seed=seed, verbose=False)

    metrics_by_eps = {}
    for eps in eps_values:
        metrics = evaluate_modes(true_modes, result.modes, eps)
        metrics_by_eps[str(eps)] = {
            "recall": float(metrics.recall),
            "precision": float(metrics.precision),
            "f1": float(metrics.f1),
            "soft_error": float(metrics.soft_error),
            "n_found": int(metrics.n_found),
        }

    return {
        "nfe": int(result.nfe),
        "n_modes_returned": int(len(result.modes) if np.ndim(result.modes) > 0 else 0),
        "metrics_by_eps": metrics_by_eps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/presets/dim10.yaml")
    parser.add_argument("--K", type=int, default=7)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-init-seeds", type=int, default=3)
    parser.add_argument("--eps", default="1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--output-dir", default="./results_10d/v2_eps_sensitivity_k7")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging)

    eps_values = [float(x.strip()) for x in args.eps.split(",") if x.strip()]
    device = resolve_device(cfg.device)

    logger.info(
        "V2 EPS SENSITIVITY: dim=%d K=%d n_seeds=%d n_init_seeds=%d eps=%s",
        cfg.dim, args.K, args.n_seeds, args.n_init_seeds, eps_values,
    )

    all_runs = []

    for dist_seed in range(args.n_seeds):
        mixture = GaussianMixture.random(
            K=args.K,
            dim=cfg.dim,
            sigma_range=(0.5, 1.2),
            min_sep=2.0,
            bounds=(cfg.distribution.x_min, cfg.distribution.x_max),
            seed=dist_seed,
        )

        true_modes = mixture.mode_locations
        tag = f"dim{cfg.dim}_K{args.K}_dseed{dist_seed}"
        model = _train_or_load(mixture, cfg, device, args.checkpoint_dir, tag)

        for init_seed in range(args.n_init_seeds):
            logger.info(
                "RUN v2_eps_sensitivity K=%d dseed=%d init_seed=%d",
                args.K, dist_seed, init_seed,
            )

            run = run_v2_once(
                model=model,
                true_modes=true_modes,
                cfg=cfg,
                seed=init_seed,
                eps_values=eps_values,
            )

            run["K"] = args.K
            run["dist_seed"] = dist_seed
            run["init_seed"] = init_seed
            all_runs.append(run)

            parts = []
            for eps in eps_values:
                m = run["metrics_by_eps"][str(eps)]
                parts.append(
                    f"eps={eps}: recall={m['recall']:.4f}, "
                    f"precision={m['precision']:.4f}, "
                    f"f1={m['f1']:.4f}, "
                    f"soft={m['soft_error']:.4f}, "
                    f"n_found={m['n_found']}"
                )

            logger.info(
                "DONE v2_eps_sensitivity K=%d dseed=%d init_seed=%d nfe=%d modes=%d | %s",
                args.K,
                dist_seed,
                init_seed,
                run["nfe"],
                run["n_modes_returned"],
                " | ".join(parts),
            )

    summary = {}
    for eps in eps_values:
        key = str(eps)
        summary[key] = {}

        for metric in ["recall", "precision", "f1", "soft_error", "n_found"]:
            arr = np.array(
                [r["metrics_by_eps"][key][metric] for r in all_runs],
                dtype=float,
            )
            summary[key][metric] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            }

    nfe_arr = np.array([r["nfe"] for r in all_runs], dtype=float)
    modes_arr = np.array([r["n_modes_returned"] for r in all_runs], dtype=float)

    summary["nfe"] = {
        "mean": float(nfe_arr.mean()),
        "std": float(nfe_arr.std()),
    }
    summary["n_modes_returned"] = {
        "mean": float(modes_arr.mean()),
        "std": float(modes_arr.std()),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"v2_eps_sensitivity_K{args.K}_dim{cfg.dim}.json"

    payload = {
        "config": {
            "config_path": args.config,
            "dim": cfg.dim,
            "K": args.K,
            "n_seeds": args.n_seeds,
            "n_init_seeds": args.n_init_seeds,
            "eps_values": eps_values,
            "method": "v2_original_eps_sensitivity",
            "note": "V2 is run once per seed/init; the same found modes are evaluated at multiple eps_hit radii.",
        },
        "runs": all_runs,
        "summary": summary,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\nSUMMARY")
    for eps in eps_values:
        key = str(eps)
        s = summary[key]
        print(
            f"eps={eps}: "
            f"recall={s['recall']['mean']:.4f}±{s['recall']['std']:.4f}, "
            f"precision={s['precision']['mean']:.4f}±{s['precision']['std']:.4f}, "
            f"f1={s['f1']['mean']:.4f}±{s['f1']['std']:.4f}, "
            f"soft={s['soft_error']['mean']:.4f}±{s['soft_error']['std']:.4f}, "
            f"n_found={s['n_found']['mean']:.1f}±{s['n_found']['std']:.1f}"
        )

    print(
        f"nfe={summary['nfe']['mean']:.0f}±{summary['nfe']['std']:.0f}, "
        f"returned_modes={summary['n_modes_returned']['mean']:.1f}±{summary['n_modes_returned']['std']:.1f}"
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
