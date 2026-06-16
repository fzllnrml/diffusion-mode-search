from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch

from src.config import load_config, setup_logging
from src.utils.device import resolve_device
from src.utils.distribution import GaussianMixture
from src.utils.metrics import evaluate_modes

from src.algorithms.mode_finder import ModeFinder
from src.algorithms.mode_finder_v2 import ModeFinderV2
from src.algorithms.mode_finder_v3_f1 import ModeFinderV3F1
from src.algorithms.mode_finder_v3_f2 import ModeFinderV3F2
from src.algorithms.mode_finder_v3_f3 import ModeFinderV3F3
from src.algorithms.baseline import BaselineModeFinder

from scripts.run_all_experiments import _train_or_load


logger = logging.getLogger(__name__)


def evaluate_many_eps(true_modes, found_modes, eps_values):
    out = {}
    for eps in eps_values:
        m = evaluate_modes(true_modes, found_modes, eps)
        out[str(eps)] = {
            "recall": float(m.recall),
            "precision": float(m.precision),
            "f1": float(m.f1),
            "soft_error": float(m.soft_error),
            "n_found": int(m.n_found),
        }
    return out


def run_method_once(method_key, model, cfg, seed=0):
    x_min, x_max = cfg.distribution.x_min, cfg.distribution.x_max

    if method_key == "v1":
        mf = cfg.mode_finder_v1
        finder = ModeFinder(
            model=model,
            timesteps=mf.timesteps,
            step_size=mf.step_size,
            split_eps=mf.split_eps,
            split_threshold=mf.split_threshold,
            merge_radius=mf.merge_radius,
            ascent_steps=mf.ascent_steps,
            refine_steps=mf.refine_steps,
            refine_step_scale=mf.refine_step_scale,
            n_starts=mf.n_starts,
            starts_min_sep=mf.starts_min_sep,
            split_directions=mf.split_directions,
            x_min=x_min,
            x_max=x_max,
        )

    elif method_key == "v2":
        mf = cfg.mode_finder_v2
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

    elif method_key == "v3f1":
        mf = cfg.mode_finder_v3_f1
        finder = ModeFinderV3F1(
            model=model,
            timesteps=mf.timesteps,
            step_size=mf.step_size,
            split_eps=mf.split_eps,
            softness_threshold=mf.softness_threshold,
            amplification_threshold=mf.amplification_threshold,
            tau_abs_min=mf.tau_abs_min,
            max_split_directions=mf.max_split_directions,
            hessian_fd_eps=mf.hessian_fd_eps,
            merge_factor=mf.merge_factor,
            merge_radius_min=mf.merge_radius_min,
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

    elif method_key == "v3f2":
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
            x_min=x_min,
            x_max=x_max,
        )

    elif method_key == "v3f3":
        mf = cfg.mode_finder_v3_f3
        finder = ModeFinderV3F3(
            model=model,
            t_start=mf.t_start,
            t_end=mf.t_end,
            ode_steps_coarse=mf.ode_steps_coarse,
            ode_steps_fine=mf.ode_steps_fine,
            use_adaptive_step=mf.use_adaptive_step,
            trace_stability_threshold=mf.trace_stability_threshold,
            n_substeps_per_interval=mf.n_substeps_per_interval,
            n_particles=mf.n_particles,
            cluster_every=mf.cluster_every,
            merge_factor=mf.merge_factor,
            merge_radius_min=mf.merge_radius_min,
            n_trace_probe=mf.n_trace_probe,
            hessian_fd_eps=mf.hessian_fd_eps,
            refine_steps=mf.refine_steps,
            refine_alpha=mf.refine_alpha,
            ddim_steps=mf.ddim_steps,
            x_min=x_min,
            x_max=x_max,
        )

    else:
        raise ValueError(f"Unknown method: {method_key}")

    result = finder.find_modes(seed=seed, verbose=False)
    return result.modes, int(result.nfe)


def run_baseline_once(label, model, cfg, seed_offset=0, n_samples_override=None):
    torch.manual_seed(seed_offset)
    np.random.seed(seed_offset)

    refine_steps = 0 if label == "b0" else 10
    n_samples = cfg.baseline.n_samples if n_samples_override is None else int(n_samples_override)

    finder = BaselineModeFinder(
        model=model,
        n_samples=n_samples,
        refine_steps=refine_steps,
        refine_alpha=cfg.baseline.refine_alpha,
        merge_radius=cfg.baseline.merge_radius,
    )

    result = finder.find_modes()
    return result.modes, int(result.nfe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/presets/dim10.yaml")
    parser.add_argument("--methods", default="v2,v3f1,v3f2,v3f3,b0,b10")
    parser.add_argument("--k-values", default="2,3,4,5,6,7")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-init-seeds", type=int, default=3)
    parser.add_argument("--eps", default="1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--output-dir", default="./results_10d/eps_sensitivity_all")
    parser.add_argument("--baseline-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging)

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    eps_values = [float(x.strip()) for x in args.eps.split(",") if x.strip()]

    device = resolve_device(cfg.device)

    logger.info(
        "EPS SENSITIVITY ALL: dim=%d K=%s methods=%s n_seeds=%d n_init_seeds=%d eps=%s",
        cfg.dim, k_values, methods, args.n_seeds, args.n_init_seeds, eps_values
    )

    payload: Dict[str, Any] = {
        "config": {
            "config_path": args.config,
            "dim": cfg.dim,
            "k_values": k_values,
            "methods": methods,
            "n_seeds": args.n_seeds,
            "n_init_seeds": args.n_init_seeds,
            "eps_values": eps_values,
            "baseline_samples": cfg.baseline.n_samples if args.baseline_samples is None else args.baseline_samples,
            "note": "Each method is run once per K/dist_seed/init_seed; same found modes are evaluated at multiple eps_hit radii.",
        },
        "runs": [],
        "summary_by_k": {},
        "summary_overall": {},
    }

    for K in k_values:
        for dist_seed in range(args.n_seeds):
            mixture = GaussianMixture.random(
                K=K,
                dim=cfg.dim,
                sigma_range=(0.5, 1.2),
                min_sep=2.0,
                bounds=(cfg.distribution.x_min, cfg.distribution.x_max),
                seed=dist_seed,
            )

            true_modes = mixture.mode_locations
            tag = f"dim{cfg.dim}_K{K}_dseed{dist_seed}"
            model = _train_or_load(mixture, cfg, device, args.checkpoint_dir, tag)

            for init_seed in range(args.n_init_seeds):
                for method in methods:
                    logger.info(
                        "RUN method=%s K=%d dseed=%d init_seed=%d",
                        method, K, dist_seed, init_seed
                    )

                    if method in ("b0", "b10"):
                        seed_offset = init_seed + (1000 if method == "b10" else 0)
                        modes, nfe = run_baseline_once(
                            method,
                            model,
                            cfg,
                            seed_offset=seed_offset,
                            n_samples_override=args.baseline_samples,
                        )
                    else:
                        modes, nfe = run_method_once(method, model, cfg, seed=init_seed)

                    metrics_by_eps = evaluate_many_eps(true_modes, modes, eps_values)
                    n_modes_returned = int(len(modes) if np.ndim(modes) > 0 else 0)

                    run = {
                        "method": method,
                        "K": K,
                        "dist_seed": dist_seed,
                        "init_seed": init_seed,
                        "nfe": nfe,
                        "n_modes_returned": n_modes_returned,
                        "metrics_by_eps": metrics_by_eps,
                    }
                    payload["runs"].append(run)

                    parts = []
                    for eps in eps_values:
                        m = metrics_by_eps[str(eps)]
                        parts.append(
                            f"eps={eps}: R={m['recall']:.3f}, P={m['precision']:.3f}, "
                            f"F1={m['f1']:.3f}, soft={m['soft_error']:.3f}"
                        )

                    logger.info(
                        "DONE method=%s K=%d dseed=%d init_seed=%d nfe=%d returned=%d | %s",
                        method, K, dist_seed, init_seed, nfe, n_modes_returned, " | ".join(parts)
                    )

    # summary by K
    for K in k_values:
        payload["summary_by_k"][str(K)] = {}
        for method in methods:
            runs = [r for r in payload["runs"] if r["K"] == K and r["method"] == method]
            payload["summary_by_k"][str(K)][method] = summarize_runs(runs, eps_values)

    # overall summary across all K
    for method in methods:
        runs = [r for r in payload["runs"] if r["method"] == method]
        payload["summary_overall"][method] = summarize_runs(runs, eps_values)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eps_sensitivity_all_dim{cfg.dim}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\nOVERALL SUMMARY")
    for method in methods:
        print(f"\n{method}")
        s_all = payload["summary_overall"][method]
        for eps in eps_values:
            key = str(eps)
            s = s_all[key]
            print(
                f"  eps={eps}: "
                f"recall={s['recall']['mean']:.4f}±{s['recall']['std']:.4f}, "
                f"precision={s['precision']['mean']:.4f}±{s['precision']['std']:.4f}, "
                f"f1={s['f1']['mean']:.4f}±{s['f1']['std']:.4f}, "
                f"soft={s['soft_error']['mean']:.4f}±{s['soft_error']['std']:.4f}, "
                f"n_found={s['n_found']['mean']:.1f}±{s['n_found']['std']:.1f}"
            )
        print(
            f"  nfe={s_all['nfe']['mean']:.0f}±{s_all['nfe']['std']:.0f}, "
            f"returned={s_all['n_modes_returned']['mean']:.1f}±{s_all['n_modes_returned']['std']:.1f}"
        )

    print(f"\nSaved to {out_path}")


def summarize_runs(runs, eps_values):
    out = {}

    for eps in eps_values:
        key = str(eps)
        out[key] = {}
        for metric in ["recall", "precision", "f1", "soft_error", "n_found"]:
            arr = np.array([r["metrics_by_eps"][key][metric] for r in runs], dtype=float)
            out[key][metric] = {
                "mean": float(arr.mean()) if len(arr) else None,
                "std": float(arr.std()) if len(arr) else None,
            }

    for metric in ["nfe", "n_modes_returned"]:
        arr = np.array([r[metric] for r in runs], dtype=float)
        out[metric] = {
            "mean": float(arr.mean()) if len(arr) else None,
            "std": float(arr.std()) if len(arr) else None,
        }

    return out


if __name__ == "__main__":
    main()
