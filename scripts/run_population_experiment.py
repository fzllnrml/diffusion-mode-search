from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import load_config, setup_logging, resolve_checkpoint, get_results_dir
from src.models.diffusion import DiffusionModel
from src.algorithms.mode_finder_v3_f2 import ModeFinderV3F2
from src.algorithms.mode_finder_v3_f3 import ModeFinderV3F3
from src.algorithms.baseline import BaselineModeFinder
from src.utils.distribution import GaussianMixture
from src.utils.metrics import evaluate_modes
from src.utils.device import resolve_device
from src.utils.plotting import _setup_style, _color, _label, _marker

logger = logging.getLogger(__name__)


def _train_or_load(mixture, cfg, device, checkpoint_dir, tag):
    ckpt_path = resolve_checkpoint(checkpoint_dir, dim=cfg.dim, tag=tag)
    model = DiffusionModel(
        dim=cfg.dim, T=cfg.model.T,
        beta_start=cfg.model.beta_start, beta_end=cfg.model.beta_end,
        hidden_dims=cfg.model.hidden_dims, device=device,
    )
    if ckpt_path.exists():
        model.load_checkpoint(str(ckpt_path))
        if model._train_step >= cfg.training.num_steps:
            return model
        remaining = cfg.training.num_steps - model._train_step
    else:
        remaining = cfg.training.num_steps

    model.train_on_data(
        sample_fn=lambda n: mixture.sample_torch(n, device),
        num_steps=remaining,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.learning_rate, lr_min=cfg.training.lr_min,
        scheduler=cfg.training.scheduler,
        log_every=max(1, remaining // 10),
        save_every=max(1, remaining // 4),
        save_path=str(ckpt_path),
    )
    model.save_checkpoint(str(ckpt_path))
    return model


def _sweep_one_method(
    method_key: str,   # "v3f2" или "v3f3"
    model,
    true_modes,
    cfg,
    n_particles_list: List[int],
    n_runs: int,
    output_dir: str,
    K: int,
):
    eps = cfg.metrics.eps_hit
    x_min, x_max = cfg.distribution.x_min, cfg.distribution.x_max
    dim_label = f"{cfg.dim}D"

    data = {n: {"recall": [], "soft_error": [], "nfe": []} for n in n_particles_list}

    for n_particles in n_particles_list:
        logger.info("  %s n_particles=%d", method_key, n_particles)
        for run_seed in range(n_runs):
            if method_key == "v3f2":
                mf = cfg.mode_finder_v3_f2
                finder = ModeFinderV3F2(
                    model=model,
                    timesteps=mf.timesteps, step_size=mf.step_size,
                    n_particles=n_particles,
                    merge_factor=mf.merge_factor, merge_radius_min=mf.merge_radius_min,
                    ascent_steps=mf.ascent_steps, normalize_score=mf.normalize_score,
                    refine_steps=mf.refine_steps, refine_step_scale=mf.refine_step_scale,
                    ddim_steps=mf.ddim_steps, init_stop_t=mf.init_stop_t,
                    x_min=x_min, x_max=x_max,
                )
            else:  # v3f3
                mf = cfg.mode_finder_v3_f3
                finder = ModeFinderV3F3(
                    model=model,
                    t_start=mf.t_start, t_end=mf.t_end,
                    ode_steps_coarse=mf.ode_steps_coarse, ode_steps_fine=mf.ode_steps_fine,
                    use_adaptive_step=mf.use_adaptive_step,
                    trace_stability_threshold=mf.trace_stability_threshold,
                    n_substeps_per_interval=mf.n_substeps_per_interval,
                    n_particles=n_particles, cluster_every=mf.cluster_every,
                    merge_factor=mf.merge_factor, merge_radius_min=mf.merge_radius_min,
                    n_trace_probe=mf.n_trace_probe, hessian_fd_eps=mf.hessian_fd_eps,
                    refine_steps=mf.refine_steps, refine_alpha=mf.refine_alpha,
                    ddim_steps=mf.ddim_steps, x_min=x_min, x_max=x_max,
                )

            result = finder.find_modes(seed=run_seed, verbose=False)
            m = evaluate_modes(true_modes, result.modes, eps)
            data[n_particles]["recall"].append(m.recall)
            data[n_particles]["soft_error"].append(m.soft_error)
            data[n_particles]["nfe"].append(result.nfe)

    agg = {
        "recall_mean": [], "recall_std": [],
        "soft_mean":   [], "soft_std":   [],
        "nfe_mean":    [],
    }
    for n in n_particles_list:
        for key, src in [("recall_mean", "recall"), ("recall_std", "recall"),
                         ("soft_mean", "soft_error"), ("soft_std", "soft_error"),
                         ("nfe_mean", "nfe")]:
            arr = np.array(data[n][src])
            agg[key].append(arr.mean() if "mean" in key else arr.std())

    out = get_results_dir(output_dir, method_key, "sweep_population")

    print(f"\n  {method_key} — sweep n_particles ({dim_label}, K={K})")
    print(f"  {'n':>6} | {'recall':>14} | {'soft_error':>14} | {'NFE':>8}")
    print("  " + "-" * 50)
    for i, n in enumerate(n_particles_list):
        r = f"{agg['recall_mean'][i]:.3f}±{agg['recall_std'][i]:.3f}"
        s = f"{agg['soft_mean'][i]:.3f}±{agg['soft_std'][i]:.3f}"
        print(f"  {n:>6} | {r:>14} | {s:>14} | {agg['nfe_mean'][i]:>8.0f}")

    _setup_style()
    nfe_arr = np.array(agg["nfe_mean"])

    for metric, mean_key, std_key, ylabel in [
        ("recall",     "recall_mean", "recall_std", "Recall"),
        ("soft_error", "soft_mean",   "soft_std",   "Soft Error"),
    ]:
        fig, ax = plt.subplots()
        mean_arr = np.array(agg[mean_key])
        std_arr  = np.array(agg[std_key])

        ax.plot(n_particles_list, mean_arr, color=_color(method_key),
                marker=_marker(method_key), label=_label(method_key), linewidth=2)
        ax.fill_between(n_particles_list, mean_arr - std_arr, mean_arr + std_arr,
                        color=_color(method_key), alpha=0.15)

        ax.set_xlabel("n_particles")
        ax.set_ylabel(ylabel)
        if metric == "recall":
            ax.set_ylim(-0.05, 1.15)
        ax.set_title(f"{method_key}: {ylabel} vs n_particles ({dim_label}, K={K})")
        ax.legend()
        plt.tight_layout()
        p = out / f"sweep_population_{metric}_vs_n_{dim_label}_K{K}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  График: {p}")

    fig, ax = plt.subplots()
    mean_r = np.array(agg["recall_mean"])
    std_r  = np.array(agg["recall_std"])
    ax.plot(nfe_arr, mean_r, color=_color(method_key),
            marker=_marker(method_key), label=_label(method_key), linewidth=2)
    ax.fill_between(nfe_arr, mean_r - std_r, mean_r + std_r,
                    color=_color(method_key), alpha=0.15)
    ax.set_xlabel("NFE")
    ax.set_ylabel("Recall")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title(f"{method_key}: Recall vs NFE ({dim_label}, K={K})")
    ax.legend()
    plt.tight_layout()
    p = out / f"sweep_population_recall_vs_nfe_{dim_label}_K{K}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  График: {p}")

    json_data = {
        "method": method_key,
        "n_particles": n_particles_list,
        **{k: [float(v) for v in vs] for k, vs in agg.items()},
        "raw": {str(n): {x: data[n][x] for x in ["recall", "soft_error", "nfe"]}
                for n in n_particles_list},
        "config": {"dim": cfg.dim, "K": K, "n_runs": n_runs},
    }
    p = out / f"sweep_population_{dim_label}_K{K}.json"
    with open(p, "w") as f:
        json.dump(json_data, f, indent=2, default=float)
    logger.info("JSON: %s", p)

    return agg


def sweep_population(
    cfg,
    K: int = 4,
    n_particles_list: List[int] = None,
    n_runs: int = 10,
    method: str = "v3",
    checkpoint_dir: str = "./checkpoints",
    output_dir: str = "./results",
):
    if n_particles_list is None:
        n_particles_list = [10, 20, 30, 50, 75, 100]

    run_f2 = method in ("v3f2", "v3")
    run_f3 = method in ("v3f3", "v3")

    device = resolve_device(cfg.device)

    logger.info("SWEEP POPULATION: dim=%d, K=%d, n_runs=%d, n_p=%s",
                cfg.dim, K, n_runs, n_particles_list)

    mixture = GaussianMixture.random(
        K=K, dim=cfg.dim, sigma_range=(0.5, 1.2),
        min_sep=2.0, bounds=(cfg.distribution.x_min, cfg.distribution.x_max),
        seed=42,
    )
    true_modes = mixture.mode_locations
    tag = f"dim{cfg.dim}_K{K}_pop_dseed42"
    model = _train_or_load(mixture, cfg, device, checkpoint_dir, tag)

    results = {}

    if run_f2:
        results["v3f2"] = _sweep_one_method(
            "v3f2", model, true_modes, cfg,
            n_particles_list, n_runs, output_dir, K,
        )

    if run_f3:
        results["v3f3"] = _sweep_one_method(
            "v3f3", model, true_modes, cfg,
            n_particles_list, n_runs, output_dir, K,
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sweep по n_particles для v3f2 и v3f3"
    )
    parser.add_argument("--config", type=str, default="configs/presets/dim1.yaml")
    parser.add_argument(
        "--method", type=str, default="v3",
        choices=["v3f2", "v3f3", "v3"],
        help=(
            "v3f2 → results/v3f2/sweep_population/\n"
            "v3f3 → results/v3f3/sweep_population/\n"
            "v3   → оба, каждый в свою папку"
        ),
    )
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument(
        "--n-particles", type=str, default="10,20,30,50,75,100",
        help="Список через запятую",
    )
    parser.add_argument("--n-runs",       type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--output-dir",     type=str, default=None)
    parser.add_argument("--dim", type=int, default=None,
                        help="Переопределить dim из конфига")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging)

    if args.dim is not None:
        cfg.dim = args.dim

    n_particles_list = [int(x) for x in args.n_particles.split(",")]
    ckpt_dir = args.checkpoint_dir or cfg.checkpoint.checkpoint_dir
    out_dir  = args.output_dir     or cfg.outputs.output_dir

    t0 = time.perf_counter()
    sweep_population(
        cfg=cfg,
        K=args.K,
        n_particles_list=n_particles_list,
        n_runs=args.n_runs,
        method=args.method,
        checkpoint_dir=ckpt_dir,
        output_dir=out_dir,
    )
    logger.info("Завершено за %.1f сек", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
