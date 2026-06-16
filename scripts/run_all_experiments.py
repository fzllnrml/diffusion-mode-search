from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.config import load_config, setup_logging, resolve_checkpoint, get_results_dir
from src.models.diffusion import DiffusionModel
from src.algorithms.mode_finder import ModeFinder
from src.algorithms.mode_finder_v2 import ModeFinderV2
from src.algorithms.mode_finder_v3_f1 import ModeFinderV3F1
from src.algorithms.mode_finder_v3_f2 import ModeFinderV3F2
from src.algorithms.mode_finder_v3_f3 import ModeFinderV3F3
from src.algorithms.baseline import BaselineModeFinder
from src.utils.distribution import GaussianMixture
from src.utils.metrics import evaluate_modes
from src.utils.device import resolve_device
from src.utils.plotting import (
    plot_metric_vs_k, plot_metric_vs_nfe, plot_starts_comparison,
    print_summary_table, format_mean_std,
)

logger = logging.getLogger(__name__)


def _methods_to_run(method: str) -> Dict[str, bool]:
    m = method.lower()
    return {
        "v1":   m in ("v1",   "both", "v3all"),
        "v2":   m in ("v2",   "both", "v3all"),
        "v3f1": m in ("v3f1", "v3",   "v3all"),
        "v3f2": m in ("v3f2", "v3",   "v3all"),
        "v3f3": m in ("v3f3", "v3",   "v3all"),
    }


def _active(flags: Dict[str, bool]) -> List[str]:
    ORDER = ["v1", "v2", "v3f1", "v3f2", "v3f3"]
    return [k for k in ORDER if flags.get(k)]


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
        logger.info("Доучиваем %s: %d шагов", tag, remaining)
    else:
        remaining = cfg.training.num_steps
        logger.info("Обучаем %s: %d шагов", tag, remaining)

    model.train_on_data(
        sample_fn=lambda n: mixture.sample_torch(n, device),
        num_steps=remaining,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.learning_rate,
        lr_min=cfg.training.lr_min,
        scheduler=cfg.training.scheduler,
        log_every=max(1, remaining // 10),
        save_every=max(1, remaining // 4),
        save_path=str(ckpt_path),
    )
    model.save_checkpoint(str(ckpt_path))
    return model


def _run_method(method_key, model, true_modes, cfg, seed=0):
    eps = cfg.metrics.eps_hit
    x_min, x_max = cfg.distribution.x_min, cfg.distribution.x_max

    if method_key == "v1":
        mf = cfg.mode_finder_v1
        finder = ModeFinder(
            model=model,
            timesteps=mf.timesteps, step_size=mf.step_size,
            split_eps=mf.split_eps, split_threshold=mf.split_threshold,
            merge_radius=mf.merge_radius, ascent_steps=mf.ascent_steps,
            refine_steps=mf.refine_steps, refine_step_scale=mf.refine_step_scale,
            n_starts=mf.n_starts, starts_min_sep=mf.starts_min_sep,
            split_directions=mf.split_directions, x_min=x_min, x_max=x_max,
        )

    elif method_key == "v2":
        mf = cfg.mode_finder_v2
        finder = ModeFinderV2(
            model=model,
            timesteps=mf.timesteps, step_size=mf.step_size,
            split_method=mf.split_method, split_eps=mf.split_eps,
            split_threshold=mf.split_threshold, hessian_fd_eps=mf.hessian_fd_eps,
            hessian_split_eigenvalue_threshold=mf.hessian_split_eigenvalue_threshold,
            max_split_directions=mf.max_split_directions,
            merge_radius=mf.merge_radius, adaptive_merge=mf.adaptive_merge,
            ascent_steps=mf.ascent_steps, normalize_score=mf.normalize_score,
            refine_steps=mf.refine_steps, refine_step_scale=mf.refine_step_scale,
            n_starts=mf.n_starts, start_method=mf.start_method,
            ddim_steps=mf.ddim_steps, n_pilot_multiplier=mf.n_pilot_multiplier,
            pilot_cluster_radius=mf.pilot_cluster_radius,
            starts_min_sep=mf.starts_min_sep, max_active_per_start=mf.max_active_per_start,
            x_min=x_min, x_max=x_max,
        )

    elif method_key == "v3f1":
        mf = cfg.mode_finder_v3_f1
        finder = ModeFinderV3F1(
            model=model,
            timesteps=mf.timesteps, step_size=mf.step_size,
            split_eps=mf.split_eps, softness_threshold=mf.softness_threshold,
            amplification_threshold=mf.amplification_threshold,
            tau_abs_min=mf.tau_abs_min, max_split_directions=mf.max_split_directions,
            hessian_fd_eps=mf.hessian_fd_eps, merge_factor=mf.merge_factor,
            merge_radius_min=mf.merge_radius_min, ascent_steps=mf.ascent_steps,
            normalize_score=mf.normalize_score, refine_steps=mf.refine_steps,
            refine_step_scale=mf.refine_step_scale, n_starts=mf.n_starts,
            start_method=mf.start_method, ddim_steps=mf.ddim_steps,
            n_pilot_multiplier=mf.n_pilot_multiplier,
            pilot_cluster_radius=mf.pilot_cluster_radius,
            starts_min_sep=mf.starts_min_sep,
            max_active_per_start=mf.max_active_per_start,
            x_min=x_min, x_max=x_max,
        )

    elif method_key == "v3f2":
        mf = cfg.mode_finder_v3_f2
        finder = ModeFinderV3F2(
            model=model,
            timesteps=mf.timesteps, step_size=mf.step_size,
            n_particles=mf.n_particles, merge_factor=mf.merge_factor,
            merge_radius_min=mf.merge_radius_min, ascent_steps=mf.ascent_steps,
            normalize_score=mf.normalize_score, refine_steps=mf.refine_steps,
            refine_step_scale=mf.refine_step_scale,
            ddim_steps=mf.ddim_steps, init_stop_t=mf.init_stop_t,
            x_min=x_min, x_max=x_max,
        )

    elif method_key == "v3f3":
        mf = cfg.mode_finder_v3_f3
        finder = ModeFinderV3F3(
            model=model,
            t_start=mf.t_start, t_end=mf.t_end,
            ode_steps_coarse=mf.ode_steps_coarse, ode_steps_fine=mf.ode_steps_fine,
            use_adaptive_step=mf.use_adaptive_step,
            trace_stability_threshold=mf.trace_stability_threshold,
            n_substeps_per_interval=mf.n_substeps_per_interval,
            n_particles=mf.n_particles, cluster_every=mf.cluster_every,
            merge_factor=mf.merge_factor, merge_radius_min=mf.merge_radius_min,
            n_trace_probe=mf.n_trace_probe, hessian_fd_eps=mf.hessian_fd_eps,
            refine_steps=mf.refine_steps, refine_alpha=mf.refine_alpha,
            ddim_steps=mf.ddim_steps, x_min=x_min, x_max=x_max,
        )
    else:
        raise ValueError(f"Неизвестный метод: {method_key}")

    result = finder.find_modes(seed=seed, verbose=False)
    metrics = evaluate_modes(true_modes, result.modes, eps)
    return {
        "recall": metrics.recall, "precision": metrics.precision,
        "f1": metrics.f1, "soft_error": metrics.soft_error,
        "nfe": result.nfe, "n_found": metrics.n_found,
    }


def _run_baseline(model, true_modes, cfg, n_samples, refine_steps, seed_offset=0):
    torch.manual_seed(seed_offset)
    np.random.seed(seed_offset)
    bl_c = cfg.baseline
    finder = BaselineModeFinder(
        model=model, n_samples=n_samples, refine_steps=refine_steps,
        refine_alpha=bl_c.refine_alpha, merge_radius=bl_c.merge_radius,
    )
    result = finder.find_modes()
    metrics = evaluate_modes(true_modes, result.modes, cfg.metrics.eps_hit)
    return {
        "recall": metrics.recall, "precision": metrics.precision,
        "f1": metrics.f1, "soft_error": metrics.soft_error,
        "nfe": result.nfe, "n_found": metrics.n_found,
    }


def experiment_sweep_k(
    cfg,
    k_values: List[int] = None,
    n_seeds: int = 5,
    n_init_seeds: int = 3,
    method: str = "both",
    checkpoint_dir: str = "./checkpoints",
    output_dir: str = "./results",
):
    if k_values is None:
        k_values = list(range(2, 8))

    flags = _methods_to_run(method)
    active = _active(flags)
    all_methods = active + ["b0", "b10"]
    dim_label = f"{cfg.dim}D"
    device = resolve_device(cfg.device)

    all_results = {
        m: {x: [] for x in ["recall", "soft_error", "nfe"]}
        for m in all_methods
    }

    logger.info("SWEEP K: dim=%d, K=%s, methods=%s", cfg.dim, k_values, all_methods)

    for K in k_values:
        logger.info("--- K=%d ---", K)
        k_data = {m: {x: [] for x in ["recall", "soft_error", "nfe"]}
                  for m in all_methods}

        for dist_seed in range(n_seeds):
            try:
                mixture = GaussianMixture.random(
                    K=K, dim=cfg.dim, sigma_range=(0.5, 1.2),
                    min_sep=2.0, bounds=(cfg.distribution.x_min, cfg.distribution.x_max),
                    seed=dist_seed,
                )
            except RuntimeError as e:
                logger.warning("K=%d dseed=%d пропущен: %s", K, dist_seed, e)
                continue

            true_modes = mixture.mode_locations
            tag = f"dim{cfg.dim}_K{K}_dseed{dist_seed}"
            model = _train_or_load(mixture, cfg, device, checkpoint_dir, tag)

            for init_seed in range(n_init_seeds):
                for m in active:
                    logger.info(
                        "RUN method=%s K=%d dseed=%d init_seed=%d",
                        m, K, dist_seed, init_seed
                    )
                    r = _run_method(m, model, true_modes, cfg, seed=init_seed)
                    for x in ["recall", "soft_error", "nfe"]:
                        k_data[m][x].append(r[x])

                for label, refine, offset in [("b0", 0, 0), ("b10", 10, 1000)]:
                    r = _run_baseline(model, true_modes, cfg,
                                      cfg.baseline.n_samples, refine,
                                      seed_offset=init_seed + offset)
                    for x in ["recall", "soft_error", "nfe"]:
                        k_data[label][x].append(r[x])

        for m in all_methods:
            for x in ["recall", "soft_error", "nfe"]:
                all_results[m][x].append(k_data[m][x])

    plot_data = {m: {} for m in all_methods}
    for m in all_methods:
        for x in ["recall", "soft_error", "nfe"]:
            means, stds = [], []
            for k_list in all_results[m][x]:
                arr = np.array(k_list) if k_list else np.array([0.0])
                means.append(arr.mean()); stds.append(arr.std())
            plot_data[m][x] = (np.array(means), np.array(stds))

    print(f"\n{'='*70}\n  SWEEP K ({dim_label})\n{'='*70}")
    print_summary_table(plot_data, k_values)

    combined_out = get_results_dir(output_dir, method, "sweep_k")
    for metric in ["recall", "soft_error"]:
        plot_metric_vs_k(plot_data, k_values, metric=metric,
                         title=f"{metric} vs K ({dim_label})",
                         save_path=str(combined_out / f"sweep_k_{metric}_{dim_label}.png"),
                         dim_label=dim_label)

    json_data = {
        m: {x: {"mean": plot_data[m][x][0].tolist(),
                "std":  plot_data[m][x][1].tolist()}
            for x in ["recall", "soft_error", "nfe"]}
        for m in all_methods
    }
    json_data["k_values"] = k_values
    json_data["config"] = {"dim": cfg.dim, "n_seeds": n_seeds,
                           "n_init_seeds": n_init_seeds, "methods": all_methods}

    p = combined_out / f"sweep_k_{dim_label}.json"
    with open(p, "w") as f:
        json.dump(json_data, f, indent=2, default=float)
    logger.info("JSON (combined): %s", p)

    for m_key in active:
        per_out = get_results_dir(output_dir, m_key, "sweep_k")
        per_data = {
            "method": m_key,
            "k_values": k_values,
            "recall": {"mean": plot_data[m_key]["recall"][0].tolist(),
                       "std":  plot_data[m_key]["recall"][1].tolist()},
            "soft_error": {"mean": plot_data[m_key]["soft_error"][0].tolist(),
                           "std":  plot_data[m_key]["soft_error"][1].tolist()},
            "nfe": {"mean": plot_data[m_key]["nfe"][0].tolist(),
                    "std":  plot_data[m_key]["nfe"][1].tolist()},
        }
        pp = per_out / f"sweep_k_{dim_label}.json"
        with open(pp, "w") as f:
            json.dump(per_data, f, indent=2, default=float)
        logger.info("JSON (%s): %s", m_key, pp)

    return plot_data


def experiment_random_starts(
    cfg,
    K: int = 4,
    n_runs: int = 30,
    method: str = "both",
    checkpoint_dir: str = "./checkpoints",
    output_dir: str = "./results",
):
    flags = _methods_to_run(method)
    STARTS_METHODS = {"v1", "v2", "v3f1"}
    active = [m for m in _active(flags) if m in STARTS_METHODS]
    if not active and any(flags.get(m) for m in ("v3f2", "v3f3")):
        logger.warning(
            "sweep_starts: v3f2/v3f3 исключены (population-based). "
            "Используйте run_population_experiment.py для sweep по n_particles."
        )
    if not active:
        logger.info("sweep_starts: нет подходящих методов для запуска.")
        return {}

    all_methods = active + ["b0", "b10"]
    dim_label = f"{cfg.dim}D"
    device = resolve_device(cfg.device)

    results = {m: {"recall": [], "soft_error": [], "nfe": []} for m in all_methods}

    logger.info("SWEEP STARTS: dim=%d, K=%d, n_runs=%d, methods=%s",
                cfg.dim, K, n_runs, all_methods)

    mixture = GaussianMixture.random(
        K=K, dim=cfg.dim, sigma_range=(0.5, 1.2),
        min_sep=2.0, bounds=(cfg.distribution.x_min, cfg.distribution.x_max),
        seed=42,
    )
    true_modes = mixture.mode_locations
    tag = f"dim{cfg.dim}_K{K}_starts_dseed42"
    model = _train_or_load(mixture, cfg, device, checkpoint_dir, tag)

    for run_seed in range(n_runs):
        for m in active:
            r = _run_method(m, model, true_modes, cfg, seed=run_seed)
            for x in ["recall", "soft_error", "nfe"]:
                results[m][x].append(r[x])

        for label, refine, offset in [("b0", 0, 0), ("b10", 10, 1000)]:
            r = _run_baseline(model, true_modes, cfg,
                              cfg.baseline.n_samples, refine,
                              seed_offset=run_seed + offset)
            for x in ["recall", "soft_error", "nfe"]:
                results[label][x].append(r[x])

    print(f"\n{'='*70}\n  SWEEP STARTS ({dim_label}, K={K}, n={n_runs})\n{'='*70}")
    summary = {}
    for m in all_methods:
        summary[m] = {}
        for x in ["recall", "soft_error", "nfe"]:
            arr = np.array(results[m][x])
            mn, sd = arr.mean(), arr.std()
            summary[m][x] = (mn, sd)
            print(f"  {m:>8}/{x:<12}: {format_mean_std(mn, sd)}")

    combined_out = get_results_dir(output_dir, method, "sweep_starts")
    plot_starts_comparison(summary,
                           title=f"Случайные старты ({dim_label}, K={K}, n={n_runs})",
                           save_path=str(combined_out / f"starts_{dim_label}_K{K}.png"))

    json_data = {
        m: {x: {"values": results[m][x],
                "mean": float(summary[m][x][0]),
                "std":  float(summary[m][x][1])}
            for x in ["recall", "soft_error", "nfe"]}
        for m in all_methods
    }
    json_data["config"] = {"dim": cfg.dim, "K": K, "n_runs": n_runs,
                           "methods": all_methods,
                           "note": "v3f2/v3f3 excluded (population-based)"}
    p = combined_out / f"starts_{dim_label}_K{K}.json"
    with open(p, "w") as f:
        json.dump(json_data, f, indent=2, default=float)
    logger.info("JSON (combined): %s", p)

    for m_key in active:
        per_out = get_results_dir(output_dir, m_key, "sweep_starts")
        per_data = {
            "method": m_key, "K": K, "n_runs": n_runs,
            **{x: {"values": results[m_key][x],
                   "mean": float(summary[m_key][x][0]),
                   "std":  float(summary[m_key][x][1])}
               for x in ["recall", "soft_error", "nfe"]},
        }
        pp = per_out / f"starts_{dim_label}_K{K}.json"
        with open(pp, "w") as f:
            json.dump(per_data, f, indent=2, default=float)
        logger.info("JSON (%s): %s", m_key, pp)

    return summary


def experiment_sweep_nfe(
    cfg,
    K: int = 4,
    baseline_n_samples_list: List[int] = None,
    n_runs: int = 15,
    method: str = "both",
    checkpoint_dir: str = "./checkpoints",
    output_dir: str = "./results",
):
    if baseline_n_samples_list is None:
        baseline_n_samples_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    flags = _methods_to_run(method)
    active = _active(flags)
    dim_label = f"{cfg.dim}D"
    device = resolve_device(cfg.device)

    logger.info("SWEEP NFE: dim=%d, K=%d, n_runs=%d, methods=%s",
                cfg.dim, K, n_runs, active)

    mixture = GaussianMixture.random(
        K=K, dim=cfg.dim, sigma_range=(0.5, 1.2),
        min_sep=2.0, bounds=(cfg.distribution.x_min, cfg.distribution.x_max),
        seed=42,
    )
    true_modes = mixture.mode_locations
    tag = f"dim{cfg.dim}_K{K}_nfe_dseed42"
    model = _train_or_load(mixture, cfg, device, checkpoint_dir, tag)

    ours_stats = {m: {"recalls": [], "softs": [], "nfes": []} for m in active}
    for run_seed in range(n_runs):
        for m in active:
            r = _run_method(m, model, true_modes, cfg, seed=run_seed)
            ours_stats[m]["recalls"].append(r["recall"])
            ours_stats[m]["softs"].append(r["soft_error"])
            ours_stats[m]["nfes"].append(r["nfe"])

    for m, st in ours_stats.items():
        st.update({
            "recall_mean": np.mean(st["recalls"]), "recall_std": np.std(st["recalls"]),
            "soft_mean":   np.mean(st["softs"]),   "soft_std":   np.std(st["softs"]),
            "nfe_mean":    np.mean(st["nfes"]),
        })

    bl_results = {bl: {"recall_mean": [], "recall_std": [],
                        "soft_mean": [], "soft_std": [], "nfe_mean": []}
                  for bl in ["b0", "b10"]}

    for n_samples in baseline_n_samples_list:
        for bl, refine, offset in [("b0", 0, 2000), ("b10", 10, 3000)]:
            recalls, softs, nfes = [], [], []
            for run_seed in range(n_runs):
                r = _run_baseline(model, true_modes, cfg, n_samples, refine,
                                  seed_offset=run_seed + offset)
                recalls.append(r["recall"]); softs.append(r["soft_error"])
                nfes.append(r["nfe"])
            bl_results[bl]["recall_mean"].append(np.mean(recalls))
            bl_results[bl]["recall_std"].append(np.std(recalls))
            bl_results[bl]["soft_mean"].append(np.mean(softs))
            bl_results[bl]["soft_std"].append(np.std(softs))
            bl_results[bl]["nfe_mean"].append(np.mean(nfes))

    nfe_range = np.array(bl_results["b0"]["nfe_mean"])
    plot_data = {}
    for m, st in ours_stats.items():
        plot_data[m] = {
            "recall":     (np.full_like(nfe_range, st["recall_mean"]),
                           np.full_like(nfe_range, st["recall_std"])),
            "soft_error": (np.full_like(nfe_range, st["soft_mean"]),
                           np.full_like(nfe_range, st["soft_std"])),
            "nfe_values": nfe_range,
        }
    for bl in ["b0", "b10"]:
        plot_data[bl] = {
            "recall":     (np.array(bl_results[bl]["recall_mean"]),
                           np.array(bl_results[bl]["recall_std"])),
            "soft_error": (np.array(bl_results[bl]["soft_mean"]),
                           np.array(bl_results[bl]["soft_std"])),
            "nfe_values": np.array(bl_results[bl]["nfe_mean"]),
        }

    combined_out = get_results_dir(output_dir, method, "sweep_nfe")
    for metric in ["recall", "soft_error"]:
        plot_metric_vs_nfe(
            plot_data, baseline_n_samples_list, metric=metric,
            title=f"{metric} vs NFE ({dim_label}, K={K})",
            save_path=str(combined_out / f"sweep_nfe_{metric}_{dim_label}_K{K}.png"),
            dim_label=dim_label,
        )

    json_data = {m: {k: float(v) for k, v in st.items() if not isinstance(v, list)}
                 for m, st in ours_stats.items()}
    json_data["baseline_n_samples"] = baseline_n_samples_list
    for bl in ["b0", "b10"]:
        json_data[bl] = {k: [float(x) for x in vs]
                         for k, vs in bl_results[bl].items()}
    json_data["config"] = {"dim": cfg.dim, "K": K, "n_runs": n_runs, "methods": active}

    p = combined_out / f"sweep_nfe_{dim_label}_K{K}.json"
    with open(p, "w") as f:
        json.dump(json_data, f, indent=2, default=float)
    logger.info("JSON (combined): %s", p)

    for m_key in active:
        st = ours_stats[m_key]
        per_out = get_results_dir(output_dir, m_key, "sweep_nfe")
        per_data = {
            "method": m_key,
            "recall_mean": st["recall_mean"], "recall_std": st["recall_std"],
            "soft_mean": st["soft_mean"],     "soft_std": st["soft_std"],
            "nfe_mean": st["nfe_mean"],
            "baseline_n_samples": baseline_n_samples_list,
            "b0":  {k: [float(x) for x in vs] for k, vs in bl_results["b0"].items()},
            "b10": {k: [float(x) for x in vs] for k, vs in bl_results["b10"].items()},
            "config": {"dim": cfg.dim, "K": K, "n_runs": n_runs},
        }
        pp = per_out / f"sweep_nfe_{dim_label}_K{K}.json"
        with open(pp, "w") as f:
            json.dump(per_data, f, indent=2, default=float)
        logger.info("JSON (%s): %s", m_key, pp)

    return plot_data


def main():
    parser = argparse.ArgumentParser(description="Sweep-эксперименты поиска мод")
    parser.add_argument("--config", type=str, default="configs/presets/dim1.yaml",
                        help="Путь к YAML-конфигурации (preset)")
    parser.add_argument(
        "--experiment", type=str, default="all",
        choices=["all", "sweep_k", "sweep_starts", "sweep_nfe"],
    )
    parser.add_argument(
        "--method", type=str, default="both",
        choices=["v1", "v2", "both", "v3f1", "v3f2", "v3f3", "v3", "v3all"],
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--output-dir",     type=str, default=None)
    parser.add_argument("--n-seeds",      type=int, default=5,
                        help="Число random распределений на K (sweep_k)")
    parser.add_argument("--n-init-seeds", type=int, default=3,
                        help="Число init seeds на распределение (sweep_k)")
    parser.add_argument("--n-runs",       type=int, default=15,
                        help="Число запусков (sweep_starts, sweep_nfe)")
    parser.add_argument("--K",            type=int, default=4,
                        help="Число мод (sweep_starts, sweep_nfe)")
    parser.add_argument("--dim", type=int, default=None,
                        help="Переопределить dim из конфига (любое целое)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging)

    if args.dim is not None:
        cfg.dim = args.dim
    if args.checkpoint_dir:
        cfg.checkpoint.checkpoint_dir = args.checkpoint_dir
    ckpt_dir = cfg.checkpoint.checkpoint_dir
    out_dir = args.output_dir or cfg.outputs.output_dir

    to_run = []
    if args.experiment in ("all", "sweep_k"):      to_run.append("sweep_k")
    if args.experiment in ("all", "sweep_starts"):  to_run.append("sweep_starts")
    if args.experiment in ("all", "sweep_nfe"):     to_run.append("sweep_nfe")

    for exp in to_run:
        t0 = time.perf_counter()
        if exp == "sweep_k":
            experiment_sweep_k(cfg, n_seeds=args.n_seeds, n_init_seeds=args.n_init_seeds,
                               method=args.method, checkpoint_dir=ckpt_dir, output_dir=out_dir)
        elif exp == "sweep_starts":
            experiment_random_starts(cfg, K=args.K, n_runs=args.n_runs,
                                     method=args.method, checkpoint_dir=ckpt_dir, output_dir=out_dir)
        elif exp == "sweep_nfe":
            experiment_sweep_nfe(cfg, K=args.K, n_runs=args.n_runs,
                                 method=args.method, checkpoint_dir=ckpt_dir, output_dir=out_dir)
        logger.info("%s завершён за %.1f сек", exp, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
