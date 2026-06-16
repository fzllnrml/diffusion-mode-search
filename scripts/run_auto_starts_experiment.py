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
from src.algorithms.auto_starts import (
    StartStrategy, FixedStarts, PilotSampleStarts,
    IncrementalStarts, ScoreGridStarts,
)
from src.algorithms.auto_starts_v2 import SmartFixedV2, CoarseToFineStrategy, get_v2_strategies
from src.algorithms.mode_finder_v3_f1 import ModeFinderV3F1
from src.algorithms.baseline import BaselineModeFinder
from src.utils.distribution import GaussianMixture
from src.utils.metrics import evaluate_modes
from src.utils.device import resolve_device

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
        logger.info("Доучиваем %s: %d шагов (target=%d)",
                    tag, remaining, cfg.training.num_steps)
    else:
        remaining = cfg.training.num_steps
        logger.info("Обучаем %s: %d шагов", tag, remaining)

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


def run_baseline(model, true_modes, cfg, n_samples=1000, refine_steps=0):
    bl_c = cfg.baseline
    bl = BaselineModeFinder(
        model=model, n_samples=n_samples, refine_steps=refine_steps,
        refine_alpha=bl_c.refine_alpha, merge_radius=bl_c.merge_radius,
    )
    result = bl.find_modes()
    metrics = evaluate_modes(true_modes, result.modes, cfg.metrics.eps_hit)
    return {
        "recall": metrics.recall, "precision": metrics.precision,
        "f1": metrics.f1, "soft_error": metrics.soft_error,
        "nfe": result.nfe, "n_found": metrics.n_found,
        "n_starts": n_samples,
    }


def run_strategy(strategy, model, true_modes, mf_kwargs, eps_hit, seed=0):
    result = strategy.find_modes(model, mf_kwargs, seed=seed)
    metrics = evaluate_modes(true_modes, result.modes, eps_hit)
    return {
        "recall": metrics.recall, "precision": metrics.precision,
        "f1": metrics.f1, "soft_error": metrics.soft_error,
        "nfe": result.nfe,
        "nfe_overhead": result.nfe_overhead,
        "nfe_search": result.nfe_search,
        "n_found": metrics.n_found,
        "n_starts": result.n_starts_chosen,
    }


def run_v3f1(model, true_modes, cfg, seed=0):
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
        x_min=cfg.distribution.x_min, x_max=cfg.distribution.x_max,
    )
    result = finder.find_modes(seed=seed, verbose=False)
    metrics = evaluate_modes(true_modes, result.modes, cfg.metrics.eps_hit)
    return {
        "recall": metrics.recall, "precision": metrics.precision,
        "f1": metrics.f1, "soft_error": metrics.soft_error,
        "nfe": result.nfe,
        "nfe_overhead": getattr(result, "nfe_starts", 0),
        "nfe_search":   getattr(result, "nfe_search",  result.nfe),
        "n_found": metrics.n_found,
        "n_starts": mf.n_starts,
    }


def run_experiment(
    cfg,
    k_values: List[int] = None,
    n_dist_seeds: int = 5,
    n_init_seeds: int = 3,
    include_v3f1: bool = False,
    checkpoint_dir: str = "./checkpoints",
    output_dir: str = "./results",
    method: str = "v1v2",
):
    if k_values is None:
        k_values = list(range(2, 8))

    device = resolve_device(cfg.device)
    dim_label = f"{cfg.dim}D"

    mf = cfg.mode_finder_v1
    mf_kwargs = {
        "timesteps":  mf.timesteps, "step_size": mf.step_size,
        "split_eps":  mf.split_eps, "split_threshold": mf.split_threshold,
        "merge_radius": mf.merge_radius, "ascent_steps": mf.ascent_steps,
        "refine_steps": mf.refine_steps, "refine_step_scale": mf.refine_step_scale,
        "starts_min_sep": mf.starts_min_sep,
        "split_directions": mf.split_directions,
        "x_min": cfg.distribution.x_min, "x_max": cfg.distribution.x_max,
        "merge_radius": cfg.mode_finder_v2.merge_radius,
        "split_threshold": cfg.mode_finder_v2.split_threshold,
        "max_active_per_start": cfg.mode_finder_v2.max_active_per_start,
    }

    run_strategies = (method in ("v1v2", "v3f1all"))

    if run_strategies:
        strategies: List[StartStrategy] = [
            FixedStarts(n_starts=1),
            FixedStarts(n_starts=3),
            FixedStarts(n_starts=5),
            FixedStarts(n_starts=10),
            PilotSampleStarts(n_pilot=20, cluster_radius=1.5),
            IncrementalStarts(max_rounds=15, patience=3, merge_radius=0.5),
            ScoreGridStarts(grid_size=50, t_scan=300, scan_ascent_steps=30),
        ]
        strategies.extend(get_v2_strategies())
    else:
        strategies = []

    strategy_names = [s.name for s in strategies]
    extra_methods = ["v3f1"] if include_v3f1 else []
    baseline_names = ["b0", "b10"]
    all_method_names = strategy_names + extra_methods + baseline_names

    METRICS = ["recall", "precision", "soft_error", "nfe", "n_starts"]
    all_results = {m: {x: [] for x in METRICS} for m in all_method_names}

    logger.info("=" * 70)
    logger.info("AUTO STARTS: dim=%d, K=%s, v3f1=%s", cfg.dim, k_values, include_v3f1)
    logger.info("Стратегии: %s", strategy_names + extra_methods)
    logger.info("=" * 70)

    for K in k_values:
        logger.info("━━━ K=%d ━━━", K)
        k_res = {m: {x: [] for x in METRICS} for m in all_method_names}

        for dist_seed in range(n_dist_seeds):
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
                logger.info("  K=%d ds=%d is=%d", K, dist_seed, init_seed)

                for strategy in strategies:
                    try:
                        r = run_strategy(strategy, model, true_modes,
                                         mf_kwargs, cfg.metrics.eps_hit, seed=init_seed)
                        for x in METRICS:
                            k_res[strategy.name][x].append(r.get(x, 0.0))
                    except Exception as e:
                        logger.error("  Ошибка %s: %s", strategy.name, e)

                if include_v3f1:
                    try:
                        r = run_v3f1(model, true_modes, cfg, seed=init_seed)
                        for x in METRICS:
                            k_res["v3f1"][x].append(r.get(x, 0.0))
                    except Exception as e:
                        logger.error("  Ошибка v3f1: %s", e)

                for label, refine, offset in [("b0", 0, 0), ("b10", 10, 1000)]:
                    torch.manual_seed(init_seed + offset)
                    np.random.seed(init_seed + offset)
                    r = run_baseline(model, true_modes, cfg,
                                     n_samples=cfg.baseline.n_samples, refine_steps=refine)
                    for x in METRICS:
                        k_res[label][x].append(r.get(x, 0.0))

        for m in all_method_names:
            for x in METRICS:
                all_results[m][x].append(k_res[m][x])

    summary = {}
    for m in all_method_names:
        summary[m] = {}
        for x in METRICS:
            means, stds = [], []
            for k_list in all_results[m][x]:
                arr = np.array(k_list) if k_list else np.array([0.0])
                means.append(float(arr.mean())); stds.append(float(arr.std()))
            summary[m][x] = (np.array(means), np.array(stds))

    _print_tables(summary, k_values, all_method_names, dim_label)

    out = get_results_dir(output_dir, method, "auto_starts")
    _plot_all(summary, k_values, all_method_names, dim_label, out)

    json_data = {}
    for m in all_method_names:
        json_data[m] = {}
        for x in METRICS:
            json_data[m][x] = {
                "mean": summary[m][x][0].tolist(),
                "std":  summary[m][x][1].tolist(),
            }
    json_data["k_values"] = k_values
    json_data["config"] = {"dim": cfg.dim, "n_dist_seeds": n_dist_seeds,
                           "n_init_seeds": n_init_seeds, "include_v3f1": include_v3f1}

    p = out / f"auto_starts_{dim_label}.json"
    with open(p, "w") as f:
        json.dump(json_data, f, indent=2, default=float)
    logger.info("JSON: %s", p)
    return summary


STYLE = {
    "fixed_1":          {"color": "#BDBDBD", "marker": "x",  "ls": "--", "label": "Fixed 1"},
    "fixed_3":          {"color": "#9E9E9E", "marker": "+",  "ls": "--", "label": "Fixed 3"},
    "fixed_5":          {"color": "#757575", "marker": "1",  "ls": "--", "label": "Fixed 5"},
    "fixed_10":         {"color": "#424242", "marker": "2",  "ls": "--", "label": "Fixed 10"},
    "pilot_20":         {"color": "#2196F3", "marker": "o",  "ls": "-",  "label": "Pilot (n=20)"},
    "incremental_p3":   {"color": "#E91E63", "marker": "s",  "ls": "-",  "label": "Incremental"},
    "score_grid_50":    {"color": "#4CAF50", "marker": "^",  "ls": "-",  "label": "Score Grid"},
    "v2_smart3_H":      {"color": "#00BCD4", "marker": "p",  "ls": "-",  "label": "V2 Smart3+H"},
    "v2_smart5_H":      {"color": "#009688", "marker": "h",  "ls": "-",  "label": "V2 Smart5+H"},
    "v2_smart10_H":     {"color": "#006064", "marker": "H",  "ls": "-",  "label": "V2 Smart10+H"},
    "c2f_100":          {"color": "#F44336", "marker": "*",  "ls": "-",  "label": "C2F (100)"},
    "c2f_200":          {"color": "#B71C1C", "marker": "P",  "ls": "-",  "label": "C2F (200)"},
    "v3f1":             {"color": "#9C27B0", "marker": "D",  "ls": "-",  "label": "v3 F1"},
    "b0":               {"color": "#FF9800", "marker": "D",  "ls": ":",  "label": "Baseline b0"},
    "b10":              {"color": "#FF5722", "marker": "v",  "ls": ":",  "label": "Baseline b10"},
}


def _sty(method):
    return STYLE.get(method, {"color": "#000000", "marker": ".", "ls": "-", "label": method})


def _print_tables(summary, k_values, methods, dim_label):
    print(f"\n{'='*100}\n  АВТОВЫБОР n_starts ({dim_label})\n{'='*100}")
    for title, metric in [("RECALL", "recall"), ("NFE", "nfe"), ("N_STARTS", "n_starts")]:
        print(f"\n  {title}:")
        hdr = f"  {'K':>3}"
        for m in methods:
            hdr += f" | {_sty(m)['label']:>16}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for i, k in enumerate(k_values):
            row = f"  {k:>3}"
            for m in methods:
                mn, sd = summary[m][metric][0][i], summary[m][metric][1][i]
                if metric == "nfe":
                    row += f" | {mn:>16.0f}"
                else:
                    row += f" | {mn:>7.3f}±{sd:>5.3f}  "
            print(row)


def _plot_metric(ax, summary, k_values, methods, metric, ylabel):
    for m in methods:
        if m not in summary:
            continue
        s = _sty(m)
        mean, std = summary[m][metric]
        ax.plot(k_values, mean, color=s["color"], marker=s["marker"],
                linestyle=s["ls"], label=s["label"], markersize=5, linewidth=1.5)
        ax.fill_between(k_values, mean - std, mean + std, color=s["color"], alpha=0.1)
    ax.set_xlabel("Число мод K")
    ax.set_ylabel(ylabel)
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)


def _plot_all(summary, k_values, methods, dim_label, out_dir):
    plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 11})

    adaptive = [m for m in methods if m in ("pilot_20", "incremental_p3", "score_grid_50")]
    v2_m = [m for m in methods if m.startswith("v2_") or m.startswith("c2f_")]
    v3_m = [m for m in methods if m == "v3f1"]

    for fname, title, subset, ylim in [
        ("auto_recall_all",      f"Recall vs K ({dim_label}) — все",     methods,                    (-0.05, 1.15)),
        ("auto_recall_adaptive", f"Recall vs K ({dim_label}) — адаптивные",
         adaptive + v2_m + v3_m + ["fixed_3", "b10"], (-0.05, 1.15)),
        ("auto_soft_error",      f"Soft Error vs K ({dim_label})",       methods,                    None),
        ("auto_nfe",             f"NFE vs K ({dim_label})",               methods,                    None),
    ]:
        fig, ax = plt.subplots()
        metric = "recall" if "recall" in fname else ("soft_error" if "soft" in fname else "nfe")
        ylabel = {"recall": "Recall", "soft_error": "Soft Error", "nfe": "NFE"}[metric]
        _plot_metric(ax, summary, k_values, subset, metric, ylabel)
        if metric == "nfe":
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(ylim)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8, ncol=2)
        plt.tight_layout()
        fig.savefig(out_dir / f"{fname}_{dim_label}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(k_values, k_values, "r--", linewidth=1, alpha=0.5, label="Оракул (K)")
    _plot_metric(ax, summary, k_values, adaptive + v2_m + v3_m, "n_starts", "n_starts выбрано")
    ax.set_title(f"Авто n_starts vs K ({dim_label})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f"auto_nstarts_{dim_label}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    for m in methods:
        if m not in summary:
            continue
        s = _sty(m)
        r_mn = summary[m]["recall"][0].mean()
        r_sd = summary[m]["recall"][1].mean()
        nfe = summary[m]["nfe"][0].mean()
        ax.errorbar(nfe, r_mn, yerr=r_sd, color=s["color"], marker=s["marker"],
                    markersize=8, capsize=4, label=s["label"])
    ax.set_xscale("log")
    ax.set_xlabel("NFE (среднее по K)")
    ax.set_ylabel("Recall (среднее по K)")
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Эффективность: Recall vs NFE ({dim_label})")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_dir / f"auto_efficiency_{dim_label}.png", dpi=150)
    plt.close(fig)

    logger.info("Графики: %s", out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Сравнение стратегий автовыбора n_starts"
    )
    parser.add_argument("--config", type=str, default="configs/presets/dim1.yaml",
                        help="YAML-конфигурация (dim1.yaml или dim2.yaml)")
    parser.add_argument(
        "--method", type=str, default="v1v2",
        choices=["v1v2", "v3f1", "v3f1all"],
        help=(
            "v1v2    — стратегии v1 и v2 (без v3f1)\n"
            "v3f1    — только v3f1 как метод сравнения + b0/b10\n"
            "v3f1all — стратегии v1/v2 + v3f1 + b0/b10"
        ),
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=7)
    parser.add_argument("--n-dist-seeds", type=int, default=5)
    parser.add_argument("--n-init-seeds", type=int, default=3)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--output-dir",     type=str, default=None)
    parser.add_argument("--dim", type=int, default=None,
                        help="Переопределить dim из конфига (любое целое)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging)

    if args.dim is not None:
        cfg.dim = args.dim
    ckpt_dir = args.checkpoint_dir or cfg.checkpoint.checkpoint_dir
    out_dir  = args.output_dir     or cfg.outputs.output_dir

    include_v3f1 = args.method in ("v3f1", "v3f1all")

    k_values = list(range(args.k_min, args.k_max + 1))

    t0 = time.perf_counter()
    run_experiment(
        cfg=cfg,
        k_values=k_values,
        n_dist_seeds=args.n_dist_seeds,
        n_init_seeds=args.n_init_seeds,
        include_v3f1=include_v3f1,
        checkpoint_dir=ckpt_dir,
        output_dir=out_dir,
        method=args.method,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Завершено за %.1f сек (%.1f мин)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
