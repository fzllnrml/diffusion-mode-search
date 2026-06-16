from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.config import (
    load_config, setup_logging, ExperimentConfig,
    resolve_checkpoint, get_results_dir,
)
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

logger = logging.getLogger(__name__)


class ExperimentLogger:
    def __init__(self, cfg: ExperimentConfig):
        self._wandb_run = None
        if cfg.logging.use_wandb:
            try:
                import wandb
                flat = {}
                for sname, sec in asdict(cfg).items():
                    if isinstance(sec, dict):
                        for k, v in sec.items():
                            flat[f"{sname}.{k}"] = v
                    else:
                        flat[sname] = sec
                self._wandb_run = wandb.init(
                    project=cfg.logging.wandb_project,
                    entity=cfg.logging.wandb_entity,
                    name=cfg.name, config=flat,
                )
                logger.info("W&B: %s", wandb.run.url)
            except Exception as e:
                logger.warning("W&B недоступен: %s", e)

    def log(self, data: dict, step: Optional[int] = None) -> None:
        logger.info("  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in data.items()
        ))
        if self._wandb_run:
            import wandb
            wandb.log(data, step=step)

    def finish(self):
        if self._wandb_run:
            import wandb
            wandb.finish()


def get_or_train_model(
    cfg: ExperimentConfig,
    mixture: GaussianMixture,
    device: torch.device,
) -> DiffusionModel:
    model = DiffusionModel(
        dim=cfg.dim, T=cfg.model.T,
        beta_start=cfg.model.beta_start, beta_end=cfg.model.beta_end,
        hidden_dims=cfg.model.hidden_dims, device=device,
    )
    ckpt_path = resolve_checkpoint(
        cfg.checkpoint.checkpoint_dir, dim=cfg.dim, tag=cfg.name,
        filename_override=cfg.checkpoint.filename_override,
    )
    if ckpt_path.exists():
        logger.info("Загружаем чекпоинт: %s", ckpt_path)
        model.load_checkpoint(str(ckpt_path))
        if model._train_step >= cfg.training.num_steps:
            return model
        remaining = cfg.training.num_steps - model._train_step
        logger.info("Доучиваем: %d шагов", remaining)
    else:
        remaining = cfg.training.num_steps

    model.train_on_data(
        sample_fn=lambda n: mixture.sample_torch(n, device),
        num_steps=remaining,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.learning_rate,
        lr_min=cfg.training.lr_min,
        scheduler=cfg.training.scheduler,
        log_every=cfg.training.log_every,
        save_every=cfg.training.save_every,
        save_path=str(ckpt_path),
    )
    model.save_checkpoint(str(ckpt_path))
    return model


def _run_v1(cfg, model, true_modes, exp_logger):
    mf = cfg.mode_finder_v1
    finder = ModeFinder(
        model=model,
        timesteps=mf.timesteps, step_size=mf.step_size,
        split_eps=mf.split_eps, split_threshold=mf.split_threshold,
        merge_radius=mf.merge_radius, ascent_steps=mf.ascent_steps,
        refine_steps=mf.refine_steps, refine_step_scale=mf.refine_step_scale,
        n_starts=mf.n_starts, starts_min_sep=mf.starts_min_sep,
        split_directions=mf.split_directions,
        x_min=cfg.distribution.x_min, x_max=cfg.distribution.x_max,
    )
    t0 = time.perf_counter()
    result = finder.find_modes(seed=cfg.seed)
    elapsed = time.perf_counter() - t0
    metrics = evaluate_modes(true_modes, result.modes, cfg.metrics.eps_hit)
    exp_logger.log({"v1/recall": metrics.recall, "v1/f1": metrics.f1,
                    "v1/nfe": result.nfe, "v1/time": elapsed})
    return {"metrics": asdict(metrics), "nfe": result.nfe, "time": elapsed}


def _run_v2(cfg, model, true_modes, exp_logger):
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
        x_min=cfg.distribution.x_min, x_max=cfg.distribution.x_max,
    )
    t0 = time.perf_counter()
    result = finder.find_modes(seed=cfg.seed)
    elapsed = time.perf_counter() - t0
    metrics = evaluate_modes(true_modes, result.modes, cfg.metrics.eps_hit)
    exp_logger.log({"v2/recall": metrics.recall, "v2/nfe": result.nfe,
                    "v2/nfe_starts": result.nfe_starts, "v2/time": elapsed})
    return {"metrics": asdict(metrics), "nfe": result.nfe,
            "nfe_starts": result.nfe_starts, "time": elapsed}


def _run_v3f1(cfg, model, true_modes, exp_logger):
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
        starts_min_sep=mf.starts_min_sep, max_active_per_start=mf.max_active_per_start,
        x_min=cfg.distribution.x_min, x_max=cfg.distribution.x_max,
    )
    t0 = time.perf_counter()
    result = finder.find_modes(seed=cfg.seed)
    elapsed = time.perf_counter() - t0
    metrics = evaluate_modes(true_modes, result.modes, cfg.metrics.eps_hit)
    exp_logger.log({"v3f1/recall": metrics.recall, "v3f1/nfe": result.nfe})
    return {"metrics": asdict(metrics), "nfe": result.nfe, "time": elapsed}


def _run_v3f2(cfg, model, true_modes, exp_logger):
    mf = cfg.mode_finder_v3_f2
    finder = ModeFinderV3F2(
        model=model,
        timesteps=mf.timesteps, step_size=mf.step_size,
        n_particles=mf.n_particles, merge_factor=mf.merge_factor,
        merge_radius_min=mf.merge_radius_min, ascent_steps=mf.ascent_steps,
        normalize_score=mf.normalize_score, refine_steps=mf.refine_steps,
        refine_step_scale=mf.refine_step_scale,
        ddim_steps=mf.ddim_steps, init_stop_t=mf.init_stop_t,
        x_min=cfg.distribution.x_min, x_max=cfg.distribution.x_max,
    )
    t0 = time.perf_counter()
    result = finder.find_modes(seed=cfg.seed)
    elapsed = time.perf_counter() - t0
    metrics = evaluate_modes(true_modes, result.modes, cfg.metrics.eps_hit)
    exp_logger.log({"v3f2/recall": metrics.recall, "v3f2/nfe": result.nfe})
    return {"metrics": asdict(metrics), "nfe": result.nfe, "time": elapsed}


def _run_v3f3(cfg, model, true_modes, exp_logger):
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
        ddim_steps=mf.ddim_steps,
        x_min=cfg.distribution.x_min, x_max=cfg.distribution.x_max,
    )
    t0 = time.perf_counter()
    result = finder.find_modes(seed=cfg.seed)
    elapsed = time.perf_counter() - t0
    metrics = evaluate_modes(true_modes, result.modes, cfg.metrics.eps_hit)
    exp_logger.log({"v3f3/recall": metrics.recall, "v3f3/nfe": result.nfe})
    return {"metrics": asdict(metrics), "nfe": result.nfe, "time": elapsed}


def _run_baseline(cfg, model, true_modes, exp_logger):
    bl = cfg.baseline
    finder = BaselineModeFinder(
        model=model, n_samples=bl.n_samples, refine_steps=bl.refine_steps,
        refine_alpha=bl.refine_alpha, merge_radius=bl.merge_radius,
    )
    t0 = time.perf_counter()
    result = finder.find_modes()
    elapsed = time.perf_counter() - t0
    metrics = evaluate_modes(true_modes, result.modes, cfg.metrics.eps_hit)
    label = "b0" if bl.refine_steps == 0 else "b10"
    exp_logger.log({f"{label}/recall": metrics.recall, f"{label}/nfe": result.nfe})
    return {"metrics": asdict(metrics), "nfe": result.nfe, "time": elapsed}


def run_experiment(cfg: ExperimentConfig, method: str = "v1") -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    exp_logger = ExperimentLogger(cfg)

    mixture = GaussianMixture(
        mus=cfg.distribution.mus,
        sigmas=cfg.distribution.sigmas,
        weights=cfg.distribution.weights,
    )
    true_modes = mixture.mode_locations
    logger.info("K=%d мод, dim=%d", mixture.K, mixture.dim)

    device = resolve_device(cfg.device)
    model = get_or_train_model(cfg, mixture, device)

    results = {}
    for key, flag, runner in [
        ("v1",   method in ("v1",   "both"),        _run_v1),
        ("v2",   method in ("v2",   "both"),        _run_v2),
        ("v3f1", method in ("v3f1", "v3"),          _run_v3f1),
        ("v3f2", method in ("v3f2", "v3"),          _run_v3f2),
        ("v3f3", method in ("v3f3", "v3"),          _run_v3f3),
    ]:
        if flag:
            logger.info("Запуск %s...", key)
            results[key] = runner(cfg, model, true_modes, exp_logger)

    logger.info("Запуск baseline...")
    results["baseline"] = _run_baseline(cfg, model, true_modes, exp_logger)

    logger.info("=" * 60)
    logger.info("СВОДКА: %s (dim=%d, K=%d)", cfg.name, cfg.dim, mixture.K)
    for key, r in results.items():
        m = r["metrics"]
        logger.info(
            "%-10s recall=%.2f precision=%.2f F1=%.2f soft_err=%.3f NFE=%d time=%.1fs",
            key + ":", m["recall"], m["precision"], m["f1"],
            m["soft_error"], r["nfe"], r["time"],
        )
    logger.info("=" * 60)

    for key, r in results.items():
        mkey = key if key != "baseline" else "baseline"
        out_dir = get_results_dir(cfg.outputs.output_dir, mkey, "single_run")
        save_path = out_dir / f"{cfg.name}.json"
        with open(save_path, "w") as f:
            json.dump(r, f, indent=2, default=float)
        logger.debug("Сохранено: %s", save_path)

    exp_logger.finish()
    return results


def main():
    parser = argparse.ArgumentParser(description="Одиночный запуск поиска мод")
    parser.add_argument("--config", type=str, default="configs/presets/dim1.yaml")
    parser.add_argument(
        "--method", type=str, default="v1",
        choices=["v1", "v2", "v3f1", "v3f2", "v3f3", "v3", "both"],
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--output-dir",     type=str, default=None)
    parser.add_argument("--dim", type=int, default=None,
                        help="Переопределить dim из конфига (любое целое)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging)

    if args.dim is not None:
        cfg.dim = args.dim
    if args.checkpoint_dir:
        cfg.checkpoint.checkpoint_dir = args.checkpoint_dir
    if args.output_dir:
        cfg.outputs.output_dir = args.output_dir

    run_experiment(cfg, method=args.method)


if __name__ == "__main__":
    main()
