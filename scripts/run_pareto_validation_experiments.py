#!/usr/bin/env python3
"""Run Pareto validation experiments for synthetic diffusion-mode models."""

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
import os
import platform
import re
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import logsumexp
from scipy.spatial.distance import cdist




THIS = Path(__file__).resolve()
CANDIDATE_ROOTS = [THIS.parent.parent, THIS.parent, Path.cwd()]
PROJECT_ROOT = None
for _root in CANDIDATE_ROOTS:
    if (_root / "src").is_dir() and (_root / "configs").is_dir():
        PROJECT_ROOT = _root.resolve()
        break
if PROJECT_ROOT is None:
    raise RuntimeError(
        "Project root with src/ and configs/ was not found. "
        "Project root containing src/ and configs/ was not found."
    )
sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.clustering import agglomerative_merge, merge_close  
from src.algorithms.mode_finder_v2 import ddim_sample  
from src.algorithms.mode_finder_v3_f2 import ModeFinderV3F2  
from src.config import load_config  
from src.models.diffusion import DiffusionModel  
from src.utils.device import resolve_device  
from src.utils.distribution import GaussianMixture  

LOGGER = logging.getLogger("pareto_validation")
SCRIPT_VERSION = "1.0.0"

DEFAULT_CONFIGS = {
    10: "configs/presets/dim10.yaml",
    30: "configs/presets/dim30.yaml",
    50: "configs/presets/dim50.yaml",
}
DEFAULT_EPS = {
    10: [1.0, 1.5, 2.0, 2.5, 3.0],
    30: [1.0, 2.0, 3.0, 3.5, 4.0],
    50: [1.0, 2.0, 3.0, 4.5, 5.0],
}
DEFAULT_SCORE_TIMESTEPS = [800, 600, 400, 200, 100, 50, 20, 5, 0]
ALL_EXPERIMENTS = {"score_alignment", "search_comparison", "gmm_mode_audit"}





def utc_now() :
    return datetime.now(timezone.utc).isoformat()


def json_default(obj) :
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def atomic_write_json(path, payload) :
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default)
    tmp.replace(path)


def append_jsonl(path, payload) :
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_json(path) :
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_int_list(text) :
    result = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            step = 1 if end >= start else -1
            result.extend(range(start, end + step, step))
        else:
            result.append(int(part))
    return sorted(set(result))


def parse_str_set(text) :
    return {x.strip() for x in text.split(",") if x.strip()}


def as_2d(points, dim = None) :
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, dim or 0), dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        if dim == 1:
            arr = arr.reshape(-1, 1)
        else:
            arr = arr.reshape(1, -1)
    return arr


def file_sha256(path, block_size = 1024 * 1024) :
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def setup_logging(output_dir, level) :
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    return log_path


def cleanup_device(device) :
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass





def checkpoint_candidates(checkpoint_dir, dim, tag) :
    return [
        checkpoint_dir / f"dim_{dim}" / f"model_{tag}.pth",
        checkpoint_dir / f"model_{tag}.pth",
    ]


def find_checkpoint_load_only(checkpoint_dir, dim, K, dseed) :
    tag = f"dim{dim}_K{K}_dseed{dseed}"
    candidates = checkpoint_candidates(checkpoint_dir, dim, tag)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Checkpoint is missing. Training is disabled. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def torch_load_compat(path, map_location) :
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model_only(cfg, checkpoint_path, device) :
    checkpoint = torch_load_compat(checkpoint_path, map_location="cpu")
    ckpt_dim = int(checkpoint.get("dim", cfg.dim))
    ckpt_T = int(checkpoint.get("T", cfg.model.T))
    if ckpt_dim != int(cfg.dim):
        raise ValueError(f"Checkpoint dim={ckpt_dim}, config dim={cfg.dim}: {checkpoint_path}")
    if ckpt_T != int(cfg.model.T):
        raise ValueError(f"Checkpoint T={ckpt_T}, config T={cfg.model.T}: {checkpoint_path}")

    hidden_dims = checkpoint.get("config", {}).get("hidden_dims", cfg.model.hidden_dims)
    model = DiffusionModel(
        dim=cfg.dim,
        T=cfg.model.T,
        beta_start=cfg.model.beta_start,
        beta_end=cfg.model.beta_end,
        hidden_dims=hidden_dims,
        activation=cfg.model.activation,
        device=device,
    )
    model.net.load_state_dict(checkpoint["state_dict"])
    model._train_step = int(checkpoint.get("train_step", 0))
    model._last_loss = float(checkpoint.get("last_loss", float("nan")))
    model.net.eval()
    return model, checkpoint





def analytic_noisy_mixture_score(
    mixture,
    alpha_bar,
    x,
) :
    """Exact score of the noised isotropic GMM q_t(x)."""
    x2 = as_2d(x, mixture.dim)
    mus = np.asarray(mixture.mus, dtype=np.float64)
    sigmas = np.asarray(mixture.sigmas, dtype=np.float64)
    weights = np.asarray(mixture.weights, dtype=np.float64)

    means_t = math.sqrt(max(alpha_bar, 0.0)) * mus
    vars_t = alpha_bar * sigmas**2 + (1.0 - alpha_bar)
    diff = x2[:, None, :] - means_t[None, :, :]
    sq = np.sum(diff * diff, axis=2)
    logp = (
        np.log(weights[None, :] + 1e-300)
        - 0.5 * mixture.dim * np.log(2.0 * np.pi * vars_t[None, :])
        - 0.5 * sq / vars_t[None, :]
    )
    responsibilities = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
    component_scores = (means_t[None, :, :] - x2[:, None, :]) / vars_t[None, :, None]
    return np.sum(responsibilities[:, :, None] * component_scores, axis=1)


class AnalyticNoisyGMMScoreModel:
    """Minimal model interface required by ModeFinderV3F2."""

    def __init__(self, mixture, learned_model):
        self.mixture = mixture
        self.dim = mixture.dim
        self.device = learned_model.device
        self.schedule = learned_model.schedule
        self._nfe = 0

    def enable_nfe_counting(self) :
        self._nfe = 0

    def reset_nfe(self) :
        self._nfe = 0

    @property
    def nfe(self) :
        return int(self._nfe)

    def score_numpy(self, x, t) :
        arr = np.asarray(x, dtype=np.float64)
        x2 = as_2d(arr, self.dim)
        alpha_bar = float(self.schedule.alphas_cumprod[int(t)].detach().cpu().item())
        score = analytic_noisy_mixture_score(self.mixture, alpha_bar, x2)
        self._nfe += int(x2.shape[0])
        if arr.ndim == 1:
            return score[0]
        return score





def original_metrics(true_points, candidates, eps) :
    true2 = as_2d(true_points)
    cand2 = as_2d(candidates, true2.shape[1])
    K, M = len(true2), len(cand2)
    if M == 0:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "soft_error": float("inf"),
            "n_true": K,
            "n_found": 0,
            "n_hits": 0,
            "n_correct": 0,
            "per_true_min_distance": [float("inf")] * K,
        }
    dist = cdist(true2, cand2)
    per_true = dist.min(axis=1)
    per_candidate = dist.min(axis=0)
    hits = per_true <= eps
    correct = per_candidate <= eps
    recall = float(np.mean(hits)) if K else 0.0
    precision = float(np.mean(correct)) if M else 0.0
    f1 = 2.0 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "f1": float(f1),
        "soft_error": float(np.mean(per_true)) if K else float("nan"),
        "n_true": K,
        "n_found": M,
        "n_hits": int(np.sum(hits)),
        "n_correct": int(np.sum(correct)),
        "per_true_min_distance": per_true.astype(float).tolist(),
    }


def one_to_one_metrics(true_points, candidates, eps) :
    true2 = as_2d(true_points)
    cand2 = as_2d(candidates, true2.shape[1])
    K, M = len(true2), len(cand2)
    if K == 0 or M == 0:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "matched": 0,
            "n_true": K,
            "n_found": M,
            "mean_matched_distance": None,
            "max_matched_distance": None,
            "assignments": [],
        }

    dist = cdist(true2, cand2)
    
    
    penalty = max(1e6, float(np.nanmax(dist) + 1.0) * (K + M + 1))
    cost = np.where(dist <= eps, dist, penalty)
    rows, cols = linear_sum_assignment(cost)
    valid = dist[rows, cols] <= eps
    matched_rows = rows[valid]
    matched_cols = cols[valid]
    matched_dist = dist[matched_rows, matched_cols]
    matched = int(valid.sum())
    recall = matched / K
    precision = matched / M
    f1 = 2.0 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
    assignments = [
        {
            "true_index": int(r),
            "candidate_index": int(c),
            "distance": float(d),
        }
        for r, c, d in zip(matched_rows, matched_cols, matched_dist)
    ]
    return {
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "matched": matched,
        "n_true": K,
        "n_found": M,
        "mean_matched_distance": float(np.mean(matched_dist)) if matched else None,
        "max_matched_distance": float(np.max(matched_dist)) if matched else None,
        "assignments": assignments,
    }


def evaluate_candidate_set(
    true_points,
    candidates,
    eps_values,
) :
    true2 = as_2d(true_points)
    cand2 = as_2d(candidates, true2.shape[1])
    return {
        "n_found": int(len(cand2)),
        "candidate_to_true_ratio": float(len(cand2) / len(true2)) if len(true2) else None,
        "metrics_by_eps": {
            str(float(eps)): {
                "coverage": original_metrics(true2, cand2, float(eps)),
                "one_to_one": one_to_one_metrics(true2, cand2, float(eps)),
            }
            for eps in eps_values
        },
    }





def cluster_points(points, radius, dim) :
    pts = as_2d(points, dim)
    if len(pts) == 0:
        return pts
    if dim == 1:
        return merge_close(pts.flatten(), radius).reshape(-1, 1)
    return agglomerative_merge(pts, radius)


def build_v3f2(model, cfg, overrides) :
    mf = cfg.mode_finder_v3_f2
    return ModeFinderV3F2(
        model=model,
        timesteps=mf.timesteps,
        step_size=mf.step_size,
        n_particles=overrides.get("n_particles") or mf.n_particles,
        merge_factor=mf.merge_factor,
        merge_radius_min=mf.merge_radius_min,
        ascent_steps=overrides.get("ascent_steps") or mf.ascent_steps,
        normalize_score=mf.normalize_score,
        refine_steps=(
            mf.refine_steps
            if overrides.get("refine_steps") is None
            else int(overrides["refine_steps"])
        ),
        refine_step_scale=mf.refine_step_scale,
        ddim_steps=overrides.get("ddim_steps") or mf.ddim_steps,
        init_stop_t=mf.init_stop_t,
        x_min=cfg.distribution.x_min,
        x_max=cfg.distribution.x_max,
    )


def effective_ddim_transitions(model, num_steps, stop_t = 0) :
    indices = np.linspace(model.schedule.T - 1, max(stop_t, 0), num_steps + 1, dtype=int)
    indices = np.unique(indices)[::-1]
    return max(0, len(indices) - 1)


def generate_ddim(
    model,
    n,
    ddim_steps,
    seed,
    stop_t = 0,
) :
    model.enable_nfe_counting()
    model.reset_nfe()
    samples = ddim_sample(model, n=n, num_steps=ddim_steps, stop_t=stop_t, seed=seed)
    samples2 = as_2d(samples, model.dim)
    return samples2, int(model.nfe)





def sample_true_xt(
    mixture,
    alpha_bar,
    n,
    rng,
) :
    x0 = as_2d(mixture.sample_numpy(n, rng=rng), mixture.dim)
    noise = rng.standard_normal(size=x0.shape)
    return math.sqrt(max(alpha_bar, 0.0)) * x0 + math.sqrt(max(1.0 - alpha_bar, 0.0)) * noise


def distribution_stats(values) :
    x = np.asarray(values, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if len(finite) == 0:
        return {k: float("nan") for k in ["mean", "std", "min", "q10", "q25", "median", "q75", "q90", "max"]}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "q10": float(np.quantile(finite, 0.10)),
        "q25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "q75": float(np.quantile(finite, 0.75)),
        "q90": float(np.quantile(finite, 0.90)),
        "max": float(np.max(finite)),
    }


def run_score_alignment(
    model,
    mixture,
    timesteps,
    n,
    seed,
    raw_npz_path,
) :
    rng = np.random.default_rng(seed)
    result = {"n": int(n), "seed": int(seed), "timesteps": {}}
    raw = {}

    for t in timesteps:
        t_int = int(t)
        if t_int < 0 or t_int >= model.schedule.T:
            LOGGER.warning("Score alignment: timestep %d is outside [0,%d), skipped", t_int, model.schedule.T)
            continue
        alpha_bar = float(model.schedule.alphas_cumprod[t_int].detach().cpu().item())
        x = sample_true_xt(mixture, alpha_bar, n=n, rng=rng)
        true_score = analytic_noisy_mixture_score(mixture, alpha_bar, x)
        with torch.no_grad():
            learned_t = model.score(torch.tensor(x, dtype=torch.float32, device=model.device), t_int)
        learned_score = learned_t.detach().cpu().numpy().astype(np.float64)

        true_norm = np.linalg.norm(true_score, axis=1)
        learned_norm = np.linalg.norm(learned_score, axis=1)
        dot = np.sum(true_score * learned_score, axis=1)
        cosine = dot / np.maximum(true_norm * learned_norm, 1e-12)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cosine))
        abs_error = np.linalg.norm(learned_score - true_score, axis=1)
        rel_error = abs_error / np.maximum(true_norm, 1e-12)
        norm_ratio = learned_norm / np.maximum(true_norm, 1e-12)

        result["timesteps"][str(t_int)] = {
            "alpha_bar": alpha_bar,
            "cosine": distribution_stats(cosine),
            "angle_deg": distribution_stats(angle_deg),
            "relative_l2_error": distribution_stats(rel_error),
            "absolute_l2_error": distribution_stats(abs_error),
            "true_score_norm": distribution_stats(true_norm),
            "learned_score_norm": distribution_stats(learned_norm),
            "norm_ratio": distribution_stats(norm_ratio),
            "fraction_cosine_positive": float(np.mean(cosine > 0.0)),
            "fraction_cosine_above_0_5": float(np.mean(cosine >= 0.5)),
            "fraction_cosine_above_0_9": float(np.mean(cosine >= 0.9)),
        }
        if raw_npz_path is not None:
            prefix = f"t{t_int}"
            raw[f"{prefix}_x"] = x.astype(np.float32)
            raw[f"{prefix}_true_score"] = true_score.astype(np.float32)
            raw[f"{prefix}_learned_score"] = learned_score.astype(np.float32)
            raw[f"{prefix}_cosine"] = cosine.astype(np.float32)
            raw[f"{prefix}_relative_l2_error"] = rel_error.astype(np.float32)

    if raw_npz_path is not None:
        raw_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw_npz_path, **raw)
        result["raw_npz"] = str(raw_npz_path)
    return result





def analytic_log_density(mixture, x) :
    point = np.asarray(x, dtype=np.float64).reshape(1, -1)
    diff = point[:, None, :] - mixture.mus[None, :, :]
    sq = np.sum(diff * diff, axis=2)[0]
    vars_ = mixture.sigmas**2
    log_terms = (
        np.log(mixture.weights + 1e-300)
        - 0.5 * mixture.dim * np.log(2.0 * np.pi * vars_)
        - 0.5 * sq / vars_
    )
    return float(logsumexp(log_terms))


def analytic_gmm_hessian_log_density(mixture, x) :
    point = np.asarray(x, dtype=np.float64).reshape(1, -1)
    diff = mixture.mus - point[0]
    vars_ = mixture.sigmas**2
    sq = np.sum((point[:, None, :] - mixture.mus[None, :, :]) ** 2, axis=2)[0]
    log_terms = (
        np.log(mixture.weights + 1e-300)
        - 0.5 * mixture.dim * np.log(2.0 * np.pi * vars_)
        - 0.5 * sq / vars_
    )
    resp = np.exp(log_terms - logsumexp(log_terms))
    component_scores = diff / vars_[:, None]
    score = np.sum(resp[:, None] * component_scores, axis=0)
    hessian = np.zeros((mixture.dim, mixture.dim), dtype=np.float64)
    eye = np.eye(mixture.dim)
    for k in range(mixture.K):
        a = component_scores[k]
        hessian += resp[k] * (-eye / vars_[k] + np.outer(a, a))
    hessian -= np.outer(score, score)
    return 0.5 * (hessian + hessian.T)


def deduplicate_points(points, radius) :
    unique = []
    for p in points:
        p = np.asarray(p, dtype=np.float64)
        if not unique or min(np.linalg.norm(p - q) for q in unique) > radius:
            unique.append(p)
    return unique


def run_gmm_mode_audit(
    mixture,
    bounds,
    n_random_starts,
    seed,
    grad_tol,
    dedup_radius,
    maxiter,
) :
    rng = np.random.default_rng(seed)
    starts = [m.copy() for m in mixture.mus]
    if n_random_starts > 0:
        starts.extend(as_2d(mixture.sample_numpy(n_random_starts, rng=rng), mixture.dim))

    opt_results = []
    stationary = []
    for idx, start in enumerate(starts):
        fun = lambda z: -analytic_log_density(mixture, z)
        jac = lambda z: -np.asarray(mixture.score_numpy(np.asarray(z, dtype=np.float64)), dtype=np.float64).reshape(-1)
        res = minimize(
            fun,
            np.asarray(start, dtype=np.float64),
            jac=jac,
            method="L-BFGS-B",
            bounds=[bounds] * mixture.dim,
            options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": grad_tol},
        )
        x = np.asarray(res.x, dtype=np.float64)
        score_norm = float(np.linalg.norm(mixture.score_numpy(x)))
        accepted = bool(np.isfinite(res.fun) and score_norm <= max(grad_tol * 10.0, 1e-5))
        opt_results.append(
            {
                "start_index": idx,
                "success": bool(res.success),
                "accepted_stationary": accepted,
                "status": int(res.status),
                "message": str(res.message),
                "iterations": int(getattr(res, "nit", -1)),
                "score_norm": score_norm,
                "negative_log_density": float(res.fun),
            }
        )
        if accepted:
            stationary.append(x)

    stationary_unique = deduplicate_points(stationary, dedup_radius)
    local_modes = []
    stationary_info = []
    for p in stationary_unique:
        H = analytic_gmm_hessian_log_density(mixture, p)
        eig = np.linalg.eigvalsh(H)
        is_mode = bool(np.max(eig) < -1e-8)
        info = {
            "point": p.tolist(),
            "log_density": analytic_log_density(mixture, p),
            "score_norm": float(np.linalg.norm(mixture.score_numpy(p))),
            "hessian_eigenvalue_min": float(np.min(eig)),
            "hessian_eigenvalue_max": float(np.max(eig)),
            "is_strict_local_mode": is_mode,
        }
        stationary_info.append(info)
        if is_mode:
            local_modes.append(p)

    modes_arr = as_2d(local_modes, mixture.dim)
    centers = as_2d(mixture.mode_locations, mixture.dim)
    if len(modes_arr):
        center_to_mode = cdist(centers, modes_arr).min(axis=1)
        mode_to_center = cdist(modes_arr, centers).min(axis=1)
    else:
        center_to_mode = np.full(len(centers), np.inf)
        mode_to_center = np.empty(0)

    return {
        "warning": "Numerical audit only; it does not prove that all mathematical modes were found.",
        "n_starts": len(starts),
        "n_stationary_unique": len(stationary_unique),
        "n_strict_local_modes": len(local_modes),
        "strict_local_modes": modes_arr.tolist(),
        "stationary_points": stationary_info,
        "center_to_nearest_mode_distance": center_to_mode.astype(float).tolist(),
        "center_to_nearest_mode_mean": float(np.mean(center_to_mode)),
        "center_to_nearest_mode_max": float(np.max(center_to_mode)),
        "mode_to_nearest_center_distance": mode_to_center.astype(float).tolist(),
        "optimization_summary": {
            "successful": int(sum(r["success"] for r in opt_results)),
            "accepted_stationary": int(sum(r["accepted_stationary"] for r in opt_results)),
            "failed": int(sum(not r["success"] for r in opt_results)),
        },
        "optimization_runs": opt_results,
    }





def build_old_results_index(old_results_dir) :
    index = {}
    manifest = {"root": None, "files": [], "indexed_runs": 0}
    if old_results_dir is None:
        return index, manifest
    root = old_results_dir.resolve()
    manifest["root"] = str(root)
    if not root.exists():
        LOGGER.warning("Old results directory does not exist: %s", root)
        return index, manifest

    for path in sorted(root.rglob("*.json")):
        try:
            payload = read_json(path)
        except Exception as exc:
            manifest["files"].append({"path": str(path), "error": str(exc)})
            continue
        runs = payload.get("runs") if isinstance(payload, dict) else None
        cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
        if not isinstance(runs, list):
            continue
        dim = cfg.get("dim")
        count = 0
        for run in runs:
            try:
                key = (
                    int(dim),
                    str(run["method"]),
                    int(run["K"]),
                    int(run["dist_seed"]),
                    int(run["init_seed"]),
                )
            except Exception:
                continue
            index[key] = {"source": str(path), "run": run}
            count += 1
        manifest["files"].append({"path": str(path), "runs_indexed": count})

    
    
    for path in sorted(root.rglob("*.csv")):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception as exc:
            manifest["files"].append({"path": str(path), "error": str(exc)})
            continue
        required = {"method", "K", "dseed", "init_seed", "eps", "recall", "precision", "f1", "soft_error"}
        if not rows or not required.issubset(rows[0].keys()):
            continue
        dim_match = re.search(r"results[_ -]?(\d+)d", str(path), flags=re.IGNORECASE)
        if dim_match is None:
            dim_match = re.search(r"dim(\d+)", str(path), flags=re.IGNORECASE)
        if dim_match is None:
            continue
        dim = int(dim_match.group(1))
        grouped = {}
        for row in rows:
            try:
                key = (
                    dim,
                    str(row["method"]),
                    int(row["K"]),
                    int(row["dseed"]),
                    int(row["init_seed"]),
                )
                eps_key = str(float(row["eps"]))
            except Exception:
                continue
            run = grouped.setdefault(
                key,
                {
                    "method": key[1],
                    "K": key[2],
                    "dist_seed": key[3],
                    "init_seed": key[4],
                    "nfe": None,
                    "n_modes_returned": None,
                    "metrics_by_eps": {},
                },
            )
            run["metrics_by_eps"][eps_key] = {
                "recall": float(row["recall"]),
                "precision": float(row["precision"]),
                "f1": float(row["f1"]),
                "soft_error": float(row["soft_error"]),
                "n_found": int(float(row.get("n_found", 0) or 0)),
            }
            if row.get("nfe") not in (None, ""):
                run["nfe"] = int(float(row["nfe"]))
            if row.get("n_found") not in (None, ""):
                run["n_modes_returned"] = int(float(row["n_found"]))
        count = 0
        for key, run in grouped.items():
            if key not in index:
                index[key] = {"source": str(path), "run": run}
                count += 1
        if count:
            manifest["files"].append({"path": str(path), "runs_indexed": count, "format": "csv"})

    manifest["indexed_runs"] = len(index)
    return index, manifest


def old_result_comparison(
    old_index,
    dim,
    K,
    dseed,
    init_seed,
    new_evaluation,
) :
    old = old_index.get((dim, "v3f2", K, dseed, init_seed))
    if old is None:
        return None
    old_run = old["run"]
    deltas = {}
    for eps_key, old_metrics in old_run.get("metrics_by_eps", {}).items():
        new_eps = new_evaluation.get("metrics_by_eps", {}).get(str(float(eps_key)))
        if not new_eps:
            continue
        coverage = new_eps["coverage"]
        deltas[str(float(eps_key))] = {
            metric: float(coverage[metric] - old_metrics[metric])
            for metric in ["recall", "precision", "f1", "soft_error"]
            if metric in old_metrics
        }
    return {
        "source": old["source"],
        "old_nfe": old_run.get("nfe"),
        "new_nfe": None,
        "old_n_modes_returned": old_run.get("n_modes_returned"),
        "new_n_modes_returned": new_evaluation.get("n_found"),
        "coverage_metric_deltas_new_minus_old": deltas,
    }





def run_search_comparison(
    model,
    mixture,
    cfg,
    eps_values,
    init_seed,
    overrides,
    artifact_path,
    old_index,
    dim,
    K,
    dseed,
) :
    mf = cfg.mode_finder_v3_f2
    n_particles = int(overrides.get("n_particles") or mf.n_particles)
    ddim_steps = int(overrides.get("ddim_steps") or mf.ddim_steps)
    stop_t = int(mf.init_stop_t)
    true_centers = as_2d(mixture.mode_locations, mixture.dim)

    
    starts, init_nfe = generate_ddim(
        model=model,
        n=n_particles,
        ddim_steps=ddim_steps,
        seed=init_seed,
        stop_t=stop_t,
    )

    same_merge_v3 = cluster_points(starts, float(mf.merge_radius_min), mixture.dim)
    same_merge_baseline = cluster_points(starts, float(cfg.baseline.merge_radius), mixture.dim)

    
    learned_finder = build_v3f2(model, cfg, overrides)
    learned_t0 = time.perf_counter()
    learned_result = learned_finder.find_modes(starts=starts.copy(), seed=init_seed, verbose=False)
    learned_seconds = time.perf_counter() - learned_t0
    learned_modes = as_2d(learned_result.modes, mixture.dim)
    learned_search_nfe = int(learned_result.nfe)
    learned_total_nfe = int(init_nfe + learned_search_nfe)

    
    analytic_model = AnalyticNoisyGMMScoreModel(mixture, model)
    exact_finder = build_v3f2(analytic_model, cfg, overrides)
    exact_t0 = time.perf_counter()
    exact_result = exact_finder.find_modes(starts=starts.copy(), seed=init_seed, verbose=False)
    exact_seconds = time.perf_counter() - exact_t0
    exact_modes = as_2d(exact_result.modes, mixture.dim)
    exact_score_evals = int(exact_result.nfe)

    
    
    transitions = effective_ddim_transitions(model, ddim_steps, stop_t)
    if transitions <= 0:
        raise RuntimeError("DDIM has zero transitions; cannot build equal-NFE baseline")
    equal_nfe_n_samples = max(1, int(learned_total_nfe // transitions))
    equal_samples, equal_nfe_actual = generate_ddim(
        model=model,
        n=equal_nfe_n_samples,
        ddim_steps=ddim_steps,
        seed=init_seed,
        stop_t=stop_t,
    )
    equal_merge_v3 = cluster_points(equal_samples, float(mf.merge_radius_min), mixture.dim)
    equal_merge_baseline = cluster_points(equal_samples, float(cfg.baseline.merge_radius), mixture.dim)

    candidate_sets = {
        "ddim_same_particles_raw": starts,
        "ddim_same_particles_merge_v3f2": same_merge_v3,
        "ddim_same_particles_merge_baseline": same_merge_baseline,
        "v3f2_learned": learned_modes,
        "v3f2_exact": exact_modes,
        "ddim_equal_nfe_raw": equal_samples,
        "ddim_equal_nfe_merge_v3f2": equal_merge_v3,
        "ddim_equal_nfe_merge_baseline": equal_merge_baseline,
    }
    evaluations = {
        name: evaluate_candidate_set(true_centers, points, eps_values)
        for name, points in candidate_sets.items()
    }

    old_cmp = old_result_comparison(
        old_index, dim, K, dseed, init_seed, evaluations["v3f2_learned"]
    )
    if old_cmp is not None:
        old_cmp["new_nfe"] = learned_total_nfe

    if artifact_path is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact_path,
            true_centers=true_centers.astype(np.float32),
            mixture_mus=np.asarray(mixture.mus, dtype=np.float32),
            mixture_sigmas=np.asarray(mixture.sigmas, dtype=np.float32),
            mixture_weights=np.asarray(mixture.weights, dtype=np.float32),
            ddim_same_particles=starts.astype(np.float32),
            ddim_same_particles_merge_v3f2=same_merge_v3.astype(np.float32),
            ddim_same_particles_merge_baseline=same_merge_baseline.astype(np.float32),
            v3f2_learned_modes=learned_modes.astype(np.float32),
            v3f2_exact_modes=exact_modes.astype(np.float32),
            ddim_equal_nfe_samples=equal_samples.astype(np.float32),
            ddim_equal_nfe_merge_v3f2=equal_merge_v3.astype(np.float32),
            ddim_equal_nfe_merge_baseline=equal_merge_baseline.astype(np.float32),
        )

    return {
        "init_seed": int(init_seed),
        "eps_values": [float(x) for x in eps_values],
        "parameters": {
            "n_particles": n_particles,
            "ddim_steps": ddim_steps,
            "ddim_transitions": transitions,
            "timesteps": list(mf.timesteps),
            "step_size": float(mf.step_size),
            "ascent_steps": int(overrides.get("ascent_steps") or mf.ascent_steps),
            "refine_steps": int(
                mf.refine_steps if overrides.get("refine_steps") is None else overrides["refine_steps"]
            ),
            "refine_step_scale": float(mf.refine_step_scale),
            "normalize_score": bool(mf.normalize_score),
            "merge_factor": float(mf.merge_factor),
            "merge_radius_min": float(mf.merge_radius_min),
            "baseline_merge_radius": float(cfg.baseline.merge_radius),
        },
        "cost": {
            "ddim_same_particles_neural_nfe": int(init_nfe),
            "v3f2_learned_search_neural_nfe": learned_search_nfe,
            "v3f2_learned_total_neural_nfe": learned_total_nfe,
            "v3f2_exact_neural_nfe_for_shared_starts": int(init_nfe),
            "v3f2_exact_analytic_score_evaluations": exact_score_evals,
            "ddim_equal_nfe_n_samples": equal_nfe_n_samples,
            "ddim_equal_nfe_actual_neural_nfe": int(equal_nfe_actual),
            "v3f2_learned_search_seconds": float(learned_seconds),
            "v3f2_exact_search_seconds": float(exact_seconds),
        },
        "evaluations": evaluations,
        "old_result_comparison": old_cmp,
        "artifact_npz": str(artifact_path) if artifact_path is not None else None,
    }





def load_result_files(folder) :
    records = []
    if not folder.exists():
        return records
    for path in sorted(folder.glob("*.json")):
        try:
            records.append(read_json(path))
        except Exception as exc:
            LOGGER.error("Cannot read result file %s: %s", path, exc)
    return records


def mean_std(values) :
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return None, None, 0
    return float(np.mean(arr)), float(np.std(arr)), int(len(arr))


def write_search_summary(output_dir, records) :
    groups = defaultdict(lambda: defaultdict(list))
    cost_groups = defaultdict(lambda: defaultdict(list))

    for record in records:
        dim = int(record["dim"])
        search = record.get("search_comparison")
        if not search:
            continue
        for method, evaluation in search["evaluations"].items():
            for eps_key, metrics in evaluation["metrics_by_eps"].items():
                eps = float(eps_key)
                for family in ["coverage", "one_to_one"]:
                    for metric in ["recall", "precision", "f1"]:
                        groups[(dim, method, eps)][f"{family}_{metric}"].append(metrics[family][metric])
                groups[(dim, method, eps)]["coverage_soft_error"].append(metrics["coverage"]["soft_error"])
                groups[(dim, method, eps)]["n_found"].append(evaluation["n_found"])
                groups[(dim, method, eps)]["candidate_to_true_ratio"].append(
                    evaluation["candidate_to_true_ratio"]
                )

        cost = search.get("cost", {})
        mapping = {
            "ddim_same_particles_raw": "ddim_same_particles_neural_nfe",
            "ddim_same_particles_merge_v3f2": "ddim_same_particles_neural_nfe",
            "ddim_same_particles_merge_baseline": "ddim_same_particles_neural_nfe",
            "v3f2_learned": "v3f2_learned_total_neural_nfe",
            "v3f2_exact": "v3f2_exact_analytic_score_evaluations",
            "ddim_equal_nfe_raw": "ddim_equal_nfe_actual_neural_nfe",
            "ddim_equal_nfe_merge_v3f2": "ddim_equal_nfe_actual_neural_nfe",
            "ddim_equal_nfe_merge_baseline": "ddim_equal_nfe_actual_neural_nfe",
        }
        for method, cost_key in mapping.items():
            if cost.get(cost_key) is not None:
                cost_groups[(dim, method)]["cost"].append(float(cost[cost_key]))

    rows = []
    nested = {}
    for (dim, method, eps), metrics in sorted(groups.items()):
        row = {"dim": dim, "method": method, "eps": eps}
        for metric_name, values in sorted(metrics.items()):
            mean, std, n = mean_std(values)
            row[f"{metric_name}_mean"] = mean
            row[f"{metric_name}_std"] = std
            row[f"{metric_name}_n"] = n
        cost_mean, cost_std, cost_n = mean_std(cost_groups[(dim, method)].get("cost", []))
        row["cost_mean"] = cost_mean
        row["cost_std"] = cost_std
        row["cost_n"] = cost_n
        rows.append(row)
        nested.setdefault(str(dim), {}).setdefault(method, {})[str(eps)] = row

    csv_path = output_dir / "summary_search_metrics.csv"
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    atomic_write_json(output_dir / "summary_search_metrics.json", nested)
    return {"rows": len(rows), "csv": str(csv_path)}


def write_score_summary(output_dir, records) :
    groups = defaultdict(lambda: defaultdict(list))
    for record in records:
        dim = int(record["dim"])
        score = record.get("score_alignment")
        if not score:
            continue
        for t_key, vals in score["timesteps"].items():
            t = int(t_key)
            groups[(dim, t)]["cosine_mean"].append(vals["cosine"]["mean"])
            groups[(dim, t)]["cosine_median"].append(vals["cosine"]["median"])
            groups[(dim, t)]["angle_deg_mean"].append(vals["angle_deg"]["mean"])
            groups[(dim, t)]["relative_l2_error_mean"].append(vals["relative_l2_error"]["mean"])
            groups[(dim, t)]["fraction_cosine_above_0_9"].append(vals["fraction_cosine_above_0_9"])

    rows = []
    for (dim, t), metrics in sorted(groups.items()):
        row = {"dim": dim, "t": t}
        for name, values in metrics.items():
            mean, std, n = mean_std(values)
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
            row[f"{name}_n"] = n
        rows.append(row)

    csv_path = output_dir / "summary_score_alignment.csv"
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    atomic_write_json(output_dir / "summary_score_alignment.json", rows)
    return {"rows": len(rows), "csv": str(csv_path)}


def write_gmm_summary(output_dir, records) :
    rows = []
    for record in records:
        audit = record.get("gmm_mode_audit")
        if not audit:
            continue
        rows.append(
            {
                "dim": int(record["dim"]),
                "K": int(record["K"]),
                "dseed": int(record["dist_seed"]),
                "n_strict_local_modes": audit["n_strict_local_modes"],
                "center_to_nearest_mode_mean": audit["center_to_nearest_mode_mean"],
                "center_to_nearest_mode_max": audit["center_to_nearest_mode_max"],
                "n_stationary_unique": audit["n_stationary_unique"],
                "n_starts": audit["n_starts"],
            }
        )
    csv_path = output_dir / "summary_gmm_mode_audit.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    atomic_write_json(output_dir / "summary_gmm_mode_audit.json", rows)
    return {"rows": len(rows), "csv": str(csv_path)}





def config_path_for_dim(args, dim) :
    explicit = {
        10: args.config_10,
        30: args.config_30,
        50: args.config_50,
    }.get(dim)
    rel = explicit or DEFAULT_CONFIGS.get(dim)
    if rel is None:
        raise ValueError(f"No config mapping for dim={dim}; use a supported dimension")
    path = Path(rel)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def eps_values_for_dim(dim) :
    if dim not in DEFAULT_EPS:
        raise ValueError(f"No default eps grid for dim={dim}")
    return DEFAULT_EPS[dim]


def mixture_payload(mixture) :
    return {
        "mus": np.asarray(mixture.mus, dtype=float).tolist(),
        "sigmas": np.asarray(mixture.sigmas, dtype=float).tolist(),
        "weights": np.asarray(mixture.weights, dtype=float).tolist(),
        "min_separation": float(mixture.min_separation),
    }


def result_stem(dim, K, dseed) :
    return f"dim{dim}_K{K}_dseed{dseed}"


def search_stem(dim, K, dseed, init_seed) :
    return f"dim{dim}_K{K}_dseed{dseed}_init{init_seed}"


def make_parser() :
    p = argparse.ArgumentParser(description="Run Pareto validation experiments.")
    p.add_argument("--profile", choices=["pilot", "full", "custom"], default="pilot")
    p.add_argument("--dims", default=None, help="Comma/range list, e.g. 10,30,50")
    p.add_argument("--k-values", default=None, help="Comma/range list, e.g. 2-7")
    p.add_argument("--dist-seeds", default=None, help="Comma/range list, e.g. 0-4")
    p.add_argument("--init-seeds", default=None, help="Comma/range list, e.g. 0-2")
    p.add_argument(
        "--experiments",
        default="score_alignment,search_comparison,gmm_mode_audit",
        help="Subset of score_alignment,search_comparison,gmm_mode_audit",
    )
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    p.add_argument("--output-dir", default="./results_pareto_validation")
    p.add_argument("--old-results-dir", default=None)
    p.add_argument("--config-10", default=None)
    p.add_argument("--config-30", default=None)
    p.add_argument("--config-50", default=None)
    p.add_argument("--device", default=None, help="auto, mps, cuda, cpu; default uses each YAML")
    p.add_argument("--score-n", type=int, default=None)
    p.add_argument("--score-timesteps", default=",".join(map(str, DEFAULT_SCORE_TIMESTEPS)))
    p.add_argument("--gmm-random-starts", type=int, default=None)
    p.add_argument("--gmm-maxiter", type=int, default=500)
    p.add_argument("--gmm-grad-tol", type=float, default=1e-6)
    p.add_argument("--gmm-dedup-radius", type=float, default=1e-3)
    p.add_argument("--n-particles", type=int, default=None, help="Diagnostic override only")
    p.add_argument("--ddim-steps", type=int, default=None, help="Diagnostic override only")
    p.add_argument("--ascent-steps", type=int, default=None, help="Diagnostic override only")
    p.add_argument("--refine-steps", type=int, default=None, help="Diagnostic override only")
    p.add_argument("--no-raw-arrays", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--check-only", action="store_true", help="Only audit checkpoint presence/metadata")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p


def resolve_profile(args) :
    if args.profile == "pilot":
        defaults = {
            "dims": [10, 30, 50],
            "k_values": [4],
            "dist_seeds": [0],
            "init_seeds": [0],
            "score_n": 512,
            "gmm_random_starts": 32,
        }
    elif args.profile == "full":
        defaults = {
            "dims": [10, 30, 50],
            "k_values": [2, 3, 4, 5, 6, 7],
            "dist_seeds": [0, 1, 2, 3, 4],
            "init_seeds": [0, 1, 2],
            "score_n": 512,
            "gmm_random_starts": 64,
        }
    else:
        defaults = {
            "dims": [10, 30, 50],
            "k_values": [4],
            "dist_seeds": [0],
            "init_seeds": [0],
            "score_n": 512,
            "gmm_random_starts": 32,
        }

    return {
        "dims": parse_int_list(args.dims) if args.dims else defaults["dims"],
        "k_values": parse_int_list(args.k_values) if args.k_values else defaults["k_values"],
        "dist_seeds": parse_int_list(args.dist_seeds) if args.dist_seeds else defaults["dist_seeds"],
        "init_seeds": parse_int_list(args.init_seeds) if args.init_seeds else defaults["init_seeds"],
        "score_n": int(args.score_n or defaults["score_n"]),
        "gmm_random_starts": int(
            defaults["gmm_random_starts"]
            if args.gmm_random_starts is None
            else args.gmm_random_starts
        ),
    }


def main() :
    args = make_parser().parse_args()
    profile = resolve_profile(args)
    experiments = parse_str_set(args.experiments)
    unknown = experiments - ALL_EXPERIMENTS
    if unknown:
        raise ValueError(f"Unknown experiments: {sorted(unknown)}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (PROJECT_ROOT / checkpoint_dir).resolve()
    old_results_dir = Path(args.old_results_dir) if args.old_results_dir else None
    if old_results_dir is not None and not old_results_dir.is_absolute():
        old_results_dir = (PROJECT_ROOT / old_results_dir).resolve()

    log_path = setup_logging(output_dir, args.log_level)
    LOGGER.info("Pareto validation script v%s", SCRIPT_VERSION)
    LOGGER.info("Project root: %s", PROJECT_ROOT)
    LOGGER.info("Output dir: %s", output_dir)
    LOGGER.info("Checkpoint dir: %s", checkpoint_dir)
    LOGGER.info("Profile: %s | %s", args.profile, profile)
    LOGGER.info("Experiments: %s", sorted(experiments))
    LOGGER.info("Checkpoints are loaded without model training")

    old_index, old_manifest = build_old_results_index(old_results_dir)
    atomic_write_json(output_dir / "old_results_index_manifest.json", old_manifest)
    if old_results_dir is not None:
        LOGGER.info("Old result runs indexed: %d", len(old_index))

    score_timesteps = parse_int_list(args.score_timesteps)
    overrides = {
        "n_particles": args.n_particles,
        "ddim_steps": args.ddim_steps,
        "ascent_steps": args.ascent_steps,
        "refine_steps": args.refine_steps,
    }

    combinations = [
        (dim, K, dseed)
        for dim in profile["dims"]
        for K in profile["k_values"]
        for dseed in profile["dist_seeds"]
    ]
    total_search_runs = len(combinations) * len(profile["init_seeds"])

    manifest = {
        "script_version": SCRIPT_VERSION,
        "started_at_utc": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "log_path": str(log_path),
        "profile_name": args.profile,
        "profile": profile,
        "experiments": sorted(experiments),
        "score_timesteps": score_timesteps,
        "overrides": overrides,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "command": sys.argv,
        "total_mixtures": len(combinations),
        "total_search_runs": total_search_runs,
        "load_only": True,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)

    checkpoint_rows = []
    errors_path = output_dir / "errors.jsonl"
    mixture_results_dir = output_dir / "mixtures"
    search_results_dir = output_dir / "search_runs"
    artifacts_dir = output_dir / "artifacts"
    raw_score_dir = output_dir / "score_alignment_raw"

    completed_mixtures = 0
    completed_search = 0
    global_start = time.perf_counter()

    for mix_idx, (dim, K, dseed) in enumerate(combinations, start=1):
        mix_name = result_stem(dim, K, dseed)
        LOGGER.info(
            "[%d/%d mixtures] START %s",
            mix_idx,
            len(combinations),
            mix_name,
        )
        cfg_path = config_path_for_dim(args, dim)
        cfg = load_config(cfg_path)
        if int(cfg.dim) != dim:
            raise ValueError(f"Config {cfg_path} resolved dim={cfg.dim}, expected {dim}")
        eps_values = eps_values_for_dim(dim)
        device = resolve_device(args.device or cfg.device)

        mixture = GaussianMixture.random(
            K=K,
            dim=dim,
            sigma_range=(0.5, 1.2),
            min_sep=2.0,
            bounds=(cfg.distribution.x_min, cfg.distribution.x_max),
            seed=dseed,
        )

        try:
            checkpoint_path = find_checkpoint_load_only(checkpoint_dir, dim, K, dseed)
            model, checkpoint = load_model_only(cfg, checkpoint_path, device)
            checkpoint_record = {
                "dim": dim,
                "K": K,
                "dist_seed": dseed,
                "path": str(checkpoint_path),
                "sha256": file_sha256(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
                "train_step": int(checkpoint.get("train_step", 0)),
                "last_loss": float(checkpoint.get("last_loss", float("nan"))),
                "checkpoint_dim": int(checkpoint.get("dim", dim)),
                "checkpoint_T": int(checkpoint.get("T", cfg.model.T)),
                "device": str(device),
                "status": "loaded",
            }
            checkpoint_rows.append(checkpoint_record)
            LOGGER.info(
                "Loaded %s | step=%d loss=%.6g device=%s",
                checkpoint_path,
                checkpoint_record["train_step"],
                checkpoint_record["last_loss"],
                device,
            )
        except Exception as exc:
            error = {
                "time_utc": utc_now(),
                "stage": "checkpoint_load",
                "dim": dim,
                "K": K,
                "dist_seed": dseed,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(errors_path, error)
            checkpoint_rows.append(
                {
                    "dim": dim,
                    "K": K,
                    "dist_seed": dseed,
                    "path": None,
                    "status": "error",
                    "error": str(exc),
                }
            )
            LOGGER.error("SKIP %s: %s", mix_name, exc)
            if args.fail_fast:
                raise
            continue

        if args.check_only:
            del model, checkpoint
            cleanup_device(device)
            continue

        mixture_path = mixture_results_dir / f"{mix_name}.json"
        if mixture_path.exists() and not args.overwrite:
            mixture_record = read_json(mixture_path)
            LOGGER.info("Resume: mixture-level result exists, skipped: %s", mixture_path)
        else:
            mixture_record = {
                "created_at_utc": utc_now(),
                "dim": dim,
                "K": K,
                "dist_seed": dseed,
                "config_path": str(cfg_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_train_step": int(checkpoint.get("train_step", 0)),
                "checkpoint_last_loss": float(checkpoint.get("last_loss", float("nan"))),
                "eps_values": eps_values,
                "mixture": mixture_payload(mixture),
            }
            try:
                if "score_alignment" in experiments:
                    raw_path = None if args.no_raw_arrays else raw_score_dir / f"{mix_name}.npz"
                    LOGGER.info("%s: score alignment on %d points", mix_name, profile["score_n"])
                    t0 = time.perf_counter()
                    mixture_record["score_alignment"] = run_score_alignment(
                        model=model,
                        mixture=mixture,
                        timesteps=score_timesteps,
                        n=profile["score_n"],
                        seed=100_000 + dseed,
                        raw_npz_path=raw_path,
                    )
                    mixture_record["score_alignment_seconds"] = time.perf_counter() - t0
                    first_t = next(iter(mixture_record["score_alignment"]["timesteps"].values()), None)
                    if first_t:
                        LOGGER.info(
                            "%s: score alignment done; first-t cosine mean=%.3f",
                            mix_name,
                            first_t["cosine"]["mean"],
                        )

                if "gmm_mode_audit" in experiments:
                    LOGGER.info(
                        "%s: numerical GMM-mode audit (%d random starts)",
                        mix_name,
                        profile["gmm_random_starts"],
                    )
                    t0 = time.perf_counter()
                    mixture_record["gmm_mode_audit"] = run_gmm_mode_audit(
                        mixture=mixture,
                        bounds=(cfg.distribution.x_min, cfg.distribution.x_max),
                        n_random_starts=profile["gmm_random_starts"],
                        seed=200_000 + dseed,
                        grad_tol=args.gmm_grad_tol,
                        dedup_radius=args.gmm_dedup_radius,
                        maxiter=args.gmm_maxiter,
                    )
                    mixture_record["gmm_mode_audit_seconds"] = time.perf_counter() - t0
                    LOGGER.info(
                        "%s: GMM audit done; K=%d, numerical strict modes=%d, max center shift=%.4f",
                        mix_name,
                        K,
                        mixture_record["gmm_mode_audit"]["n_strict_local_modes"],
                        mixture_record["gmm_mode_audit"]["center_to_nearest_mode_max"],
                    )
                atomic_write_json(mixture_path, mixture_record)
            except Exception as exc:
                error = {
                    "time_utc": utc_now(),
                    "stage": "mixture_experiments",
                    "dim": dim,
                    "K": K,
                    "dist_seed": dseed,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(errors_path, error)
                LOGGER.error("Mixture-level experiment failed for %s: %s", mix_name, exc)
                if args.fail_fast:
                    raise

        completed_mixtures += 1

        if "search_comparison" in experiments:
            for init_seed in profile["init_seeds"]:
                completed_search += 1
                run_name = search_stem(dim, K, dseed, init_seed)
                run_path = search_results_dir / f"{run_name}.json"
                if run_path.exists() and not args.overwrite:
                    LOGGER.info(
                        "[%d/%d searches] Resume skip %s",
                        completed_search,
                        total_search_runs,
                        run_name,
                    )
                    continue
                LOGGER.info(
                    "[%d/%d searches] START %s",
                    completed_search,
                    total_search_runs,
                    run_name,
                )
                try:
                    artifact_path = None if args.no_raw_arrays else artifacts_dir / f"{run_name}.npz"
                    t0 = time.perf_counter()
                    comparison = run_search_comparison(
                        model=model,
                        mixture=mixture,
                        cfg=cfg,
                        eps_values=eps_values,
                        init_seed=init_seed,
                        overrides=overrides,
                        artifact_path=artifact_path,
                        old_index=old_index,
                        dim=dim,
                        K=K,
                        dseed=dseed,
                    )
                    record = {
                        "created_at_utc": utc_now(),
                        "dim": dim,
                        "K": K,
                        "dist_seed": dseed,
                        "init_seed": init_seed,
                        "config_path": str(cfg_path),
                        "checkpoint_path": str(checkpoint_path),
                        "mixture": mixture_payload(mixture),
                        "search_comparison": comparison,
                        "total_seconds": time.perf_counter() - t0,
                    }
                    atomic_write_json(run_path, record)

                    main_eps = float(cfg.metrics.eps_hit)
                    
                    if dim == 10:
                        main_eps = 2.0
                    elif dim == 30:
                        main_eps = 3.5
                    elif dim == 50:
                        main_eps = 4.5
                    key = str(float(main_eps))
                    learned = comparison["evaluations"]["v3f2_learned"]["metrics_by_eps"][key]
                    ddim = comparison["evaluations"]["ddim_equal_nfe_merge_v3f2"]["metrics_by_eps"][key]
                    LOGGER.info(
                        "DONE %s | learned coverage F1=%.3f, 1:1 F1=%.3f, n=%d | "
                        "equal-NFE DDIM coverage F1=%.3f, 1:1 F1=%.3f, n=%d",
                        run_name,
                        learned["coverage"]["f1"],
                        learned["one_to_one"]["f1"],
                        comparison["evaluations"]["v3f2_learned"]["n_found"],
                        ddim["coverage"]["f1"],
                        ddim["one_to_one"]["f1"],
                        comparison["evaluations"]["ddim_equal_nfe_merge_v3f2"]["n_found"],
                    )
                except Exception as exc:
                    error = {
                        "time_utc": utc_now(),
                        "stage": "search_comparison",
                        "dim": dim,
                        "K": K,
                        "dist_seed": dseed,
                        "init_seed": init_seed,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    append_jsonl(errors_path, error)
                    LOGGER.error("Search run failed for %s: %s", run_name, exc)
                    if args.fail_fast:
                        raise

        del model, checkpoint
        cleanup_device(device)
        LOGGER.info(
            "[%d/%d mixtures] END %s | elapsed total %.1f min",
            mix_idx,
            len(combinations),
            mix_name,
            (time.perf_counter() - global_start) / 60.0,
        )

    
    checkpoint_csv = output_dir / "checkpoint_inventory.csv"
    if checkpoint_rows:
        fieldnames = sorted({k for row in checkpoint_rows for k in row})
        with checkpoint_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(checkpoint_rows)
    atomic_write_json(output_dir / "checkpoint_inventory.json", checkpoint_rows)

    if args.check_only:
        LOGGER.info("Checkpoint-only audit finished. Inventory: %s", checkpoint_csv)
        return 0

    mixture_records = load_result_files(mixture_results_dir)
    search_records = load_result_files(search_results_dir)
    summary = {
        "finished_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - global_start,
        "mixture_result_files": len(mixture_records),
        "search_result_files": len(search_records),
        "search_summary": write_search_summary(output_dir, search_records),
        "score_summary": write_score_summary(output_dir, mixture_records),
        "gmm_summary": write_gmm_summary(output_dir, mixture_records),
        "errors_file": str(errors_path) if errors_path.exists() else None,
        "checkpoint_inventory": str(checkpoint_csv),
    }
    atomic_write_json(output_dir / "summary_manifest.json", summary)

    LOGGER.info("ALL REQUESTED EXPERIMENTS FINISHED")
    LOGGER.info("Mixture files: %d | Search files: %d", len(mixture_records), len(search_records))
    LOGGER.info("Main summary: %s", output_dir / "summary_search_metrics.csv")
    LOGGER.info("Score summary: %s", output_dir / "summary_score_alignment.csv")
    LOGGER.info("GMM summary: %s", output_dir / "summary_gmm_mode_audit.csv")
    LOGGER.info("Full log: %s", log_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user. Completed JSON files remain resumable.")
        raise SystemExit(130)
