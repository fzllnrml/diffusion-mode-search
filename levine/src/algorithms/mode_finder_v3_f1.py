from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch

from ..models.diffusion import DiffusionModel
from .clustering import merge_close, agglomerative_merge
from .mode_finder_v2 import ddim_sample, ModeFinderResultV2  # переиспользуем

logger = logging.getLogger(__name__)


class ModeFinderV3F1:

    def __init__(
        self,
        model: DiffusionModel,
        timesteps: Optional[List[int]] = None,
        step_size: float = 0.01,
        split_eps: float = 2.0,
        softness_threshold: float = 0.15,
        amplification_threshold: float = 2.5,
        tau_abs_min: float = 0.05,
        max_split_directions: int = 2,
        hessian_fd_eps: float = 1e-3,
        merge_factor: float = 0.8,
        merge_radius_min: float = 0.05,
        ascent_steps: int = 20,
        normalize_score: bool = True,
        refine_steps: int = 100,
        refine_step_scale: float = 0.1,
        n_starts: int = 5,
        start_method: str = "smart",
        ddim_steps: int = 50,
        n_pilot_multiplier: int = 3,
        pilot_cluster_radius: float = 2.0,
        starts_min_sep: float = 0.5,
        max_active_per_start: int = 30,
        x_min: float = -10.0,
        x_max: float = 10.0,
        **_ignored,
    ):
        self.model = model
        self.dim = model.dim
        self.device = model.device

        self.timesteps = timesteps or self._default_timesteps()
        self.step_size = step_size

        self.split_eps = split_eps
        self.softness_threshold = softness_threshold
        self.amplification_threshold = amplification_threshold
        self.tau_abs_min = tau_abs_min
        self.max_split_directions = max_split_directions
        self.hessian_fd_eps = hessian_fd_eps

        self.merge_factor = merge_factor
        self.merge_radius_min = merge_radius_min

        self.ascent_steps = ascent_steps
        self.normalize_score = normalize_score

        self.refine_steps = refine_steps
        self.refine_step_scale = refine_step_scale

        self.n_starts = n_starts
        self.start_method = start_method
        self.ddim_steps = ddim_steps
        self.n_pilot_multiplier = n_pilot_multiplier
        self.pilot_cluster_radius = pilot_cluster_radius
        self.starts_min_sep = starts_min_sep

        self.max_active_per_start = max_active_per_start

        self.x_min = x_min
        self.x_max = x_max

    def _default_timesteps(self) -> List[int]:
        if self.dim <= 2:
            return [800, 500, 300, 200, 150, 100, 80, 60, 40, 30, 20, 10, 5, 0]
        elif self.dim <= 10:
            return [800, 600, 400, 300, 200, 150, 100, 70, 50, 30, 15, 5, 0]
        else:
            return [800, 500, 300, 200, 100, 50, 20, 5, 0]


    def _generate_smart_starts(self, seed: int = 0) -> Tuple[np.ndarray, int]:
        n_pilot = self.n_starts * self.n_pilot_multiplier
        self.model.enable_nfe_counting()
        self.model.reset_nfe()

        samples = ddim_sample(
            self.model, n_pilot,
            num_steps=self.ddim_steps, stop_t=0, seed=seed,
        )
        nfe_overhead = self.model.nfe

        samples_2d = np.atleast_2d(samples)
        if self.dim == 1 and samples_2d.shape[1] != 1:
            samples_2d = samples_2d.reshape(-1, 1)

        if self.dim == 1:
            centers = merge_close(samples_2d.flatten(), self.pilot_cluster_radius)
            centers = centers.reshape(-1, 1)
        else:
            centers = agglomerative_merge(samples_2d, self.pilot_cluster_radius)

        if len(centers) > self.n_starts:
            centers = self._farthest_point_sampling(centers, self.n_starts)
        elif len(centers) < self.n_starts:
            rng = np.random.default_rng(seed)
            while len(centers) < self.n_starts:
                idx = rng.integers(len(samples_2d))
                centers = np.vstack([centers, samples_2d[idx:idx + 1]])

        return centers[:self.n_starts], nfe_overhead

    def _farthest_point_sampling(self, points: np.ndarray, k: int) -> np.ndarray:
        points = np.atleast_2d(points)
        n = len(points)
        if n <= k:
            return points
        selected = [0]
        min_dists = np.full(n, np.inf)
        for _ in range(k - 1):
            last = points[selected[-1]]
            dists = np.linalg.norm(points - last, axis=1)
            min_dists = np.minimum(min_dists, dists)
            next_idx = np.argmax(min_dists)
            selected.append(next_idx)
        return points[selected]

    def _generate_uniform_starts(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        starts = []
        for _ in range(100_000):
            if len(starts) >= self.n_starts:
                break
            cand = rng.uniform(self.x_min, self.x_max, size=self.dim)
            if all(np.linalg.norm(cand - s) >= self.starts_min_sep for s in starts):
                starts.append(cand)
        return np.array(starts)


    def _gradient_ascent(self, x: np.ndarray, t: int, alpha: float,
                         num_steps: int, tol: float = 1e-6) -> np.ndarray:
        x = x.copy()
        for _ in range(num_steps):
            score = self.model.score_numpy(x, t).flatten()
            if self.normalize_score:
                score_norm = np.linalg.norm(score)
                if score_norm > 1e-8:
                    score = score / score_norm
            x_new = x + alpha * score
            x_new = np.clip(x_new, self.x_min, self.x_max)
            if np.linalg.norm(x_new - x) < tol:
                x = x_new
                break
            x = x_new
        return x


    def _approximate_hessian(self, x: np.ndarray, t: int) -> np.ndarray:
        eps = self.hessian_fd_eps
        d = self.dim
        H = np.zeros((d, d))
        for j in range(d):
            e_j = np.zeros(d)
            e_j[j] = eps
            s_plus = self.model.score_numpy(x + e_j, t).flatten()
            s_minus = self.model.score_numpy(x - e_j, t).flatten()
            H[:, j] = (s_plus - s_minus) / (2 * eps)
        return (H + H.T) / 2

    def _select_split_directions(
        self, eigenvalues: np.ndarray, eigenvectors: np.ndarray
    ) -> np.ndarray:
        eps_psd = 1e-6
        n_max = min(self.max_split_directions, 2)  # жёсткий cap для безопасности

        pos_mask = eigenvalues > eps_psd
        neg_mask = eigenvalues < -eps_psd
        n_pos = pos_mask.sum()

        selected_indices = []

        lambda_min = eigenvalues[neg_mask].min() if neg_mask.any() else -eps_psd

        if n_pos == 0:
            neg_indices = np.where(neg_mask)[0]
            if len(neg_indices) == 0:
                return np.empty((0, self.dim))
            ratios = eigenvalues[neg_indices] / abs(lambda_min)  # в [0, 1]
            soft_mask = ratios > self.softness_threshold
            soft_indices = neg_indices[soft_mask]
            if len(soft_indices) > 0:
                best = soft_indices[np.argmax(ratios[soft_mask])]
                selected_indices = [best]

        elif n_pos == 1:
            pos_idx = np.where(pos_mask)[0][0]
            selected_indices = [pos_idx]
            neg_indices = np.where(neg_mask)[0]
            if len(neg_indices) > 0:
                ratios_neg = eigenvalues[neg_indices] / abs(lambda_min)
                softest_neg_idx = neg_indices[np.argmax(ratios_neg)]
                if ratios_neg.max() > self.softness_threshold:
                    selected_indices.append(softest_neg_idx)

        else:
            pos_indices = np.where(pos_mask)[0]
            top_k = min(n_max, len(pos_indices))
            sorted_pos = pos_indices[np.argsort(-eigenvalues[pos_indices])]
            selected_indices = list(sorted_pos[:top_k])

        if not selected_indices:
            return np.empty((0, self.dim))

        selected_indices = selected_indices[:n_max]
        return eigenvectors[:, selected_indices].T  # shape: (k, dim)

    def _check_split_hessian(self, x: np.ndarray, t: int) -> List[np.ndarray]:
        sigma_t = self.model.schedule.sigma(t)
        H = self._approximate_hessian(x, t)
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        split_directions = self._select_split_directions(eigenvalues, eigenvectors)

        if len(split_directions) == 0:
            refined = self._gradient_ascent(
                x, t, self.step_size, num_steps=self.ascent_steps // 2,
            )
            return [refined]

        delta = max(1e-4, self.split_eps * sigma_t)
        initial_separation = 2 * delta

        split_points: List[np.ndarray] = []

        for direction in split_directions:
            direction = direction / (np.linalg.norm(direction) + 1e-9)
            perturbation = direction * delta

            y_plus = self._gradient_ascent(
                x + perturbation, t, self.step_size,
                num_steps=self.ascent_steps // 2,
            )
            y_minus = self._gradient_ascent(
                x - perturbation, t, self.step_size,
                num_steps=self.ascent_steps // 2,
            )

            dist = np.linalg.norm(y_plus - y_minus)
            amp = dist / max(initial_separation, 1e-8)
            if amp > self.amplification_threshold and dist > self.tau_abs_min:
                split_points.extend([y_plus, y_minus])

        if not split_points:
            refined = self._gradient_ascent(
                x, t, self.step_size, num_steps=self.ascent_steps // 2,
            )
            return [refined]

        return split_points


    def _get_merge_radius(self, t: int) -> float:
        if t == 0:
            sigma_t = 0.01
        else:
            sigma_t = self.model.schedule.sigma(t)
        return max(self.merge_radius_min, self.merge_factor * sigma_t)

    def _cluster(self, points: List[np.ndarray],
                 merge_r: Optional[float] = None) -> np.ndarray:
        if not points:
            return np.empty((0, self.dim))

        arr = np.array(points)
        if arr.ndim == 1 and self.dim == 1:
            arr = arr.reshape(-1, 1)

        r = merge_r if merge_r is not None else self.merge_radius_min

        if self.dim == 1:
            merged = merge_close(arr.flatten(), r)
            return merged.reshape(-1, 1)
        else:
            return agglomerative_merge(arr, r)

    def _cap_active_points(self, points: List[np.ndarray]) -> List[np.ndarray]:
        if len(points) <= self.max_active_per_start:
            return points
        logger.warning(
            "    CAP: %d active -> %d (max_active_per_start)",
            len(points), self.max_active_per_start,
        )
        arr = np.array(points)
        selected = self._farthest_point_sampling(arr, self.max_active_per_start)
        return [selected[i] for i in range(len(selected))]


    def find_modes(
        self,
        starts: Optional[np.ndarray] = None,
        seed: int = 0,
        verbose: bool = True,
    ) -> ModeFinderResultV2:
        self.model.enable_nfe_counting()
        self.model.reset_nfe()

        nfe_starts = 0

        if starts is None:
            if self.start_method == "smart":
                starts, nfe_starts = self._generate_smart_starts(seed)
            else:
                starts = self._generate_uniform_starts(seed)

        starts = np.atleast_2d(starts)
        if self.dim == 1 and starts.shape[1] != 1:
            starts = starts.reshape(-1, 1)

        self.model.reset_nfe()
        all_modes: List[np.ndarray] = []

        for i, start in enumerate(starts):
            nfe_before = self.model.nfe
            modes_from_start = self._run_single_start(start, verbose)
            all_modes.extend(modes_from_start)

            if verbose:
                logger.info(
                    "  Start %d/%d: %d modes, NFE=%d",
                    i + 1, len(starts), len(modes_from_start),
                    self.model.nfe - nfe_before,
                )

        nfe_search = self.model.nfe

        final_modes = self._cluster(all_modes, self.merge_radius_min)
        if self.dim == 1:
            final_modes = final_modes.flatten()

        total_nfe = nfe_starts + nfe_search

        result = ModeFinderResultV2(
            modes=final_modes,
            nfe=total_nfe,
            nfe_starts=nfe_starts,
            nfe_search=nfe_search,
            history=[],
            starts=starts,
        )

        logger.info(
            "V3F1 done: %d modes, NFE=%d (starts=%d, search=%d)",
            len(final_modes) if final_modes.ndim > 0 else 0,
            total_nfe, nfe_starts, nfe_search,
        )
        return result

    def _run_single_start(self, x0: np.ndarray, verbose: bool) -> List[np.ndarray]:
        active = [x0.copy()]

        for t in self.timesteps:
            t = int(t)
            merge_r = self._get_merge_radius(t)

            after_ascent = [
                self._gradient_ascent(x, t, self.step_size, self.ascent_steps)
                for x in active
            ]

            after_split: List[np.ndarray] = []
            for x in after_ascent:
                after_split.extend(self._check_split_hessian(x, t))

            clustered = self._cluster(after_split, merge_r)
            active = [clustered[i] for i in range(len(clustered))]
            active = self._cap_active_points(active)

            sigma = self.model.schedule.sigma(t) if t > 0 else 0.0
            logger.debug(
                "    t=%d (sigma=%.3f): %d active (merge_r=%.3f)",
                t, sigma, len(active), merge_r,
            )

        alpha_refine = self.step_size * self.refine_step_scale
        finals = [
            self._gradient_ascent(x, 0, alpha_refine, self.refine_steps)
            for x in active
        ]
        return finals
