from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch

from ..models.diffusion import DiffusionModel
from .clustering import merge_close, agglomerative_merge

logger = logging.getLogger(__name__)


@dataclass
class ModeFinderResultV2:
    modes: np.ndarray
    nfe: int
    nfe_starts: int = 0
    nfe_search: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    starts: np.ndarray = field(default_factory=lambda: np.empty(0))


def ddim_sample(
    model: DiffusionModel,
    n: int,
    num_steps: int = 50,
    stop_t: int = 0,
    seed: int = 0,
) -> np.ndarray:
    torch.manual_seed(seed)
    model.net.eval()
    s = model.schedule
    device = model.device
    T = s.T

    step_indices = np.linspace(T - 1, max(stop_t, 0), num_steps + 1, dtype=int)
    step_indices = np.unique(step_indices)[::-1]

    x = torch.randn(n, model.dim, device=device)

    with torch.no_grad():
        for i in range(len(step_indices) - 1):
            t_curr = int(step_indices[i])
            t_next = int(step_indices[i + 1])

            t = torch.full((n,), t_curr, device=device, dtype=torch.long)

            if model._nfe_counter is not None:
                eps_pred = model._nfe_counter(x, t)
            else:
                eps_pred = model.net(x, t)

            alpha_t = s.alphas_cumprod[t_curr]
            alpha_next = (
                s.alphas_cumprod[t_next]
                if t_next > 0
                else torch.tensor(1.0, device=device)
            )

            x0_pred = (x - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
            x = (
                torch.sqrt(alpha_next) * x0_pred
                + torch.sqrt(1 - alpha_next) * eps_pred
            )

    result = x.cpu().numpy()
    return result.squeeze(-1) if model.dim == 1 else result


class ModeFinderV2:

    def __init__(
        self,
        model: DiffusionModel,
        timesteps: Optional[List[int]] = None,
        step_size: float = 0.01,
        split_method: str = "hessian",
        split_eps: float = 2.0,
        split_threshold: float = 0.5,
        hessian_fd_eps: float = 1e-3,
        hessian_split_eigenvalue_threshold: float = -1.0,
        max_split_directions: int = 3,
        merge_radius: float = 0.5,
        adaptive_merge: bool = True,
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

        self.split_method = split_method
        self.split_eps = split_eps
        self.split_threshold = split_threshold
        self.hessian_fd_eps = hessian_fd_eps
        self.hessian_split_eigenvalue_threshold = hessian_split_eigenvalue_threshold
        self.max_split_directions = max_split_directions

        self.merge_radius = merge_radius
        self.adaptive_merge = adaptive_merge

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

        logger.info(
            "Smart starts: %d pilot -> %d centers -> %d starts (nfe=%d)",
            n_pilot, len(centers), self.n_starts, nfe_overhead,
        )
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

    def _check_split_hessian(self, x: np.ndarray, t: int) -> List[np.ndarray]:
        sigma_t = self.model.schedule.sigma(t)
        H = self._approximate_hessian(x, t)
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        split_mask = eigenvalues > self.hessian_split_eigenvalue_threshold
        split_directions = eigenvectors[:, split_mask].T

        if len(split_directions) > self.max_split_directions:
            split_evals = eigenvalues[split_mask]
            top_idx = np.argsort(-split_evals)[:self.max_split_directions]
            split_directions = split_directions[top_idx]

        if len(split_directions) == 0:
            refined = self._gradient_ascent(
                x, t, self.step_size, num_steps=self.ascent_steps // 2,
            )
            return [refined]

        split_points: List[np.ndarray] = []
        delta = max(1e-4, self.split_eps * sigma_t)

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
            if dist > self.split_threshold:
                split_points.extend([y_plus, y_minus])

        if not split_points:
            refined = self._gradient_ascent(
                x, t, self.step_size, num_steps=self.ascent_steps // 2,
            )
            return [refined]
        return split_points

    def _check_split_axes(self, x: np.ndarray, t: int) -> List[np.ndarray]:
        sigma_t = self.model.schedule.sigma(t)
        delta = max(1e-4, self.split_eps * sigma_t)
        directions = np.eye(self.dim)
        split_points: List[np.ndarray] = []

        for direction in directions:
            perturbation = direction * delta
            y_plus = self._gradient_ascent(
                x + perturbation, t, self.step_size,
                num_steps=self.ascent_steps // 2,
            )
            y_minus = self._gradient_ascent(
                x - perturbation, t, self.step_size,
                num_steps=self.ascent_steps // 2,
            )
            if np.linalg.norm(y_plus - y_minus) > self.split_threshold:
                split_points.extend([y_plus, y_minus])

        if not split_points:
            refined = self._gradient_ascent(
                x, t, self.step_size, num_steps=self.ascent_steps // 2,
            )
            return [refined]
        return split_points

    def _check_split(self, x: np.ndarray, t: int) -> List[np.ndarray]:
        if self.split_method == "hessian":
            return self._check_split_hessian(x, t)
        else:
            return self._check_split_axes(x, t)


    def _get_merge_radius(self, t: int) -> float:
        if not self.adaptive_merge:
            return self.merge_radius

        sigma_t = self.model.schedule.sigma(t) if t > 0 else 0.01
        sigma_max = self.model.schedule.sigma(self.model.schedule.T - 1)

        dim_scale = np.sqrt(self.dim)
        noise_ratio = sigma_t / sigma_max

        return self.merge_radius * (1.0 + 2.0 * noise_ratio * dim_scale)

    def _cluster(self, points: List[np.ndarray],
                 merge_r: Optional[float] = None) -> np.ndarray:
        if not points:
            return np.empty((0, self.dim))

        arr = np.array(points)
        if arr.ndim == 1 and self.dim == 1:
            arr = arr.reshape(-1, 1)

        r = merge_r if merge_r is not None else self.merge_radius

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

        final_modes = self._cluster(all_modes)
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
            "V2 search done: %d modes, NFE=%d (starts=%d, search=%d)",
            len(final_modes) if final_modes.ndim > 0 else 0,
            total_nfe, nfe_starts, nfe_search,
        )
        return result

    def _run_single_start(self, x0: np.ndarray,
                          verbose: bool) -> List[np.ndarray]:
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
                split_result = self._check_split(x, t)
                after_split.extend(split_result)

            clustered = self._cluster(after_split, merge_r)
            active = [clustered[i] for i in range(len(clustered))]

            active = self._cap_active_points(active)

            sigma = self.model.schedule.sigma(t) if t > 0 else 0.0
            logger.info(
                "    t=%d (sigma=%.3f): %d active (merge_r=%.3f)",
                t, sigma, len(active), merge_r,
            )

        alpha_refine = self.step_size * self.refine_step_scale
        finals = [
            self._gradient_ascent(x, 0, alpha_refine, self.refine_steps)
            for x in active
        ]
        return finals


class CoarseToFineFinder:

    def __init__(
        self,
        model: DiffusionModel,
        n_samples: int = 200,
        ddim_steps: int = 50,
        cluster_method: str = "meanshift",
        n_clusters: Optional[int] = None,
        cluster_bandwidth: float = 2.0,
        refine_steps: int = 200,
        refine_alpha: float = 0.001,
        merge_radius: float = 0.5,
        normalize_score: bool = True,
        x_min: float = -10.0,
        x_max: float = 10.0,
    ):
        self.model = model
        self.dim = model.dim
        self.device = model.device

        self.n_samples = n_samples
        self.ddim_steps = ddim_steps
        self.cluster_method = cluster_method
        self.n_clusters = n_clusters
        self.cluster_bandwidth = cluster_bandwidth
        self.refine_steps = refine_steps
        self.refine_alpha = refine_alpha
        self.merge_radius = merge_radius
        self.normalize_score = normalize_score
        self.x_min = x_min
        self.x_max = x_max

    def find_modes(self, seed: int = 0, verbose: bool = True) -> ModeFinderResultV2:
        self.model.enable_nfe_counting()
        self.model.reset_nfe()

        if verbose:
            logger.info("C2F coarse: %d DDIM samples (%d steps)...",
                        self.n_samples, self.ddim_steps)

        samples = ddim_sample(
            self.model, self.n_samples,
            num_steps=self.ddim_steps, stop_t=0, seed=seed,
        )
        nfe_coarse = self.model.nfe

        centers = self._cluster_samples(np.atleast_2d(samples))

        if verbose:
            logger.info("C2F coarse: %d samples -> %d clusters (NFE=%d)",
                        self.n_samples, len(centers), nfe_coarse)

        self.model.reset_nfe()

        refined_modes = []
        for center in centers:
            mode = self._refine_to_mode(center)
            refined_modes.append(mode)

        nfe_fine = self.model.nfe

        if len(refined_modes) > 0:
            refined_arr = np.array(refined_modes)
            if self.dim == 1:
                final_modes = merge_close(refined_arr.flatten(), self.merge_radius)
            else:
                final_modes = agglomerative_merge(refined_arr, self.merge_radius)
        else:
            final_modes = np.empty((0, self.dim))

        if self.dim == 1:
            final_modes = final_modes.flatten()

        total_nfe = nfe_coarse + nfe_fine

        if verbose:
            n_modes = len(final_modes) if final_modes.ndim > 0 else 0
            logger.info("C2F done: %d modes, NFE=%d (coarse=%d, fine=%d)",
                        n_modes, total_nfe, nfe_coarse, nfe_fine)

        return ModeFinderResultV2(
            modes=final_modes,
            nfe=total_nfe,
            nfe_starts=nfe_coarse,
            nfe_search=nfe_fine,
            history=[],
            starts=centers if isinstance(centers, np.ndarray) else np.array(centers),
        )

    def _cluster_samples(self, samples: np.ndarray) -> np.ndarray:
        from sklearn.cluster import MeanShift, KMeans

        samples = np.atleast_2d(samples)

        if self.cluster_method == "kmeans" and self.n_clusters is not None:
            km = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            km.fit(samples)
            return km.cluster_centers_
        elif self.cluster_method == "meanshift":
            ms = MeanShift(bandwidth=self.cluster_bandwidth)
            ms.fit(samples)
            return ms.cluster_centers_
        else:
            return agglomerative_merge(samples, self.cluster_bandwidth)

    def _refine_to_mode(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        for _ in range(self.refine_steps):
            score = self.model.score_numpy(x, t=0).flatten()

            if self.normalize_score:
                score_norm = np.linalg.norm(score)
                if score_norm > 1e-8:
                    score = score / score_norm

            x = x + self.refine_alpha * score
            x = np.clip(x, self.x_min, self.x_max)
        return x
