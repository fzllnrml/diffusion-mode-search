from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
import torch

from ..models.diffusion import DiffusionModel
from .clustering import merge_close, agglomerative_merge

logger = logging.getLogger(__name__)


@dataclass
class ModeFinderResult:
    modes: np.ndarray             # Найденные моды, shape (M, D)
    nfe: int                      # Число вызовов нейросети
    history: List[Dict[str, Any]] # История по шагам (t, sigma, active_modes)
    starts: np.ndarray            # Стартовые точки


class ModeFinder:

    def __init__(
        self,
        model: DiffusionModel,
        timesteps: Optional[List[int]] = None,
        step_size: float = 0.01,
        split_eps: float = 2.0,
        split_threshold: float = 0.3,
        merge_radius: float = 0.25,
        ascent_steps: int = 20,
        refine_steps: int = 100,
        refine_step_scale: float = 0.1,
        n_starts: int = 3,
        starts_min_sep: float = 0.5,
        split_directions: str = "axes",
        x_min: float = -10.0,
        x_max: float = 10.0,
    ):
        self.model = model
        self.dim = model.dim
        self.device = model.device

        self.timesteps = timesteps or [800, 500, 300, 200, 150, 100, 80, 60, 40, 30, 20, 10, 5, 0]
        self.step_size = step_size
        self.split_eps = split_eps
        self.split_threshold = split_threshold
        self.merge_radius = merge_radius
        self.ascent_steps = ascent_steps
        self.refine_steps = refine_steps
        self.refine_step_scale = refine_step_scale
        self.n_starts = n_starts
        self.starts_min_sep = starts_min_sep
        self.split_directions = split_directions
        self.x_min = x_min
        self.x_max = x_max


    def _gradient_ascent(self, x: np.ndarray, t: int, alpha: float,
                         num_steps: int, tol: float = 1e-6) -> np.ndarray:
        x = x.copy()
        for _ in range(num_steps):
            score = self.model.score_numpy(x, t)
            x_new = x + alpha * score.flatten()
            x_new = np.clip(x_new, self.x_min, self.x_max)

            if np.linalg.norm(x_new - x) < tol:
                x = x_new
                break
            x = x_new
        return x


    def _gradient_ascent_batch(self, points: np.ndarray, t: int,
                               alpha: float, num_steps: int) -> np.ndarray:
        x = torch.tensor(points, device=self.device, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        n = x.shape[0]

        for _ in range(num_steps):
            t_tensor = torch.full((n,), t, device=self.device, dtype=torch.long)
            score = self.model.score(x, t)
            x = x + alpha * score
            x = x.clamp(self.x_min, self.x_max)

        return x.cpu().numpy()


    def _check_split(self, x: np.ndarray, t: int) -> List[np.ndarray]:
        sigma_t = self.model.schedule.sigma(t)
        delta = max(1e-4, self.split_eps * sigma_t)

        if self.split_directions == "axes":
            directions = np.eye(self.dim)
        else:
            d = np.random.randn(self.dim)
            d = d / (np.linalg.norm(d) + 1e-9)
            directions = d.reshape(1, -1)

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

            dist = np.linalg.norm(y_plus - y_minus)

            if dist > self.split_threshold:
                split_points.extend([y_plus, y_minus])

        if not split_points:
            refined = self._gradient_ascent(
                x, t, self.step_size, num_steps=self.ascent_steps // 2,
            )
            return [refined]

        return split_points


    def _cluster(self, points: List[np.ndarray]) -> np.ndarray:
        if not points:
            return np.empty((0, self.dim))

        arr = np.array(points)
        if arr.ndim == 1 and self.dim == 1:
            arr = arr.reshape(-1, 1)

        if self.dim == 1:
            merged = merge_close(arr.flatten(), self.merge_radius)
            return merged.reshape(-1, 1)
        else:
            return agglomerative_merge(arr, self.merge_radius)


    def _generate_starts(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        starts: List[np.ndarray] = []

        for _ in range(100_000):
            if len(starts) >= self.n_starts:
                break
            cand = rng.uniform(self.x_min, self.x_max, size=self.dim)
            if all(
                np.linalg.norm(cand - s) >= self.starts_min_sep
                for s in starts
            ):
                starts.append(cand)

        if len(starts) < self.n_starts:
            logger.warning(
                "Удалось сгенерировать только %d из %d стартовых точек",
                len(starts), self.n_starts,
            )

        return np.array(starts)


    def find_modes(
        self,
        starts: Optional[np.ndarray] = None,
        seed: int = 0,
        verbose: bool = True,
    ) -> ModeFinderResult:
        self.model.enable_nfe_counting()
        self.model.reset_nfe()

        if starts is None:
            starts = self._generate_starts(seed)

        if starts.ndim == 1 and self.dim == 1:
            starts = starts.reshape(-1, 1)

        all_modes: List[np.ndarray] = []
        total_nfe = 0

        for i, start in enumerate(starts):
            self.model.reset_nfe()
            modes_from_start = self._run_single_start(start, verbose)
            all_modes.extend(modes_from_start)
            nfe_i = self.model.nfe
            total_nfe += nfe_i

            if verbose:
                logger.info(
                    "  Старт %d/%d: найдено %d мод, NFE=%d",
                    i + 1, len(starts), len(modes_from_start), nfe_i,
                )

        final_modes = self._cluster(all_modes)

        if self.dim == 1:
            final_modes = final_modes.flatten()

        result = ModeFinderResult(
            modes=final_modes,
            nfe=total_nfe,
            history=[],
            starts=starts,
        )

        logger.info(
            "Поиск завершён: %d мод, NFE=%d",
            len(final_modes), total_nfe,
        )
        return result

    def _run_single_start(self, x0: np.ndarray,
                          verbose: bool) -> List[np.ndarray]:

        active = [x0.copy()]

        for t in self.timesteps:
            t = int(t)

            after_ascent = [
                self._gradient_ascent(x, t, self.step_size, self.ascent_steps)
                for x in active
            ]

            after_split: List[np.ndarray] = []
            for x in after_ascent:
                split_result = self._check_split(x, t)
                after_split.extend(split_result)

            clustered = self._cluster(after_split)
            active = [clustered[i] for i in range(len(clustered))]

            if verbose:
                sigma = self.model.schedule.sigma(t)
                logger.debug(
                    "  t=%d (σ=%.3f): %d активных мод", t, sigma, len(active)
                )

        alpha_refine = self.step_size * self.refine_step_scale
        finals = [
            self._gradient_ascent(x, 0, alpha_refine, self.refine_steps)
            for x in active
        ]

        return finals
