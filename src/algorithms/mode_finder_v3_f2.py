from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..models.diffusion import DiffusionModel
from .clustering import merge_close, agglomerative_merge
from .mode_finder_v2 import ddim_sample, ModeFinderResultV2

logger = logging.getLogger(__name__)


class ModeFinderV3F2:

    def __init__(
        self,
        model: DiffusionModel,
        timesteps: Optional[List[int]] = None,
        step_size: float = 0.01,
        n_particles: int = 50,
        merge_factor: float = 0.8,
        merge_radius_min: float = 0.05,
        ascent_steps: int = 20,
        normalize_score: bool = True,
        refine_steps: int = 100,
        refine_step_scale: float = 0.1,
        ddim_steps: int = 50,
        init_stop_t: int = 0,          # до какого t делаем DDIM для старта
        x_min: float = -10.0,
        x_max: float = 10.0,
        **_ignored,
    ):
        self.model = model
        self.dim = model.dim
        self.device = model.device

        self.timesteps = timesteps or self._default_timesteps()
        self.step_size = step_size

        self.n_particles = n_particles
        self.merge_factor = merge_factor
        self.merge_radius_min = merge_radius_min

        self.ascent_steps = ascent_steps
        self.normalize_score = normalize_score

        self.refine_steps = refine_steps
        self.refine_step_scale = refine_step_scale

        self.ddim_steps = ddim_steps
        self.init_stop_t = init_stop_t

        self.x_min = x_min
        self.x_max = x_max

    def _default_timesteps(self) -> List[int]:
        if self.dim <= 2:
            return [800, 500, 300, 200, 150, 100, 80, 60, 40, 30, 20, 10, 5, 0]
        elif self.dim <= 10:
            return [800, 600, 400, 300, 200, 150, 100, 70, 50, 30, 15, 5, 0]
        else:
            return [800, 500, 300, 200, 100, 50, 20, 5, 0]


    def _gradient_ascent_batch(
        self, particles: np.ndarray, t: int, alpha: float,
        num_steps: int, tol: float = 1e-6,
    ) -> np.ndarray:
        particles = particles.copy()
        for _ in range(num_steps):
            scores = np.array([
                self.model.score_numpy(particles[i], t).flatten()
                for i in range(len(particles))
            ])  # (N, dim)

            if self.normalize_score:
                norms = np.linalg.norm(scores, axis=1, keepdims=True)
                norms = np.where(norms > 1e-8, norms, 1.0)
                scores = scores / norms

            particles_new = particles + alpha * scores
            particles_new = np.clip(particles_new, self.x_min, self.x_max)

            max_shift = np.max(np.linalg.norm(particles_new - particles, axis=1))
            particles = particles_new
            if max_shift < tol:
                break

        return particles


    def _get_merge_radius(self, t: int) -> float:
        sigma_t = self.model.schedule.sigma(t) if t > 0 else 0.01
        return max(self.merge_radius_min, self.merge_factor * sigma_t)

    def _cluster_particles(self, particles: np.ndarray, merge_r: float) -> np.ndarray:
        if len(particles) == 0:
            return np.empty((0, self.dim))

        if self.dim == 1:
            flat = particles.flatten()
            centers = merge_close(flat, merge_r)
            return centers.reshape(-1, 1)
        else:
            return agglomerative_merge(particles, merge_r)


    def _init_particles(self, seed: int) -> Tuple[np.ndarray, int]:
        self.model.enable_nfe_counting()
        self.model.reset_nfe()

        samples = ddim_sample(
            self.model, self.n_particles,
            num_steps=self.ddim_steps, stop_t=self.init_stop_t, seed=seed,
        )
        nfe_init = self.model.nfe

        samples_2d = np.atleast_2d(samples)
        if self.dim == 1 and samples_2d.shape[1] != 1:
            samples_2d = samples_2d.reshape(-1, 1)

        return samples_2d, nfe_init


    def find_modes(
        self,
        starts: Optional[np.ndarray] = None,
        seed: int = 0,
        verbose: bool = True,
    ) -> ModeFinderResultV2:
        nfe_init = 0

        if starts is not None:
            particles = np.atleast_2d(starts)
            if self.dim == 1 and particles.shape[1] != 1:
                particles = particles.reshape(-1, 1)
            self.model.enable_nfe_counting()
            self.model.reset_nfe()
        else:
            particles, nfe_init = self._init_particles(seed)

        if verbose:
            logger.info(
                "V3F2: %d particles initialized (NFE=%d)", len(particles), nfe_init
            )

        self.model.reset_nfe()

        for t in self.timesteps:
            t = int(t)
            merge_r = self._get_merge_radius(t)

            particles = self._gradient_ascent_batch(
                particles, t, self.step_size, self.ascent_steps,
            )

            clustered = self._cluster_particles(particles, merge_r)

            if len(clustered) == 0:
                break

            particles = clustered

            sigma = self.model.schedule.sigma(t) if t > 0 else 0.0
            logger.debug(
                "    t=%d (sigma=%.3f): %d clusters (merge_r=%.3f)",
                t, sigma, len(particles), merge_r,
            )

        nfe_search = self.model.nfe

        alpha_refine = self.step_size * self.refine_step_scale
        final_particles = self._gradient_ascent_batch(
            particles, 0, alpha_refine, self.refine_steps,
        )

        final_modes_arr = self._cluster_particles(final_particles, self.merge_radius_min)

        if self.dim == 1:
            final_modes = final_modes_arr.flatten()
        else:
            final_modes = final_modes_arr

        total_nfe = nfe_init + nfe_search + self.model.nfe - nfe_search
        total_nfe = nfe_init + self.model.nfe

        if verbose:
            n_modes = len(final_modes) if final_modes.ndim > 0 else 0
            logger.info(
                "V3F2 done: %d modes, NFE=%d (init=%d, search=%d)",
                n_modes, total_nfe, nfe_init, self.model.nfe,
            )

        return ModeFinderResultV2(
            modes=final_modes,
            nfe=total_nfe,
            nfe_starts=nfe_init,
            nfe_search=self.model.nfe,
            history=[],
            starts=particles,
        )
