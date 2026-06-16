from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from ..models.diffusion import DiffusionModel
from .clustering import merge_close, agglomerative_merge
from .mode_finder_v2 import ddim_sample, ModeFinderResultV2

logger = logging.getLogger(__name__)


class ModeFinderV3F3:

    def __init__(
        self,
        model: DiffusionModel,
        t_start: int = 800,
        t_end: int = 0,
        ode_steps_coarse: int = 15,
        ode_steps_fine: int = 5,
        use_adaptive_step: bool = True,
        trace_stability_threshold: float = 0.3,
        n_substeps_per_interval: int = 3,
        n_particles: int = 50,
        cluster_every: int = 0,
        merge_factor: float = 0.8,
        merge_radius_min: float = 0.05,
        n_trace_probe: int = 8,
        hessian_fd_eps: float = 1e-3,
        refine_steps: int = 200,
        refine_alpha: float = 0.001,
        ddim_steps: int = 50,
        x_min: float = -10.0,
        x_max: float = 10.0,
        **_ignored,
    ):
        self.model = model
        self.dim = model.dim
        self.device = model.device

        self.t_start = t_start
        self.t_end = t_end
        self.ode_steps_coarse = ode_steps_coarse
        self.ode_steps_fine = ode_steps_fine
        self.use_adaptive_step = use_adaptive_step
        self.trace_stability_threshold = trace_stability_threshold
        self.n_substeps_per_interval = n_substeps_per_interval

        self.n_particles = n_particles
        self.cluster_every = cluster_every

        self.merge_factor = merge_factor
        self.merge_radius_min = merge_radius_min

        self.n_trace_probe = n_trace_probe
        self.hessian_fd_eps = hessian_fd_eps

        self.refine_steps = refine_steps
        self.refine_alpha = refine_alpha

        self.ddim_steps = ddim_steps
        self.x_min = x_min
        self.x_max = x_max


    def _beta_scalar(self, t: int) -> float:
        t = int(np.clip(t, 0, self.model.schedule.T - 1))
        return float(self.model.schedule.betas[t].item())

    def _score_batch(self, particles: np.ndarray, t: int) -> np.ndarray:
        return np.array([
            self.model.score_numpy(particles[i], t).flatten()
            for i in range(len(particles))
        ])  # (N, dim)

    def _vector_field_batch(self, particles: np.ndarray, t: int) -> np.ndarray:
        beta_t = self._beta_scalar(t)
        scores = self._score_batch(particles, t)   # (N, dim)
        return -0.5 * beta_t * (particles + scores)


    def _ode_interval_step(
        self, particles: np.ndarray, t_curr: int, t_next: int,
    ) -> np.ndarray:
        particles = particles.copy()

        gap = abs(t_curr - t_next)
        n_sub = max(1, min(self.n_substeps_per_interval, gap))
        dt = float(t_next - t_curr) / n_sub   # < 0

        t_float = float(t_curr)

        for _ in range(n_sub):
            t_eval = int(np.clip(round(t_float), self.t_end,
                                 self.model.schedule.T - 1))
            v = self._vector_field_batch(particles, t_eval)
            particles = particles + dt * v
            particles = np.clip(particles, self.x_min, self.x_max)
            t_float += dt

        return particles


    def _compute_trace_hessian(self, x: np.ndarray, t: int) -> float:
        eps = self.hessian_fd_eps
        trace = 0.0
        for j in range(self.dim):
            e_j = np.zeros(self.dim)
            e_j[j] = eps
            s_plus = self.model.score_numpy(x + e_j, t).flatten()
            s_minus = self.model.score_numpy(x - e_j, t).flatten()
            trace += (s_plus[j] - s_minus[j]) / (2 * eps)
        return trace

    def _trace_statistic(self, particles: np.ndarray, t: int) -> float:
        n = len(particles)
        n_probe = min(self.n_trace_probe, n)
        idx = np.round(np.linspace(0, n - 1, n_probe)).astype(int)
        probe = particles[idx]

        vals = [
            abs(self._compute_trace_hessian(x, t)) / max(self.dim, 1)
            for x in probe
        ]
        return float(np.median(vals))


    def _build_timesteps(self, particles: np.ndarray) -> List[int]:
        coarse = np.linspace(self.t_start, self.t_end,
                             self.ode_steps_coarse + 1, dtype=int)
        coarse = list(np.unique(coarse)[::-1])  # убывание

        if not self.use_adaptive_step:
            return coarse

        result: List[int] = []

        for i in range(len(coarse) - 1):
            t_curr = int(coarse[i])
            t_next = int(coarse[i + 1])
            result.append(t_curr)

            if t_curr <= self.t_end:
                continue

            stat = self._trace_statistic(particles, t_curr)

            if stat < self.trace_stability_threshold:
                max_extra = max(0, abs(t_curr - t_next) - 1)
                n_extra = max(1, min(self.ode_steps_fine, max_extra))
                extra = np.linspace(t_curr, t_next, n_extra + 2, dtype=int)[1:-1]
                for t_ex in extra:
                    t_ex = int(t_ex)
                    if t_ex != t_curr and t_ex != t_next:
                        result.append(t_ex)
                logger.debug(
                    "  Adaptive: t=%d, median|tr|/d=%.4f < %.4f → +%d pts",
                    t_curr, stat, self.trace_stability_threshold, len(extra),
                )

        result.append(int(coarse[-1]))
        return sorted(set(result), reverse=True)


    def _get_merge_radius(self, t: int) -> float:
        sigma_t = self.model.schedule.sigma(t) if t > 0 else 0.01
        return max(self.merge_radius_min, self.merge_factor * sigma_t)

    def _cluster_particles(self, particles: np.ndarray, merge_r: float) -> np.ndarray:
        if len(particles) == 0:
            return np.empty((0, self.dim))
        if self.dim == 1:
            centers = merge_close(particles.flatten(), merge_r)
            return centers.reshape(-1, 1)
        else:
            return agglomerative_merge(particles, merge_r)


    def _init_particles(self, seed: int) -> Tuple[np.ndarray, int]:
        self.model.enable_nfe_counting()
        self.model.reset_nfe()

        samples = ddim_sample(
            self.model, self.n_particles,
            num_steps=self.ddim_steps,
            stop_t=self.t_start,   # ← ключевое отличие от F2
            seed=seed,
        )
        nfe_init = self.model.nfe

        samples_2d = np.atleast_2d(samples)
        if self.dim == 1 and samples_2d.shape[1] != 1:
            samples_2d = samples_2d.reshape(-1, 1)

        logger.info(
            "V3F3 init: %d particles at t=%d (NFE=%d)",
            len(samples_2d), self.t_start, nfe_init,
        )
        return samples_2d, nfe_init


    def _final_refine(self, particles: np.ndarray) -> np.ndarray:
        particles = particles.copy()
        for _ in range(self.refine_steps):
            scores = self._score_batch(particles, t=0)
            particles = np.clip(
                particles + self.refine_alpha * scores,
                self.x_min, self.x_max,
            )
        return particles


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

        nfe_checkpoint = self.model.nfe  # = nfe_init

        timesteps = self._build_timesteps(particles)
        nfe_grid = self.model.nfe - nfe_checkpoint

        if verbose:
            logger.info(
                "V3F3: %d particles | ODE grid=%d steps | NFE(grid)=%d",
                len(particles), len(timesteps) - 1, nfe_grid,
            )

        nfe_checkpoint2 = self.model.nfe

        for step_idx, (t_curr, t_next) in enumerate(
            zip(timesteps[:-1], timesteps[1:])
        ):
            t_curr, t_next = int(t_curr), int(t_next)

            particles = self._ode_interval_step(particles, t_curr, t_next)

            if self.cluster_every and self.cluster_every > 0:
                if (step_idx + 1) % self.cluster_every == 0 and len(particles) > 1:
                    merge_r = self._get_merge_radius(t_next)
                    particles = self._cluster_particles(particles, merge_r)
                    if len(particles) == 0:
                        break

            if verbose:
                sigma = self.model.schedule.sigma(t_next) if t_next > 0 else 0.0
                logger.debug(
                    "    t: %d→%d (σ=%.3f): %d particles",
                    t_curr, t_next, sigma, len(particles),
                )

        nfe_tracking = self.model.nfe - nfe_checkpoint2

        nfe_checkpoint3 = self.model.nfe
        final_particles = self._final_refine(particles)
        nfe_refine = self.model.nfe - nfe_checkpoint3

        final_modes_arr = self._cluster_particles(
            final_particles, self.merge_radius_min,
        )
        if self.dim == 1:
            final_modes = final_modes_arr.flatten()
        else:
            final_modes = final_modes_arr

        total_nfe = self.model.nfe

        if verbose:
            n_modes = len(final_modes) if final_modes.ndim > 0 else 0
            logger.info(
                "V3F3 done: %d modes | NFE=%d (init=%d, grid=%d, tracking=%d, refine=%d)",
                n_modes, total_nfe, nfe_init, nfe_grid, nfe_tracking, nfe_refine,
            )

        return ModeFinderResultV2(
            modes=final_modes,
            nfe=total_nfe,
            nfe_starts=nfe_init,
            nfe_search=nfe_tracking,
            history=[],
            starts=particles,
        )
