from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from ..models.diffusion import DiffusionModel
from .clustering import merge_close, agglomerative_merge

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    modes: np.ndarray
    nfe: int
    raw_samples: np.ndarray


class BaselineModeFinder:
    def __init__(
        self,
        model: DiffusionModel,
        n_samples: int = 1000,
        refine_steps: int = 10,
        refine_alpha: float = 0.01,
        merge_radius: float = 0.25,
    ):
        self.model = model
        self.n_samples = n_samples
        self.refine_steps = refine_steps
        self.refine_alpha = refine_alpha
        self.merge_radius = merge_radius

    def find_modes(self):
        self.model.enable_nfe_counting()
        self.model.reset_nfe()

        logger.info(
            "Baseline. Генерация %d сэмплов T=%d шагов",
            self.n_samples, self.model.schedule.T,
        )
        raw_samples = self._sample_reverse(self.n_samples)
        nfe_sampling = self.model.nfe

        if self.refine_steps > 0:
            logger.info("Baseline. Градиентный подьем в %d шагов", self.refine_steps)
            refined = self._refine_samples(raw_samples)
        else:
            refined = raw_samples

        nfe_total = self.model.nfe

        modes = self._cluster(refined)

        logger.info(
            "Baseline результат: %d мод, NFE=%d. Было sampling=%d,  с градиентным подьемом в %d)",
            len(modes), nfe_total, nfe_sampling, nfe_total - nfe_sampling,
        )

        return BaselineResult(
            modes=modes,
            nfe=nfe_total,
            raw_samples=raw_samples,
        )

    def _sample_reverse(self, n):
        self.model.net.eval()
        s = self.model.schedule
        device = self.model.device
        dim = self.model.dim

        x = torch.randn(n, dim, device=device)

        with torch.no_grad():
            for t_step in reversed(range(s.T)):
                t = torch.full((n,), t_step, device=device, dtype=torch.long)

                if self.model._nfe_counter is not None:
                    eps_pred = self.model._nfe_counter(x, t)
                else:
                    eps_pred = self.model.net(x, t)

                beta_t = s.extract(s.betas, t, x.shape)
                sqrt_omc = s.extract(s.sqrt_one_minus_alphas_cumprod, t, x.shape)
                sqrt_recip = s.extract(s.sqrt_recip_alphas, t, x.shape)

                mean = sqrt_recip * (x - beta_t / sqrt_omc * eps_pred)

                if t_step > 0:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt(beta_t)
                    x = mean + sigma * noise
                else:
                    x = mean

        result = x.cpu().numpy()
        return result.squeeze(-1) if dim == 1 else result

    def _refine_samples(self, samples):
        refined = []
        for sample in samples:
            x = sample.copy()
            for _ in range(self.refine_steps):
                score = self.model.score_numpy(x, t=0)
                x = x + self.refine_alpha * score.flatten()
            refined.append(x)
        return np.array(refined)


    def _cluster(self, points):
        if points.ndim == 1:
            return merge_close(points, self.merge_radius)
        else:
            return agglomerative_merge(points, self.merge_radius)
