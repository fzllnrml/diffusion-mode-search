from __future__ import annotations

import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


class GaussianMixture:
    def __init__(self, mus, sigmas, weights=None):
        self.mus = np.asarray(mus, dtype=np.float64)
        self.sigmas = np.asarray(sigmas, dtype=np.float64)

        if self.mus.ndim == 1:
            self.dim = 1
            self.mus = self.mus.reshape(-1, 1)
        else:
            self.dim = self.mus.shape[1]

        self.K = self.mus.shape[0]

        if weights is None:
            self.weights = np.ones(self.K) / self.K
        else:
            self.weights = np.asarray(weights, dtype=np.float64)
            self.weights = self.weights / self.weights.sum()

        logger.debug(
            "GaussianMixture: K=%d, dim=%d, mus=%s",
            self.K, self.dim, self.mus.flatten(),
        )

    @property
    def mode_locations(self):
        return self.mus.squeeze() if self.dim == 1 else self.mus

    @property
    def min_separation(self):
        if self.K <= 1:
            return float("inf")
        from scipy.spatial.distance import pdist
        return float(pdist(self.mus).min())

    def density(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))

        result = np.zeros(x.shape[0])
        for k in range(self.K):
            diff = x - self.mus[k]
            sq_dist = np.sum(diff ** 2, axis=1)
            coeff = 1.0 / ((2 * np.pi * self.sigmas[k] ** 2) ** (self.dim / 2))
            result += self.weights[k] * coeff * np.exp(-sq_dist / (2 * self.sigmas[k] ** 2))

        return result

    def score_numpy(self, x):
        squeeze = x.ndim == 1 and self.dim == 1
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))

        log_probs = np.zeros((x.shape[0], self.K))
        for k in range(self.K):
            diff = x - self.mus[k]
            sq_dist = np.sum(diff ** 2, axis=1)
            log_probs[:, k] = (
                np.log(self.weights[k] + 1e-30)
                - 0.5 * self.dim * np.log(2 * np.pi * self.sigmas[k] ** 2)
                - sq_dist / (2 * self.sigmas[k] ** 2)
            )

        log_probs_max = log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs - log_probs_max)
        resp = probs / probs.sum(axis=1, keepdims=True)

        score = np.zeros_like(x)
        for k in range(self.K):
            diff = self.mus[k] - x
            score += resp[:, k:k+1] * diff / (self.sigmas[k] ** 2)

        return score.squeeze() if squeeze else score

    def sample_numpy(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        components = rng.choice(self.K, size=n, p=self.weights)

        samples = np.zeros((n, self.dim))
        for k in range(self.K):
            mask = components == k
            count = mask.sum()
            if count > 0:
                samples[mask] = (
                    self.mus[k]
                    + self.sigmas[k] * rng.standard_normal((count, self.dim))
                )

        return samples.squeeze(-1) if self.dim == 1 else samples

    def sample_torch(self, n, device):
        import torch
        mus_t = torch.tensor(self.mus, device=device, dtype=torch.float32)
        sigmas_t = torch.tensor(self.sigmas, device=device, dtype=torch.float32)
        weights_t = torch.tensor(self.weights, device=device, dtype=torch.float32)

        comp_idx = torch.multinomial(weights_t, n, replacement=True)
        z = torch.randn(n, self.dim, device=device)
        mu = mus_t[comp_idx]
        sigma = sigmas_t[comp_idx].unsqueeze(-1)

        return z * sigma + mu

    @classmethod
    def random(cls, K, dim=1, sigma_range=(0.5, 1.2), min_sep=2.0, bounds=(-10.0, 10.0), seed=0):
        rng = np.random.default_rng(seed)

        mus = []
        for _ in range(200_000):
            if len(mus) == K:
                break
            cand = rng.uniform(bounds[0], bounds[1], size=dim)
            if all(np.linalg.norm(cand - m) >= min_sep for m in mus):
                mus.append(cand)

        if len(mus) < K:
            raise RuntimeError(
                f"Не удалось разместить {K} мод на расстоянии {min_sep} "
                f"в {bounds}"
            )

        mus = np.array(mus)
        sigmas = rng.uniform(sigma_range[0], sigma_range[1], size=K)
        weights = rng.dirichlet(np.ones(K))

        return cls(mus=mus, sigmas=sigmas, weights=weights)
