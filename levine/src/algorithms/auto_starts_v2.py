from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import numpy as np
import torch

from ..models.diffusion import DiffusionModel
from .auto_starts import StartStrategy, AutoStartResult
from .mode_finder_v2 import ModeFinderV2, CoarseToFineFinder

logger = logging.getLogger(__name__)


class SmartFixedV2(StartStrategy):

    def __init__(
        self,
        n_starts: int = 5,
        split_method: str = "hessian",
        normalize_score: bool = True,
        adaptive_merge: bool = True,
        start_method: str = "smart",
        ddim_steps: int = 50,
    ):
        self.n_starts = n_starts
        self.split_method = split_method
        self.normalize_score = normalize_score
        self.adaptive_merge = adaptive_merge
        self.start_method = start_method
        self.ddim_steps = ddim_steps

    @property
    def name(self) -> str:
        tag = "smart" if self.start_method == "smart" else "unif"
        sp = "H" if self.split_method == "hessian" else "A"
        return f"v2_{tag}{self.n_starts}_{sp}"

    def find_modes(self, model, mf_kwargs, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)

        v2_kwargs = {
            k: v for k, v in mf_kwargs.items()
            if k not in ("n_starts", "split_directions")
        }
        v2_kwargs.update({
            "n_starts": self.n_starts,
            "split_method": self.split_method,
            "normalize_score": self.normalize_score,
            "adaptive_merge": self.adaptive_merge,
            "start_method": self.start_method,
            "ddim_steps": self.ddim_steps,
            "max_active_per_start": mf_kwargs.get("max_active_per_start", 30),
        })

        finder = ModeFinderV2(model=model, **v2_kwargs)
        result = finder.find_modes(seed=seed, verbose=False)

        return AutoStartResult(
            modes=result.modes,
            nfe=result.nfe,
            nfe_overhead=result.nfe_starts,
            nfe_search=result.nfe_search,
            n_starts_chosen=self.n_starts,
            starts=result.starts,
            strategy_name=self.name,
            extra={"method": "ModeFinderV2"},
        )


class CoarseToFineStrategy(StartStrategy):

    def __init__(
        self,
        n_samples: int = 200,
        ddim_steps: int = 50,
        cluster_bandwidth: float = 2.0,
        refine_steps: int = 200,
        refine_alpha: float = 0.001,
    ):
        self.n_samples = n_samples
        self.ddim_steps = ddim_steps
        self.cluster_bandwidth = cluster_bandwidth
        self.refine_steps = refine_steps
        self.refine_alpha = refine_alpha

    @property
    def name(self) -> str:
        return f"c2f_{self.n_samples}"

    def find_modes(self, model, mf_kwargs, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)

        x_min = mf_kwargs.get("x_min", -10.0)
        x_max = mf_kwargs.get("x_max", 10.0)
        merge_radius = mf_kwargs.get("merge_radius", 0.5)

        ctf = CoarseToFineFinder(
            model=model,
            n_samples=self.n_samples,
            ddim_steps=self.ddim_steps,
            cluster_method="agglomerative",
            cluster_bandwidth=self.cluster_bandwidth,
            refine_steps=self.refine_steps,
            refine_alpha=self.refine_alpha,
            merge_radius=merge_radius,
            normalize_score=True,
            x_min=x_min,
            x_max=x_max,
        )
        result = ctf.find_modes(seed=seed, verbose=False)

        n_modes = len(result.modes) if result.modes.ndim > 0 else 0

        return AutoStartResult(
            modes=result.modes,
            nfe=result.nfe,
            nfe_overhead=result.nfe_starts,
            nfe_search=result.nfe_search,
            n_starts_chosen=n_modes,
            starts=result.starts,
            strategy_name=self.name,
            extra={"method": "CoarseToFine"},
        )


def get_v2_strategies():
    return [
        SmartFixedV2(n_starts=3, split_method="hessian", start_method="smart"),
        SmartFixedV2(n_starts=5, split_method="hessian", start_method="smart"),
        SmartFixedV2(n_starts=10, split_method="hessian", start_method="smart"),
        CoarseToFineStrategy(n_samples=100, ddim_steps=50),
        CoarseToFineStrategy(n_samples=200, ddim_steps=50),
    ]
