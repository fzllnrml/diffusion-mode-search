from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import torch

from ..models.diffusion import DiffusionModel
from .mode_finder import ModeFinder, ModeFinderResult
from .clustering import merge_close, agglomerative_merge

logger = logging.getLogger(__name__)


@dataclass
class AutoStartResult:
    modes: np.ndarray
    nfe: int
    nfe_overhead: int
    nfe_search: int
    n_starts_chosen: int
    starts: np.ndarray
    strategy_name: str
    extra: Dict[str, Any] = None  


class StartStrategy(ABC):

    @abstractmethod
    def find_modes(
        self,
        model: DiffusionModel,
        mf_kwargs: dict,
        seed: int = 0,
    ) -> AutoStartResult:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class FixedStarts(StartStrategy):

    def __init__(self, n_starts: int = 3):
        self.n_starts = n_starts

    @property
    def name(self) -> str:
        return f"fixed_{self.n_starts}"

    def find_modes(self, model, mf_kwargs, seed=0):
        mf_kwargs = {**mf_kwargs, "n_starts": self.n_starts}
        finder = ModeFinder(model=model, **mf_kwargs)
        result = finder.find_modes(seed=seed, verbose=False)

        return AutoStartResult(
            modes=result.modes,
            nfe=result.nfe,
            nfe_overhead=0,
            nfe_search=result.nfe,
            n_starts_chosen=self.n_starts,
            starts=result.starts,
            strategy_name=self.name,
        )


class PilotSampleStarts(StartStrategy):

    def __init__(
        self,
        n_pilot: int = 20,
        cluster_radius: float = 1.5,
        min_starts: int = 1,
    ):
        self.n_pilot = n_pilot
        self.cluster_radius = cluster_radius
        self.min_starts = min_starts

    @property
    def name(self) -> str:
        return f"pilot_{self.n_pilot}"

    def find_modes(self, model, mf_kwargs, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)

        pilot_samples = model.sample(self.n_pilot)
        nfe_pilot = self.n_pilot * getattr(model, "T", 1000)

        dim = model.dim
        if dim == 1:
            centers = merge_close(pilot_samples.flatten(), self.cluster_radius)
            if centers.ndim == 1:
                centers = centers.reshape(-1, 1)
        else:
            if pilot_samples.ndim == 1:
                pilot_samples = pilot_samples.reshape(-1, dim)
            centers = agglomerative_merge(pilot_samples, self.cluster_radius)

        n_clusters = max(len(centers), self.min_starts)
        logger.info(
            "PilotSample: %d пилотных сэмплов → %d кластеров (nfe_overhead=%d)",
            self.n_pilot, n_clusters, nfe_pilot,
        )

        starts = centers[:n_clusters]
        if dim == 1:
            starts = starts.flatten()

        mf_kwargs_local = {**mf_kwargs, "n_starts": n_clusters}
        finder = ModeFinder(model=model, **mf_kwargs_local)
        result = finder.find_modes(starts=starts, seed=seed, verbose=False)

        return AutoStartResult(
            modes=result.modes,
            nfe=nfe_pilot + result.nfe,
            nfe_overhead=nfe_pilot,
            nfe_search=result.nfe,
            n_starts_chosen=n_clusters,
            starts=starts if starts.ndim > 1 else starts.reshape(-1, 1),
            strategy_name=self.name,
            extra={"pilot_samples": pilot_samples, "cluster_centers": centers},
        )


class IncrementalStarts(StartStrategy):

    def __init__(
        self,
        max_rounds: int = 15,
        patience: int = 3,
        merge_radius: float = 0.5,
    ):
        self.max_rounds = max_rounds
        self.patience = patience
        self.merge_radius = merge_radius

    @property
    def name(self) -> str:
        return f"incremental_p{self.patience}"

    def _count_new_modes(self, existing: np.ndarray, candidates: np.ndarray) -> int:
        if len(existing) == 0:
            return len(candidates)
        if len(candidates) == 0:
            return 0

        existing_2d = np.atleast_2d(existing)
        candidates_2d = np.atleast_2d(candidates)

        from scipy.spatial.distance import cdist
        dists = cdist(candidates_2d, existing_2d)
        min_dists = dists.min(axis=1)
        return int((min_dists > self.merge_radius).sum())

    def _merge_mode_sets(self, a: np.ndarray, b: np.ndarray, dim: int) -> np.ndarray:
        if len(a) == 0:
            return b
        if len(b) == 0:
            return a

        combined = np.concatenate([np.atleast_2d(a), np.atleast_2d(b)], axis=0)

        if dim == 1:
            merged = merge_close(combined.flatten(), self.merge_radius)
            return merged.reshape(-1, 1)
        else:
            return agglomerative_merge(combined, self.merge_radius)

    def find_modes(self, model, mf_kwargs, seed=0):
        dim = model.dim
        x_min = mf_kwargs.get("x_min", -10.0)
        x_max = mf_kwargs.get("x_max", 10.0)
        rng = np.random.default_rng(seed)

        all_modes = np.empty((0, dim))
        total_nfe = 0
        patience_counter = 0
        rounds_used = 0
        all_starts = []

        for round_idx in range(self.max_rounds):
            start = rng.uniform(x_min, x_max, size=dim)
            all_starts.append(start.copy())

            start_for_mf = start.reshape(1, -1)
            if dim == 1:
                start_for_mf = start_for_mf.flatten()

            mf_kwargs_local = {**mf_kwargs, "n_starts": 1}
            finder = ModeFinder(model=model, **mf_kwargs_local)
            result = finder.find_modes(
                starts=start_for_mf, seed=seed + round_idx, verbose=False,
            )
            total_nfe += result.nfe
            rounds_used = round_idx + 1

            new_modes = np.atleast_2d(result.modes.reshape(-1, dim))

            n_new = self._count_new_modes(all_modes, new_modes)

            if n_new > 0:
                all_modes = self._merge_mode_sets(all_modes, new_modes, dim)
                patience_counter = 0
                logger.debug(
                    "Incremental раунд %d: +%d новых мод (всего %d)",
                    round_idx + 1, n_new, len(all_modes),
                )
            else:
                patience_counter += 1
                logger.debug(
                    "Incremental раунд %d: нет новых мод (patience %d/%d)",
                    round_idx + 1, patience_counter, self.patience,
                )

            if patience_counter >= self.patience:
                logger.info(
                    "Incremental: конвергенция после %d раундов (%d мод, NFE=%d)",
                    rounds_used, len(all_modes), total_nfe,
                )
                break

        final_modes = all_modes.squeeze() if dim == 1 else all_modes
        starts_arr = np.array(all_starts)

        return AutoStartResult(
            modes=final_modes,
            nfe=total_nfe,
            nfe_overhead=0,  # нет отдельного overhead, всё — поиск
            nfe_search=total_nfe,
            n_starts_chosen=rounds_used,
            starts=starts_arr,
            strategy_name=self.name,
            extra={"rounds_used": rounds_used, "patience_at_end": patience_counter},
        )


class ScoreGridStarts(StartStrategy):

    def __init__(
        self,
        grid_size: int = 50,
        t_scan: int = 300,
        scan_ascent_steps: int = 30,
        scan_step_size: float = 0.05,
        cluster_radius: float = 1.0,
        min_starts: int = 1,
    ):
        self.grid_size = grid_size
        self.t_scan = t_scan
        self.scan_ascent_steps = scan_ascent_steps
        self.scan_step_size = scan_step_size
        self.cluster_radius = cluster_radius
        self.min_starts = min_starts

    @property
    def name(self) -> str:
        return f"score_grid_{self.grid_size}"

    def _create_grid(self, dim: int, x_min: float, x_max: float) -> np.ndarray:
        if dim == 1:
            return np.linspace(x_min, x_max, self.grid_size).reshape(-1, 1)
        elif dim == 2:
            n_per_axis = max(2, int(np.sqrt(self.grid_size)))
            xs = np.linspace(x_min, x_max, n_per_axis)
            ys = np.linspace(x_min, x_max, n_per_axis)
            xx, yy = np.meshgrid(xs, ys)
            return np.column_stack([xx.ravel(), yy.ravel()])
        else:
            rng = np.random.default_rng(42)
            return rng.uniform(x_min, x_max, size=(self.grid_size, dim))

    def find_modes(self, model, mf_kwargs, seed=0):
        dim = model.dim
        x_min = mf_kwargs.get("x_min", -10.0)
        x_max = mf_kwargs.get("x_max", 10.0)

        grid = self._create_grid(dim, x_min, x_max)

        model.enable_nfe_counting()
        model.reset_nfe()

        x = torch.tensor(grid, device=model.device, dtype=torch.float32)
        n = x.shape[0]

        with torch.no_grad():
            for _ in range(self.scan_ascent_steps):
                t_tensor = torch.full(
                    (n,), self.t_scan, device=model.device, dtype=torch.long,
                )
                score = model.score(x, self.t_scan)
                x = x + self.scan_step_size * score
                x = x.clamp(x_min, x_max)

        converged = x.cpu().numpy()
        nfe_scan = model.nfe

        if dim == 1:
            centers = merge_close(converged.flatten(), self.cluster_radius)
            centers = centers.reshape(-1, 1)
        else:
            centers = agglomerative_merge(converged, self.cluster_radius)

        n_clusters = max(len(centers), self.min_starts)
        starts = centers[:n_clusters]

        logger.info(
            "ScoreGrid: %d точек сетки → %d аттракторов при t=%d (nfe_overhead=%d)",
            len(grid), n_clusters, self.t_scan, nfe_scan,
        )

        if dim == 1:
            starts_for_mf = starts.flatten()
        else:
            starts_for_mf = starts

        mf_kwargs_local = {**mf_kwargs, "n_starts": n_clusters}
        finder = ModeFinder(model=model, **mf_kwargs_local)
        result = finder.find_modes(starts=starts_for_mf, seed=seed, verbose=False)

        return AutoStartResult(
            modes=result.modes,
            nfe=nfe_scan + result.nfe,
            nfe_overhead=nfe_scan,
            nfe_search=result.nfe,
            n_starts_chosen=n_clusters,
            starts=starts,
            strategy_name=self.name,
            extra={
                "grid_points": grid,
                "converged_points": converged,
                "t_scan": self.t_scan,
            },
        )


def get_default_strategies() -> List[StartStrategy]:
    return [
        FixedStarts(n_starts=1),
        FixedStarts(n_starts=3),
        FixedStarts(n_starts=5),
        FixedStarts(n_starts=10),
        PilotSampleStarts(n_pilot=20, cluster_radius=1.5),
        IncrementalStarts(max_rounds=15, patience=3, merge_radius=0.5),
        ScoreGridStarts(grid_size=50, t_scan=300, scan_ascent_steps=30),
    ]


def get_adaptive_strategies() -> List[StartStrategy]:
    return [
        PilotSampleStarts(n_pilot=20, cluster_radius=1.5),
        IncrementalStarts(max_rounds=15, patience=3, merge_radius=0.5),
        ScoreGridStarts(grid_size=50, t_scan=300, scan_ascent_steps=30),
    ]
