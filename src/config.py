from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DistributionConfig:
    mus: List[float] = field(default_factory=lambda: [-4.0, 0.0, 3.0, 8.0])
    sigmas: List[float] = field(default_factory=lambda: [0.8, 1.0, 0.6, 0.7])
    weights: List[float] = field(default_factory=lambda: [0.25, 0.30, 0.25, 0.20])
    x_min: float = -10.0
    x_max: float = 10.0


@dataclass
class ModelConfig:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "silu"


@dataclass
class TrainingConfig:
    num_steps: int = 200_000
    batch_size: int = 512
    learning_rate: float = 1e-3
    lr_min: float = 1e-5
    scheduler: str = "cosine"
    log_every: int = 1000
    save_every: int = 50_000


@dataclass
class CheckpointConfig:
    checkpoint_dir: str = "./checkpoints"
    filename_override: Optional[str] = None


@dataclass
class OutputsConfig:
    output_dir: str = "./results"


@dataclass
class MetricsConfig:
    eps_hit: float = 1.0


@dataclass
class LoggingConfig:
    level: str = "INFO"
    use_wandb: bool = False
    wandb_project: str = "diffusion-modes"
    wandb_entity: Optional[str] = None


@dataclass
class ModeFinderConfig:
    timesteps: List[int] = field(
        default_factory=lambda: [800, 500, 300, 200, 150, 100, 80, 60, 40, 30, 20, 10, 5, 0]
    )
    step_size: float = 0.01
    split_eps: float = 2.0
    split_threshold: float = 0.3
    merge_radius: float = 0.25
    ascent_steps: int = 20
    refine_steps: int = 100
    refine_step_scale: float = 0.1
    n_starts: int = 3
    starts_min_sep: float = 0.5
    split_directions: str = "axes"


@dataclass
class ModeFinderV2Config:
    timesteps: List[int] = field(
        default_factory=lambda: [800, 500, 300, 200, 150, 100, 80, 60, 40, 30, 20, 10, 5, 0]
    )
    step_size: float = 0.01
    split_method: str = "hessian"
    split_eps: float = 2.0
    split_threshold: float = 0.3
    hessian_fd_eps: float = 1e-3
    hessian_split_eigenvalue_threshold: float = -1.0
    max_split_directions: int = 3
    merge_radius: float = 0.5
    adaptive_merge: bool = True
    ascent_steps: int = 20
    normalize_score: bool = True
    refine_steps: int = 100
    refine_step_scale: float = 0.1
    n_starts: int = 5
    start_method: str = "smart"
    ddim_steps: int = 50
    n_pilot_multiplier: int = 3
    pilot_cluster_radius: float = 2.0
    starts_min_sep: float = 0.5
    max_active_per_start: int = 30


@dataclass
class ModeFinderV3F1Config:
    timesteps: List[int] = field(
        default_factory=lambda: [800, 500, 300, 200, 150, 100, 80, 60, 40, 30, 20, 10, 5, 0]
    )
    step_size: float = 0.01
    split_eps: float = 2.0
    softness_threshold: float = 0.15
    amplification_threshold: float = 2.5
    tau_abs_min: float = 0.05
    max_split_directions: int = 2
    hessian_fd_eps: float = 1e-3
    merge_factor: float = 0.8
    merge_radius_min: float = 0.05
    ascent_steps: int = 20
    normalize_score: bool = True
    refine_steps: int = 100
    refine_step_scale: float = 0.1
    n_starts: int = 5
    start_method: str = "smart"
    ddim_steps: int = 50
    n_pilot_multiplier: int = 3
    pilot_cluster_radius: float = 2.0
    starts_min_sep: float = 0.5
    max_active_per_start: int = 30


@dataclass
class ModeFinderV3F2Config:
    timesteps: List[int] = field(
        default_factory=lambda: [800, 500, 300, 200, 150, 100, 80, 60, 40, 30, 20, 10, 5, 0]
    )
    step_size: float = 0.01
    n_particles: int = 50
    merge_factor: float = 0.8
    merge_radius_min: float = 0.05
    ascent_steps: int = 20
    normalize_score: bool = True
    refine_steps: int = 100
    refine_step_scale: float = 0.1
    ddim_steps: int = 50
    init_stop_t: int = 0


@dataclass
class ModeFinderV3F3Config:
    t_start: int = 800
    t_end: int = 0
    ode_steps_coarse: int = 15
    ode_steps_fine: int = 5
    use_adaptive_step: bool = True
    trace_stability_threshold: float = 0.3
    n_substeps_per_interval: int = 3
    n_particles: int = 50
    cluster_every: int = 0
    merge_factor: float = 0.8
    merge_radius_min: float = 0.05
    n_trace_probe: int = 8
    hessian_fd_eps: float = 1e-3
    refine_steps: int = 200
    refine_alpha: float = 0.001
    ddim_steps: int = 50


@dataclass
class BaselineConfig:
    n_samples: int = 1000
    refine_steps: int = 10
    refine_alpha: float = 0.01
    merge_radius: float = 0.25


@dataclass
class ExperimentConfig:
    name: str = "default"
    seed: int = 42
    device: str = "auto"
    dim: int = 1

    distribution: DistributionConfig = field(default_factory=DistributionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    mode_finder_v1: ModeFinderConfig = field(default_factory=ModeFinderConfig)
    mode_finder_v2: ModeFinderV2Config = field(default_factory=ModeFinderV2Config)
    mode_finder_v3_f1: ModeFinderV3F1Config = field(default_factory=ModeFinderV3F1Config)
    mode_finder_v3_f2: ModeFinderV3F2Config = field(default_factory=ModeFinderV3F2Config)
    mode_finder_v3_f3: ModeFinderV3F3Config = field(default_factory=ModeFinderV3F3Config)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _strict_dataclass(cls, data: dict, section: str):
    if data is None:
        return cls()
    known = {f.name for f in fields(cls)}
    unknown = set(data.keys()) - known
    if unknown:
        raise ValueError(
            f"Неизвестные параметры в секции [{section}]: {sorted(unknown)}.\n"
            f"Допустимые: {sorted(known)}.\n"
            f"Проверьте YAML и удалите лишние ключи."
        )
    return cls(**{k: v for k, v in data.items() if k in known})


def _load_raw_recursive(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    """Загрузить YAML с рекурсивным раскрытием цепочки _base."""
    path = path.resolve()

    if path in stack:
        chain = " -> ".join(str(p) for p in (*stack, path))
        raise ValueError(f"Циклическое наследование YAML: {chain}")

    if not path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(
            f"Корень YAML должен быть словарём, получено {type(raw).__name__}: {path}"
        )

    raw = dict(raw)
    base_rel = raw.pop("_base", None)
    if not base_rel:
        return raw

    base_path = (path.parent / base_rel).resolve()
    if not base_path.exists():
        raise FileNotFoundError(
            f"_base файл не найден: {base_path} (указан в {path})"
        )

    base_raw = _load_raw_recursive(base_path, (*stack, path))
    return _deep_merge(base_raw, raw)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    raw = _load_raw_recursive(path)

    _KNOWN_TOP = {
        "experiment",
        "distribution", "model", "training", "checkpoint", "outputs",
        "metrics", "logging",
        "mode_finder_v1", "mode_finder",
        "mode_finder_v2",
        "mode_finder_v3_f1", "mode_finder_v3_f2", "mode_finder_v3_f3",
        "baseline",
    }
    unknown_top = set(raw.keys()) - _KNOWN_TOP
    if unknown_top:
        raise ValueError(
            "Неизвестные секции в YAML: {}. Допустимые: {}.".format(
                sorted(unknown_top), sorted(_KNOWN_TOP))
        )

    _KNOWN_EXP = {"name", "seed", "device", "dim"}
    exp_raw = raw.get("experiment") or {}
    unknown_exp = set(exp_raw.keys()) - _KNOWN_EXP
    if unknown_exp:
        raise ValueError(
            "Неизвестные параметры в секции [experiment]: {}. Допустимые: {}.".format(
                sorted(unknown_exp), sorted(_KNOWN_EXP))
        )

    mf_v1_raw = raw.get("mode_finder_v1") or raw.get("mode_finder")

    cfg = ExperimentConfig(
        name=exp_raw.get("name", "default"),
        seed=exp_raw.get("seed", 42),
        device=exp_raw.get("device", "auto"),
        dim=exp_raw.get("dim", 1),

        distribution=_strict_dataclass(DistributionConfig,  raw.get("distribution"),        "distribution"),
        model=        _strict_dataclass(ModelConfig,         raw.get("model"),               "model"),
        training=     _strict_dataclass(TrainingConfig,      raw.get("training"),            "training"),
        checkpoint=   _strict_dataclass(CheckpointConfig,    raw.get("checkpoint"),          "checkpoint"),
        outputs=      _strict_dataclass(OutputsConfig,       raw.get("outputs"),             "outputs"),
        metrics=      _strict_dataclass(MetricsConfig,       raw.get("metrics"),             "metrics"),
        logging=      _strict_dataclass(LoggingConfig,       raw.get("logging"),             "logging"),

        mode_finder_v1=   _strict_dataclass(ModeFinderConfig,      mf_v1_raw,                     "mode_finder_v1"),
        mode_finder_v2=   _strict_dataclass(ModeFinderV2Config,    raw.get("mode_finder_v2"),     "mode_finder_v2"),
        mode_finder_v3_f1=_strict_dataclass(ModeFinderV3F1Config,  raw.get("mode_finder_v3_f1"), "mode_finder_v3_f1"),
        mode_finder_v3_f2=_strict_dataclass(ModeFinderV3F2Config,  raw.get("mode_finder_v3_f2"), "mode_finder_v3_f2"),
        mode_finder_v3_f3=_strict_dataclass(ModeFinderV3F3Config,  raw.get("mode_finder_v3_f3"), "mode_finder_v3_f3"),
        baseline=         _strict_dataclass(BaselineConfig,        raw.get("baseline"),           "baseline"),
    )

    logger.info("Конфигурация загружена: %s (dim=%d, seed=%d)", cfg.name, cfg.dim, cfg.seed)
    return cfg


def setup_logging(cfg: LoggingConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_checkpoint(
    checkpoint_dir: str | Path,
    dim: int,
    tag: str,
    filename_override: Optional[str] = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)

    if filename_override:
        p = Path(filename_override)
        if not p.is_absolute():
            p = checkpoint_dir / p
        return p

    new_path = checkpoint_dir / f"dim_{dim}" / f"model_{tag}.pth"
    if new_path.exists():
        logger.debug("Чекпоинт (новая структура): %s", new_path)
        return new_path

    legacy_path = checkpoint_dir / f"model_{tag}.pth"
    if legacy_path.exists():
        logger.info(
            "Чекпоинт (legacy): %s. Для новых моделей dim_%d/.",
            legacy_path, dim,
        )
        return legacy_path

    logger.info("Чекпоинт не найден (%s). Будет обучена новая модель.", new_path)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    return new_path


def get_results_dir(output_dir: str | Path, method: str, experiment: str) -> Path:
    d = Path(output_dir) / method / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d
