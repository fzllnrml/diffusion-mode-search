from .metrics import compute_recall, compute_soft_error, compute_precision, evaluate_modes
from .distribution import GaussianMixture


def __getattr__(name):
    if name in ("setup_drive", "get_or_train_model"):
        from . import colab
        return getattr(colab, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "compute_recall", "compute_soft_error", "compute_precision", "evaluate_modes",
    "GaussianMixture",
    "setup_drive", "get_or_train_model",
]
