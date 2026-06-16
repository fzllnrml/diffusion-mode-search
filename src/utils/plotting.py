from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple


COLORS = {
    "v1":    "#2196F3",   # синий
    "ours":  "#2196F3",   # синий (алиас v1 для обратной совместимости)
    "v2":    "#E91E63",   # розовый
    "ours_v2": "#E91E63", # алиас v2
    "v3f1":  "#9C27B0",   # фиолетовый
    "v3f2":  "#FF5722",   # глубокий оранжевый
    "v3f3":  "#009688",   # бирюзовый
    "b0":    "#FF9800",   # оранжевый
    "b10":   "#4CAF50",   # зелёный
}

LABELS = {
    "v1":      "Метод v1",
    "ours":    "Метод v1",
    "v2":      "Метод v2",
    "ours_v2": "Метод v2",
    "v3f1":    "Метод v3 (F1)",
    "v3f2":    "Метод v3 (F2)",
    "v3f3":    "Метод v3 (F3)",
    "b0":      "Baseline b0",
    "b10":     "Baseline b10",
}

MARKERS = {
    "v1":      "o",
    "ours":    "o",
    "v2":      "D",
    "ours_v2": "D",
    "v3f1":    "^",
    "v3f2":    "s",
    "v3f3":    "P",
    "b0":      "x",
    "b10":     "+",
}

METHOD_ORDER = ["v1", "ours", "v2", "ours_v2", "v3f1", "v3f2", "v3f3", "b0", "b10"]


def _sorted_methods(results: dict) -> List[str]:
    present = set(results.keys())
    return [m for m in METHOD_ORDER if m in present] + \
           [m for m in sorted(present) if m not in METHOD_ORDER]


def _color(method: str) -> str:
    return COLORS.get(method, "#607D8B")  # серый для неизвестных методов


def _label(method: str) -> str:
    return LABELS.get(method, method)


def _marker(method: str) -> str:
    return MARKERS.get(method, "o")


def _setup_style():
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
    })


def _add_band(ax, x, mean, std, color, label, marker):
    ax.plot(x, mean, color=color, label=label, marker=marker,
            markersize=6, linewidth=2, zorder=3)
    ax.fill_between(x, mean - std, mean + std,
                    color=color, alpha=0.15, zorder=2)


def plot_metric_vs_k(
    results: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    k_values: List[int],
    metric: str = "recall",
    title: str = "",
    save_path: Optional[str] = None,
    dim_label: str = "1D",
):
    _setup_style()
    fig, ax = plt.subplots()

    x = np.array(k_values)

    for method in _sorted_methods(results):
        if metric not in results[method]:
            continue
        mean, std = results[method][metric]
        _add_band(ax, x, mean, std, _color(method), _label(method), _marker(method))

    ax.set_xlabel("Число мод K")
    ylabel = "Recall" if metric == "recall" else "Soft Error"
    ax.set_ylabel(ylabel)
    ax.set_xticks(k_values)

    if metric == "recall":
        ax.set_ylim(-0.05, 1.15)

    ax.set_title(title or f"{ylabel} vs K ({dim_label})")
    ax.legend(loc="best")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  График сохранён: {save_path}")
    plt.show()
    plt.close(fig)


def plot_metric_vs_nfe(
    results: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    nfe_values: List[int],
    metric: str = "recall",
    title: str = "",
    save_path: Optional[str] = None,
    dim_label: str = "1D",
    log_scale: bool = True,
):
    _setup_style()
    fig, ax = plt.subplots()

    for method in _sorted_methods(results):
        entry = results[method]
        if metric not in entry:
            continue
        mean, std = entry[metric]
        nfe_arr = entry.get("nfe_values", np.array(nfe_values))
        _add_band(ax, nfe_arr, mean, std, _color(method), _label(method), _marker(method))

    ax.set_xlabel("NFE")
    ylabel = "Recall" if metric == "recall" else "Soft Error"
    ax.set_ylabel(ylabel)

    if log_scale:
        ax.set_xscale("log")

    if metric == "recall":
        ax.set_ylim(-0.05, 1.15)

    ax.set_title(title or f"{ylabel} vs NFE ({dim_label})")
    ax.legend(loc="best")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  График сохранён: {save_path}")
    plt.show()
    plt.close(fig)


def plot_starts_comparison(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    title: str = "",
    save_path: Optional[str] = None,
):
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods = _sorted_methods(results)
    x = np.arange(len(methods))
    width = 0.6

    for ax_idx, metric in enumerate(["recall", "soft_error"]):
        ax = axes[ax_idx]
        means = [results[m][metric][0] for m in methods if metric in results[m]]
        stds  = [results[m][metric][1] for m in methods if metric in results[m]]
        valid = [m for m in methods if metric in results[m]]
        colors = [_color(m) for m in valid]
        labels = [_label(m) for m in valid]

        ax.bar(
            np.arange(len(valid)), means, width,
            yerr=stds, capsize=5,
            color=colors, edgecolor="white", linewidth=1.5,
            error_kw={"linewidth": 1.5},
        )
        ax.set_xticks(np.arange(len(valid)))
        ax.set_xticklabels(labels, fontsize=10)
        ylabel = "Recall" if metric == "recall" else "Soft Error"
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

        if metric == "recall":
            ax.set_ylim(0, 1.15)

    fig.suptitle(title or "Сравнение при случайных стартовых точках", fontsize=14, y=1.02)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  График сохранён: {save_path}")
    plt.show()
    plt.close(fig)


def format_mean_std(mean: float, std: float, fmt: str = ".2f") -> str:
    return f"{mean:{fmt}} ± {std:{fmt}}"


def print_summary_table(
    results: Dict,
    k_values: List[int],
    metrics: List[str] = None,
):
    if metrics is None:
        metrics = ["recall", "soft_error", "nfe"]

    all_methods = _sorted_methods(results)

    header = f"{'K':>4}"
    for method in all_methods:
        for m in metrics:
            header += f" | {method}/{m:>10}"
    print(header)
    print("-" * len(header))

    for i, k in enumerate(k_values):
        row = f"{k:>4}"
        for method in all_methods:
            if method not in results:
                for m in metrics:
                    row += f" | {'N/A':>14}"
                continue
            for m in metrics:
                if m not in results[method]:
                    row += f" | {'N/A':>14}"
                    continue
                mean_arr, std_arr = results[method][m]
                if m == "nfe":
                    row += f" | {mean_arr[i]:>8.0f}±{std_arr[i]:>5.0f}"
                else:
                    row += f" | {format_mean_std(mean_arr[i], std_arr[i]):>14}"
        print(row)
