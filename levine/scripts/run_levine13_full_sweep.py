#!/usr/bin/env python3
"""Run the Levine 13D mode-finder parameter sweep."""

from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

try:
    import torch
except Exception:  # noqa
    torch = None

from src.models.diffusion import DiffusionModel


DEFAULT_OUT_DIR = Path("results_levine13/full_sweep")
DEFAULT_DATA_PATH = Path("data/levine13/levine13_processed.npz")
DEFAULT_POP_PATH = Path("data/population_names_Levine_13dim.txt")


METHOD_IMPORTS = {
    "v2": ("src.algorithms.mode_finder_v2", "ModeFinderV2"),
    "v3f1": ("src.algorithms.mode_finder_v3_f1", "ModeFinderV3F1"),
    "v3f2": ("src.algorithms.mode_finder_v3_f2", "ModeFinderV3F2"),
    "v3f3": ("src.algorithms.mode_finder_v3_f3", "ModeFinderV3F3"),
}


@dataclass
class RunSpec:
    checkpoint_tag: str
    checkpoint_path: Path
    method: str
    r_value: float
    seed: int


def parse_csv_list(s: str, cast=str) -> List[Any]:
    if s is None or str(s).strip() == "":
        return []
    return [cast(x.strip()) for x in str(s).split(",") if x.strip() != ""]


def parse_checkpoints(s: str) -> Dict[str, Path]:
    """
    Format:
      all=checkpoints/levine13_all_100k.pt,labeled=checkpoints/levine13_labeled_100k.pt
    """
    result: Dict[str, Path] = {}
    for item in parse_csv_list(s, str):
        if "=" not in item:
            raise ValueError(f"Bad checkpoint spec '{item}'. Expected tag=path")
        tag, path = item.split("=", 1)
        result[tag.strip()] = Path(path.strip())
    return result


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("levine13_sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(out_dir / "logs" / "master.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_method_class(method: str):
    if method not in METHOD_IMPORTS:
        raise ValueError(f"Unknown method '{method}'. Available: {sorted(METHOD_IMPORTS)}")
    module_name, class_name = METHOD_IMPORTS[method]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def filter_kwargs_for_constructor(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return constructor arguments accepted by the method class."""
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    allowed = set(params.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in allowed}


def build_finder(
    method: str,
    model: Any,
    r_value: float,
    args: argparse.Namespace,
):
    cls = load_method_class(method)

    common_kwargs: Dict[str, Any] = {
        "model": model,

        # Population methods
        "n_particles": args.n_particles,

        # Start-based methods
        "n_starts": args.n_starts,
        "max_active_per_start": args.max_active_per_start,

        # Ascent / refinement
        "step_size": args.step_size,
        "ascent_steps": args.ascent_steps,
        "refine_steps": args.refine_steps,
        "refine_step_scale": args.refine_step_scale,

        # DDIM starts / sampling
        "ddim_steps": args.ddim_steps,

        # Method classes use either merge_radius_min or merge_radius.
        "merge_radius_min": r_value,
        "merge_radius": r_value,
        "merge_factor": args.merge_factor,

        # Domain clamp for standardized features
        "x_min": args.x_min,
        "x_max": args.x_max,
    }

    kwargs = filter_kwargs_for_constructor(cls, common_kwargs)
    return cls(**kwargs)


def run_finder(finder: Any, seed: int, verbose: bool = True):
    """
    Assumes methods have find_modes(seed=..., verbose=...).
    Falls back to find_modes(seed=...) if verbose is not accepted.
    """
    try:
        return finder.find_modes(seed=seed, verbose=verbose)
    except TypeError:
        return finder.find_modes(seed=seed)


def load_population_map(pop_path: Path) -> Dict[int, str]:
    if not pop_path.exists():
        return {}

    # Expected file columns: label population
    df = pd.read_csv(pop_path, sep=r"\s+", engine="python")
    cols = [c.lower() for c in df.columns]
    if "label" not in cols or "population" not in cols:
        return {}

    label_col = df.columns[cols.index("label")]
    pop_col = df.columns[cols.index("population")]

    return {int(row[label_col]): str(row[pop_col]) for _, row in df.iterrows()}


def raw_label_to_display_name(raw_label: Any, pop_map: Dict[int, str]) -> str:
    if raw_label is None or (isinstance(raw_label, float) and np.isnan(raw_label)):
        return "unknown"

    s = str(raw_label).strip()
    if s.lower() == "unassigned":
        return "unassigned"

    try:
        label_id = int(float(s))
        return pop_map.get(label_id, f"unknown_label_{label_id}")
    except Exception:
        return f"bad_label_{s}"


def load_levine_data(data_path: Path):
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find processed Levine file: {data_path}")

    d = np.load(data_path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    label_names = np.array([str(x) for x in d["label_names"]], dtype=object)
    return X, y, label_names


def evaluate_modes(
    modes: np.ndarray,
    data_path: Path,
    pop_path: Path,
    k: int,
    purity_threshold: float,
    max_unassigned_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    X, y, label_names = load_levine_data(data_path)
    pop_map = load_population_map(pop_path)

    label_names_lower = np.array([str(s).lower() for s in label_names])
    unassigned_ids = np.where(label_names_lower == "unassigned")[0]
    unassigned_id: Optional[int] = int(unassigned_ids[0]) if len(unassigned_ids) > 0 else None

    modes = np.atleast_2d(modes).astype(np.float32)

    if len(modes) == 0:
        eval_df = pd.DataFrame()
        best_df = pd.DataFrame()
        metrics = {
            "all_modes": 0,
            "confident_modes": 0,
            "covered_populations": 0,
            "covered_population_names": "",
            "mean_purity_labeled_conf": np.nan,
            "mean_unassigned_conf": np.nan,
            "mean_knn_dist_conf": np.nan,
        }
        return eval_df, best_df, metrics

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)

    dist, ind = nn.kneighbors(modes)

    rows: List[Dict[str, Any]] = []

    for j in range(len(modes)):
        neigh_labels = y[ind[j]]

        vals, counts = np.unique(neigh_labels, return_counts=True)
        order = np.argsort(counts)[::-1]
        majority = int(vals[order[0]])
        majority_count = int(counts[order[0]])

        purity_all = majority_count / k
        raw_majority = label_names[majority]

        if unassigned_id is not None:
            labeled_mask = neigh_labels != unassigned_id
            frac_unassigned = 1.0 - float(labeled_mask.mean())

            if labeled_mask.sum() > 0:
                vals_l, counts_l = np.unique(neigh_labels[labeled_mask], return_counts=True)
                order_l = np.argsort(counts_l)[::-1]
                majority_l = int(vals_l[order_l[0]])
                purity_labeled = float(counts_l[order_l[0]] / labeled_mask.sum())
                raw_majority_l = label_names[majority_l]
            else:
                raw_majority_l = "unassigned"
                purity_labeled = np.nan
        else:
            frac_unassigned = 0.0
            raw_majority_l = raw_majority
            purity_labeled = purity_all

        rows.append(
            {
                "mode_id": j,
                "nearest_label_all": raw_majority,
                "nearest_label_all_name": raw_label_to_display_name(raw_majority, pop_map),
                "purity_all": float(purity_all),
                "nearest_label_labeled_only": raw_majority_l,
                "nearest_label_labeled_only_name": raw_label_to_display_name(raw_majority_l, pop_map),
                "purity_labeled_only": purity_labeled,
                "frac_unassigned_neighbors": float(frac_unassigned),
                "mean_knn_dist": float(dist[j].mean()),
                "median_knn_dist": float(np.median(dist[j])),
            }
        )

    eval_df = pd.DataFrame(rows)

    conf = eval_df[
        (eval_df["purity_labeled_only"] >= purity_threshold)
        & (eval_df["frac_unassigned_neighbors"] <= max_unassigned_fraction)
        & (eval_df["nearest_label_labeled_only_name"] != "unassigned")
    ].copy()

    if len(conf) > 0:
        best_df = (
            conf.sort_values(
                [
                    "nearest_label_labeled_only_name",
                    "purity_labeled_only",
                    "frac_unassigned_neighbors",
                    "mean_knn_dist",
                ],
                ascending=[True, False, True, True],
            )
            .groupby("nearest_label_labeled_only_name")
            .head(1)
            .sort_values("nearest_label_labeled_only_name")
        )
    else:
        best_df = pd.DataFrame()

    covered_names = sorted(conf["nearest_label_labeled_only_name"].unique().tolist()) if len(conf) else []

    metrics = {
        "all_modes": int(len(eval_df)),
        "confident_modes": int(len(conf)),
        "covered_populations": int(len(covered_names)),
        "covered_population_names": ";".join(covered_names),
        "mean_purity_labeled_conf": float(conf["purity_labeled_only"].mean()) if len(conf) else np.nan,
        "mean_unassigned_conf": float(conf["frac_unassigned_neighbors"].mean()) if len(conf) else np.nan,
        "mean_knn_dist_conf": float(conf["mean_knn_dist"].mean()) if len(conf) else np.nan,
    }

    return eval_df, best_df, metrics


def cleanup_device_cache():
    gc.collect()
    if torch is not None:
        try:
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def save_summary(summary_rows: List[Dict[str, Any]], out_dir: Path, name: str = "summary_so_far.csv"):
    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / name, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoints",
        default="all=checkpoints/levine13_all_100k.pt,labeled=checkpoints/levine13_labeled_100k.pt",
        help="Comma-separated tag=path list.",
    )
    parser.add_argument(
        "--methods",
        default="v2,v3f1,v3f2,v3f3",
        help="Comma-separated methods. Available: v2,v3f1,v3f2,v3f3",
    )
    parser.add_argument(
        "--r-values",
        default="0.5,0.8,1.0,1.2,1.5,2.0",
        help="Comma-separated merge radius values.",
    )
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds.")

    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--pop-path", default=str(DEFAULT_POP_PATH))

    # Finder parameters
    parser.add_argument("--n-particles", type=int, default=300)
    parser.add_argument("--n-starts", type=int, default=30)
    parser.add_argument("--max-active-per-start", type=int, default=50)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--ascent-steps", type=int, default=20)
    parser.add_argument("--refine-steps", type=int, default=100)
    parser.add_argument("--refine-step-scale", type=float, default=0.1)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--merge-factor", type=float, default=0.8)
    parser.add_argument("--x-min", type=float, default=-6.0)
    parser.add_argument("--x-max", type=float, default=6.0)

    # Evaluation
    parser.add_argument("--k-neighbors", type=int, default=100)
    parser.add_argument("--purity-threshold", type=float, default=0.90)
    parser.add_argument("--max-unassigned-fraction", type=float, default=0.20)

    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--verbose-finder", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    logger = setup_logging(out_dir)

    checkpoints = parse_checkpoints(args.checkpoints)
    methods = parse_csv_list(args.methods, str)
    r_values = parse_csv_list(args.r_values, float)
    seeds = parse_csv_list(args.seeds, int)
    data_path = Path(args.data_path)
    pop_path = Path(args.pop_path)

    logger.info("=== Levine13 full sweep started ===")
    logger.info("checkpoints: %s", {k: str(v) for k, v in checkpoints.items()})
    logger.info("methods: %s", methods)
    logger.info("r_values: %s", r_values)
    logger.info("seeds: %s", seeds)
    logger.info("device: %s", args.device)
    logger.info("out_dir: %s", out_dir)
    logger.info("confident rule: purity_labeled_only >= %.3f and frac_unassigned <= %.3f",
                args.purity_threshold, args.max_unassigned_fraction)

    summary_rows: List[Dict[str, Any]] = []

    specs: List[RunSpec] = []
    for ckpt_tag, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            logger.warning("Checkpoint is missing; skipping tag=%s path=%s", ckpt_tag, ckpt_path)
            continue
        for method in methods:
            for r in r_values:
                for seed in seeds:
                    specs.append(RunSpec(ckpt_tag, ckpt_path, method, r, seed))

    logger.info("Total planned runs: %d", len(specs))

    for idx, spec in enumerate(specs, start=1):
        run_name = f"{spec.checkpoint_tag}_{spec.method}_R{spec.r_value:g}_seed{spec.seed}"
        logger.info("")
        logger.info("=== RUN %d/%d: %s ===", idx, len(specs), run_name)

        modes_path = out_dir / f"modes_{run_name}.npy"
        eval_path = out_dir / f"eval_{run_name}.csv"
        best_path = out_dir / f"best_per_population_{run_name}.csv"

        if args.skip_existing and eval_path.exists():
            logger.info("Skip existing run: %s", eval_path)
            try:
                df_prev = pd.read_csv(eval_path)
                row = {
                    "run_name": run_name,
                    "checkpoint": spec.checkpoint_tag,
                    "method": spec.method,
                    "r_value": spec.r_value,
                    "seed": spec.seed,
                    "status": "skipped_existing",
                    "all_modes": len(df_prev),
                }
                summary_rows.append(row)
                save_summary(summary_rows, out_dir)
            except Exception:
                pass
            continue

        start_time = time.time()
        row: Dict[str, Any] = {
            "run_name": run_name,
            "checkpoint": spec.checkpoint_tag,
            "checkpoint_path": str(spec.checkpoint_path),
            "method": spec.method,
            "r_value": spec.r_value,
            "seed": spec.seed,
            "status": "started",
        }

        try:
            logger.info("Loading model: %s", spec.checkpoint_path)
            model = DiffusionModel.from_checkpoint(str(spec.checkpoint_path), device=args.device)

            logger.info("Building finder: method=%s R=%.4g", spec.method, spec.r_value)
            finder = build_finder(spec.method, model, spec.r_value, args)

            logger.info("Finding modes...")
            res = run_finder(finder, seed=spec.seed, verbose=args.verbose_finder)

            modes = np.atleast_2d(np.asarray(res.modes, dtype=np.float32))
            nfe = int(getattr(res, "nfe", -1))

            np.save(modes_path, modes)
            logger.info("Found modes: %s", modes.shape)
            logger.info("NFE: %s", nfe)

            logger.info("Evaluating modes with k=%d nearest real cells...", args.k_neighbors)
            eval_df, best_df, metrics = evaluate_modes(
                modes=modes,
                data_path=data_path,
                pop_path=pop_path,
                k=args.k_neighbors,
                purity_threshold=args.purity_threshold,
                max_unassigned_fraction=args.max_unassigned_fraction,
            )

            eval_df["nfe"] = nfe
            eval_df["checkpoint"] = spec.checkpoint_tag
            eval_df["method"] = spec.method
            eval_df["r_value"] = spec.r_value
            eval_df["seed"] = spec.seed
            eval_df.to_csv(eval_path, index=False)

            if len(best_df) > 0:
                best_df["nfe"] = nfe
                best_df["checkpoint"] = spec.checkpoint_tag
                best_df["method"] = spec.method
                best_df["r_value"] = spec.r_value
                best_df["seed"] = spec.seed
            best_df.to_csv(best_path, index=False)

            elapsed = time.time() - start_time
            row.update(metrics)
            row.update(
                {
                    "nfe": nfe,
                    "elapsed_sec": elapsed,
                    "status": "ok",
                    "modes_path": str(modes_path),
                    "eval_path": str(eval_path),
                    "best_path": str(best_path),
                }
            )

            logger.info(
                "SUMMARY | modes=%s | confident=%s | covered=%s | "
                "purity=%.4f | unassigned=%.4f | knn_dist=%.4f | NFE=%s | time=%.1fs",
                row.get("all_modes"),
                row.get("confident_modes"),
                row.get("covered_populations"),
                row.get("mean_purity_labeled_conf", float("nan")),
                row.get("mean_unassigned_conf", float("nan")),
                row.get("mean_knn_dist_conf", float("nan")),
                nfe,
                elapsed,
            )
            logger.info("Covered populations: %s", row.get("covered_population_names", ""))

            # Print per-pop best modes to console in compact form
            if len(best_df) > 0:
                cols = [
                    "mode_id",
                    "nearest_label_labeled_only_name",
                    "purity_labeled_only",
                    "purity_all",
                    "frac_unassigned_neighbors",
                    "mean_knn_dist",
                ]
                logger.info("Best confident mode per population:\n%s", best_df[cols].to_string(index=False))

        except Exception as e:
            elapsed = time.time() - start_time
            row.update(
                {
                    "status": "error",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "elapsed_sec": elapsed,
                }
            )
            logger.error("Run failed: %s", run_name)
            logger.error("%s", traceback.format_exc())
            if not args.continue_on_error:
                summary_rows.append(row)
                save_summary(summary_rows, out_dir)
                raise

        finally:
            summary_rows.append(row)
            save_summary(summary_rows, out_dir)
            try:
                del model  # noqa
            except Exception:
                pass
            try:
                del finder  # noqa
            except Exception:
                pass
            cleanup_device_cache()

    summary = save_summary(summary_rows, out_dir, "summary.csv")
    logger.info("")
    logger.info("=== Levine13 full sweep finished ===")
    logger.info("Saved summary: %s", out_dir / "summary.csv")

    if len(summary) > 0:
        ok = summary[summary["status"] == "ok"].copy()
        if len(ok) > 0:
            show_cols = [
                "checkpoint",
                "method",
                "r_value",
                "seed",
                "all_modes",
                "confident_modes",
                "covered_populations",
                "mean_purity_labeled_conf",
                "mean_unassigned_conf",
                "mean_knn_dist_conf",
                "nfe",
                "elapsed_sec",
            ]
            logger.info("OK runs summary:\n%s", ok[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
