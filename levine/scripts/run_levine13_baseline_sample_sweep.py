#!/usr/bin/env python3
"""Run Levine 13D sampling baselines over sample size and merge radius."""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

try:
    import torch
except Exception:  # noqa
    torch = None

from src.models.diffusion import DiffusionModel
from src.algorithms.baseline import BaselineModeFinder
from src.algorithms.clustering import agglomerative_merge, merge_close


DEFAULT_OUT_DIR = Path("results_levine13/baseline_sample_sweep")
DEFAULT_DATA_PATH = Path("data/levine13/levine13_processed.npz")
DEFAULT_POP_PATH = Path("data/population_names_Levine_13dim.txt")


def parse_csv_list(s: str, cast=str) -> List[Any]:
    if s is None or str(s).strip() == "":
        return []
    return [cast(x.strip()) for x in str(s).split(",") if x.strip() != ""]


def parse_checkpoints(s: str) -> Dict[str, Path]:
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

    logger = logging.getLogger("levine13_baseline_sample_sweep")
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


def set_seed(seed: int):
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


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


def load_population_map(pop_path: Path) -> Dict[int, str]:
    if not pop_path.exists():
        return {}
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
            "confidently_annotated_modes": 0,
            "covered_populations": 0,
            "covered_population_names": "",
            "mean_purity_labeled_annotated": np.nan,
            "mean_unassigned_annotated": np.nan,
            "mean_knn_dist_annotated": np.nan,
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
                "nearest_label_all_name": raw_label_to_display_name(raw_majority, pop_path and pop_map),
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

    annotated = eval_df[
        (eval_df["purity_labeled_only"] >= purity_threshold)
        & (eval_df["frac_unassigned_neighbors"] <= max_unassigned_fraction)
        & (eval_df["nearest_label_labeled_only_name"] != "unassigned")
    ].copy()

    if len(annotated) > 0:
        best_df = (
            annotated.sort_values(
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

    covered_names = sorted(annotated["nearest_label_labeled_only_name"].unique().tolist()) if len(annotated) else []

    metrics = {
        "all_modes": int(len(eval_df)),
        "confidently_annotated_modes": int(len(annotated)),
        "covered_populations": int(len(covered_names)),
        "covered_population_names": ";".join(covered_names),
        "mean_purity_labeled_annotated": float(annotated["purity_labeled_only"].mean()) if len(annotated) else np.nan,
        "mean_unassigned_annotated": float(annotated["frac_unassigned_neighbors"].mean()) if len(annotated) else np.nan,
        "mean_knn_dist_annotated": float(annotated["mean_knn_dist"].mean()) if len(annotated) else np.nan,
    }

    return eval_df, best_df, metrics


def cluster_points(points: np.ndarray, r: float) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim == 1:
        return merge_close(points, r)
    return agglomerative_merge(points, r)


def save_summary(rows: List[Dict[str, Any]], out_dir: Path, name: str = "summary_so_far.csv"):
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / name, index=False)
    return df


def sample_and_refine(
    model: DiffusionModel,
    n_samples: int,
    refine_steps: int,
    refine_alpha: float,
    seed: int,
    logger: logging.Logger,
):
    set_seed(seed)

    finder = BaselineModeFinder(
        model=model,
        n_samples=n_samples,
        refine_steps=refine_steps,
        refine_alpha=refine_alpha,
        merge_radius=1.0,
    )

    model.enable_nfe_counting()
    model.reset_nfe()

    logger.info("b0: reverse sampling n_samples=%d", n_samples)
    t0 = time.time()
    raw = finder._sample_reverse(n_samples)
    raw = np.atleast_2d(np.asarray(raw, dtype=np.float32))
    sample_time = time.time() - t0

    nfe_b0_observed = int(getattr(model, "nfe", 0))
    nfe_b0_expected = int(n_samples * getattr(model, "T", 1000))
    nfe_b0 = max(nfe_b0_observed, nfe_b0_expected)

    logger.info(
        "b0 done: samples=%s NFE_observed=%d NFE_used=%d time=%.1fs",
        raw.shape, nfe_b0_observed, nfe_b0, sample_time
    )

    logger.info("b10: refining samples refine_steps=%d refine_alpha=%.4g", refine_steps, refine_alpha)
    t1 = time.time()
    refined = finder._refine_samples(raw)
    refined = np.atleast_2d(np.asarray(refined, dtype=np.float32))
    refine_time = time.time() - t1

    nfe_after_refine_observed = int(getattr(model, "nfe", 0))
    nfe_b10_expected = int(nfe_b0 + n_samples * refine_steps)
    nfe_b10 = max(nfe_after_refine_observed, nfe_b10_expected)

    logger.info(
        "b10 done: refined=%s NFE_observed=%d NFE_used=%d expected_extra=%d time=%.1fs",
        refined.shape,
        nfe_after_refine_observed,
        nfe_b10,
        n_samples * refine_steps,
        refine_time
    )

    changed = float(np.mean(np.linalg.norm(refined - raw, axis=1)))
    logger.info("Mean L2 shift from b0 samples to b10 refined samples: %.6f", changed)

    return {
        "raw": raw,
        "refined": refined,
        "nfe_b0": nfe_b0,
        "nfe_b10": nfe_b10,
        "nfe_b0_observed": nfe_b0_observed,
        "nfe_b10_observed": nfe_after_refine_observed,
        "nfe_b0_expected": nfe_b0_expected,
        "nfe_b10_expected": nfe_b10_expected,
        "sample_time_sec": sample_time,
        "refine_time_sec": refine_time,
        "mean_b10_shift": changed,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoints",
        default="all=checkpoints/levine13_all_100k.pt,labeled=checkpoints/levine13_labeled_100k.pt",
    )
    parser.add_argument("--n-samples-values", default="500,1000,2000,5000,10000")
    parser.add_argument("--r-values", default="0.5,0.8,1.0,1.2,1.5,2.0,2.5,3.0,4.0")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--refine-steps", type=int, default=10)
    parser.add_argument("--refine-alpha", type=float, default=0.01)

    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--pop-path", default=str(DEFAULT_POP_PATH))
    parser.add_argument("--k-neighbors", type=int, default=100)
    parser.add_argument("--purity-threshold", type=float, default=0.90)
    parser.add_argument("--max-unassigned-fraction", type=float, default=0.20)

    parser.add_argument("--skip-existing-samples", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    logger = setup_logging(out_dir)

    checkpoints = parse_checkpoints(args.checkpoints)
    n_samples_values = parse_csv_list(args.n_samples_values, int)
    r_values = parse_csv_list(args.r_values, float)
    seeds = parse_csv_list(args.seeds, int)
    data_path = Path(args.data_path)
    pop_path = Path(args.pop_path)

    logger.info("=== Levine13 fixed baseline sample-size sweep started ===")
    logger.info("checkpoints: %s", {k: str(v) for k, v in checkpoints.items()})
    logger.info("n_samples_values: %s", n_samples_values)
    logger.info("r_values: %s", r_values)
    logger.info("seeds: %s", seeds)
    logger.info("device: %s", args.device)
    logger.info("refine_steps=%d refine_alpha=%.4g", args.refine_steps, args.refine_alpha)
    logger.info(
        "annotation rule: purity_labeled_only >= %.3f and frac_unassigned <= %.3f",
        args.purity_threshold, args.max_unassigned_fraction
    )

    summary_rows: List[Dict[str, Any]] = []

    for ckpt_tag, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            logger.warning("Checkpoint missing, skipping: %s=%s", ckpt_tag, ckpt_path)
            continue

        for seed in seeds:
            for n_samples in n_samples_values:
                sample_tag = f"{ckpt_tag}_n{n_samples}_seed{seed}"
                raw_path = out_dir / f"raw_samples_{sample_tag}.npy"
                refined_path = out_dir / f"refined_samples_{sample_tag}.npy"
                meta_path = out_dir / f"sample_meta_{sample_tag}.csv"

                logger.info("")
                logger.info("=== SAMPLE PACK: checkpoint=%s n_samples=%d seed=%d ===", ckpt_tag, n_samples, seed)

                try:
                    if args.skip_existing_samples and raw_path.exists() and refined_path.exists() and meta_path.exists():
                        logger.info("Loading existing sample pack")
                        raw = np.load(raw_path)
                        refined = np.load(refined_path)
                        meta = pd.read_csv(meta_path).iloc[0].to_dict()
                        pack = {
                            "raw": raw,
                            "refined": refined,
                            "nfe_b0": int(meta["nfe_b0"]),
                            "nfe_b10": int(meta["nfe_b10"]),
                            "nfe_b0_observed": int(meta.get("nfe_b0_observed", meta["nfe_b0"])),
                            "nfe_b10_observed": int(meta.get("nfe_b10_observed", meta["nfe_b10"])),
                            "nfe_b0_expected": int(meta.get("nfe_b0_expected", meta["nfe_b0"])),
                            "nfe_b10_expected": int(meta.get("nfe_b10_expected", meta["nfe_b10"])),
                            "sample_time_sec": float(meta.get("sample_time_sec", np.nan)),
                            "refine_time_sec": float(meta.get("refine_time_sec", np.nan)),
                            "mean_b10_shift": float(meta.get("mean_b10_shift", np.nan)),
                        }
                    else:
                        logger.info("Loading model: %s", ckpt_path)
                        model = DiffusionModel.from_checkpoint(str(ckpt_path), device=args.device)
                        pack = sample_and_refine(
                            model=model,
                            n_samples=n_samples,
                            refine_steps=args.refine_steps,
                            refine_alpha=args.refine_alpha,
                            seed=seed,
                            logger=logger,
                        )
                        np.save(raw_path, pack["raw"])
                        np.save(refined_path, pack["refined"])
                        pd.DataFrame([{
                            "checkpoint": ckpt_tag,
                            "checkpoint_path": str(ckpt_path),
                            "n_samples": n_samples,
                            "seed": seed,
                            "nfe_b0": pack["nfe_b0"],
                            "nfe_b10": pack["nfe_b10"],
                            "nfe_b0_observed": pack["nfe_b0_observed"],
                            "nfe_b10_observed": pack["nfe_b10_observed"],
                            "nfe_b0_expected": pack["nfe_b0_expected"],
                            "nfe_b10_expected": pack["nfe_b10_expected"],
                            "sample_time_sec": pack["sample_time_sec"],
                            "refine_time_sec": pack["refine_time_sec"],
                            "mean_b10_shift": pack["mean_b10_shift"],
                        }]).to_csv(meta_path, index=False)
                        try:
                            del model
                        except Exception:
                            pass
                        cleanup_device_cache()

                except Exception as e:
                    logger.error("Sampling/refinement failed for %s n=%d seed=%d", ckpt_tag, n_samples, seed)
                    logger.error("%s", traceback.format_exc())
                    summary_rows.append({
                        "checkpoint": ckpt_tag,
                        "n_samples": n_samples,
                        "seed": seed,
                        "status": "sampling_error",
                        "error": repr(e),
                        "traceback": traceback.format_exc(),
                    })
                    save_summary(summary_rows, out_dir)
                    if not args.continue_on_error:
                        raise
                    continue

                methods = {
                    "b0": (pack["raw"], pack["nfe_b0"]),
                    "b10": (pack["refined"], pack["nfe_b10"]),
                }

                for method, (points, nfe) in methods.items():
                    for r in r_values:
                        run_name = f"{ckpt_tag}_{method}_n{n_samples}_R{r:g}_seed{seed}"
                        logger.info("")
                        logger.info("=== EVAL %s ===", run_name)

                        t0 = time.time()
                        row: Dict[str, Any] = {
                            "run_name": run_name,
                            "checkpoint": ckpt_tag,
                            "checkpoint_path": str(ckpt_path),
                            "method": method,
                            "r_value": r,
                            "n_samples": n_samples,
                            "seed": seed,
                            "nfe": nfe,
                            "nfe_b0": pack["nfe_b0"],
                            "nfe_b10": pack["nfe_b10"],
                            "mean_b10_shift": pack["mean_b10_shift"],
                            "status": "started",
                        }

                        try:
                            modes = cluster_points(points, r)
                            modes = np.atleast_2d(np.asarray(modes, dtype=np.float32))

                            modes_path = out_dir / f"modes_{run_name}.npy"
                            eval_path = out_dir / f"eval_{run_name}.csv"
                            best_path = out_dir / f"best_per_population_{run_name}.csv"

                            np.save(modes_path, modes)

                            eval_df, best_df, metrics = evaluate_modes(
                                modes=modes,
                                data_path=data_path,
                                pop_path=pop_path,
                                k=args.k_neighbors,
                                purity_threshold=args.purity_threshold,
                                max_unassigned_fraction=args.max_unassigned_fraction,
                            )

                            eval_df["checkpoint"] = ckpt_tag
                            eval_df["method"] = method
                            eval_df["r_value"] = r
                            eval_df["n_samples"] = n_samples
                            eval_df["seed"] = seed
                            eval_df["nfe"] = nfe
                            eval_df.to_csv(eval_path, index=False)

                            if len(best_df) > 0:
                                best_df["checkpoint"] = ckpt_tag
                                best_df["method"] = method
                                best_df["r_value"] = r
                                best_df["n_samples"] = n_samples
                                best_df["seed"] = seed
                                best_df["nfe"] = nfe
                            best_df.to_csv(best_path, index=False)

                            elapsed = time.time() - t0
                            row.update(metrics)
                            # backward-compatible aliases
                            row["confident_modes"] = row["confidently_annotated_modes"]
                            row["mean_purity_labeled_conf"] = row["mean_purity_labeled_annotated"]
                            row["mean_unassigned_conf"] = row["mean_unassigned_annotated"]
                            row["mean_knn_dist_conf"] = row["mean_knn_dist_annotated"]

                            row.update({
                                "all_modes": int(len(eval_df)),
                                "elapsed_sec": elapsed,
                                "status": "ok",
                                "modes_path": str(modes_path),
                                "eval_path": str(eval_path),
                                "best_path": str(best_path),
                            })

                            logger.info(
                                "SUMMARY | modes=%s | annotated=%s | covered=%s | "
                                "purity=%.4f | unassigned=%.4f | knn=%.4f | NFE=%s | time=%.1fs",
                                row.get("all_modes"),
                                row.get("confidently_annotated_modes"),
                                row.get("covered_populations"),
                                row.get("mean_purity_labeled_annotated", float("nan")),
                                row.get("mean_unassigned_annotated", float("nan")),
                                row.get("mean_knn_dist_annotated", float("nan")),
                                nfe,
                                elapsed,
                            )
                            logger.info("Covered populations: %s", row.get("covered_population_names", ""))

                        except Exception as e:
                            elapsed = time.time() - t0
                            row.update({
                                "status": "error",
                                "error": repr(e),
                                "traceback": traceback.format_exc(),
                                "elapsed_sec": elapsed,
                            })
                            logger.error("Evaluation failed: %s", run_name)
                            logger.error("%s", traceback.format_exc())
                            if not args.continue_on_error:
                                summary_rows.append(row)
                                save_summary(summary_rows, out_dir)
                                raise

                        finally:
                            summary_rows.append(row)
                            save_summary(summary_rows, out_dir)
                            cleanup_device_cache()

    summary = save_summary(summary_rows, out_dir, "summary.csv")
    logger.info("")
    logger.info("=== Levine13 fixed baseline sample-size sweep finished ===")
    logger.info("Saved summary: %s", out_dir / "summary.csv")

    ok = summary[summary["status"] == "ok"].copy() if len(summary) else pd.DataFrame()
    if len(ok) > 0:
        show_cols = [
            "checkpoint", "method", "r_value", "n_samples",
            "all_modes", "confidently_annotated_modes", "covered_populations",
            "mean_purity_labeled_annotated", "mean_unassigned_annotated",
            "mean_knn_dist_annotated", "nfe", "elapsed_sec"
        ]
        logger.info("OK runs summary:\n%s", ok[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
