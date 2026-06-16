#!/usr/bin/env python3
"""Check the integrity of saved Levine 13D result files."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def check_csv(path: Path, errors: list[str], warnings: list[str]) -> None:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        warnings.append(f"CSV has no rows/columns: {path}")
        return
    except Exception as exc:
        errors.append(f"CSV unreadable: {path}: {exc}")
        return
    if df.empty:
        warnings.append(f"CSV empty: {path}")
    # Repeated run_name is expected in candidate-, population-, k-, radius-, and
    # flag-level tables. Only enforce uniqueness in explicit run summaries.
    if path.name in {"run_summary.csv", "training_summary.csv"} and "run_name" in df.columns:
        if df["run_name"].duplicated().any():
            errors.append(f"Duplicate run_name values in run summary: {path}")
    for col in df.columns:
        low = col.lower()
        if low.endswith("_pct") or "coverage" in low and "name" not in low:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) and ((vals < -1e-8) | (vals > 100 + 1e-8)).any():
                errors.append(f"Metric outside [0,100] in {path}:{col}")
    for ref_col in ["modes_path", "eval_path", "best_path", "checkpoint_path"]:
        if ref_col not in df.columns:
            continue
        for value in df[ref_col].dropna().astype(str).unique():
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if not candidate.exists():
                warnings.append(f"Referenced file missing ({ref_col}): {value} from {path}")


def check_npy(path: Path, errors: list[str], warnings: list[str]) -> None:
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as exc:
        errors.append(f"NPY unreadable: {path}: {exc}")
        return
    if path.name.startswith(("modes_", "raw_", "refined_", "endpoints_")):
        a = np.atleast_2d(arr)
        if a.shape[1] != 13:
            errors.append(f"Expected 13 columns: {path}, got {a.shape}")
        if not np.isfinite(a).all():
            errors.append(f"Non-finite values: {path}")
        if len(a) == 0:
            warnings.append(f"Empty candidate array: {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", default=["results_levine13"])
    p.add_argument("--report", default="results_levine13/additional/verification_report.txt")
    p.add_argument("--strict", action="store_true", help="Treat warnings as failure")
    args = p.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    csv_count = npy_count = 0
    for root_s in args.roots:
        root = Path(root_s)
        if not root.exists():
            errors.append(f"Root does not exist: {root}")
            continue
        for path in root.rglob("*.csv"):
            csv_count += 1
            check_csv(path, errors, warnings)
        for path in root.rglob("*.npy"):
            npy_count += 1
            check_npy(path, errors, warnings)

    lines = [
        "Levine 13D verification report",
        f"CSV checked: {csv_count}",
        f"NPY checked: {npy_count}",
        f"Errors: {len(errors)}",
        f"Warnings: {len(warnings)}",
        "",
        "ERRORS:",
        *(errors or ["none"]),
        "",
        "WARNINGS:",
        *(warnings or ["none"]),
    ]
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:6]))
    print(f"Report: {report}")
    if errors or (args.strict and warnings):
        sys.exit(1)


if __name__ == "__main__":
    main()
