#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the leakage-safe feature package and recommended dimensions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_pipeline import build_feature_package  # noqa: E402
from src.feature_dimension_search import run_optuna_search  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true", help="Skip EDA/feature plots.")
    parser.add_argument("--n-min", type=int, default=5, help="Minimum top_k for Optuna search.")
    parser.add_argument("--n-max", type=int, default=85, help="Maximum top_k for Optuna search.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = PROJECT_ROOT / "data" / "raw_air_quality.xlsx"
    print(f"[Feature] project root: {PROJECT_ROOT}")
    print(f"[Feature] raw data: {data_path}")
    build_feature_package(data_path=data_path, make_plots=not args.no_plots, save=True)
    result = run_optuna_search(n_min=args.n_min, n_max=args.n_max)
    print("[Feature] complete")
    print(f"[Feature] best_top_k={result['best_top_k']}")
    print(f"[Feature] best_validation_rmse={result['best_validation_rmse']:.4f}")


if __name__ == "__main__":
    main()
