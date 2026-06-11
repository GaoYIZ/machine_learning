#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Colab-friendly one-click workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=None,
        help="Feature dimensions to benchmark. Defaults to 5, 8, 10, Optuna best top_k, 12, 14, 15, 20, and 85.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seq-len", type=int, default=14)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--run-peak-optuna", action="store_true", help="Run the optional peak-focused Optuna tuning stage.")
    parser.add_argument("--peak-trials", type=int, default=50)
    parser.add_argument("--run-tail-optuna", action="store_true", help="Run the optional double-tail Optuna tuning stage.")
    parser.add_argument("--tail-trials", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run([sys.executable, "scripts/run_feature_pipeline.py"])
    top_k = args.top_k
    if top_k is None:
        best_path = PROJECT_ROOT / "outputs" / "optuna" / "optuna_best_params.json"
        with open(best_path, "r", encoding="utf-8") as f:
            best_top_k = int(json.load(f)["best_top_k"])
        top_k = sorted({5, 8, 10, best_top_k, 12, 14, 15, 20, 85})
        print(f"\n[RunAll] default model dimensions: {top_k}")

    cmd = [
        sys.executable,
        "scripts/run_model_integration.py",
        "--device",
        args.device,
        "--seq-len",
        str(args.seq_len),
        "--top-k",
        *[str(k) for k in top_k],
    ]
    if args.skip_plots:
        cmd.append("--skip-plots")
    run(cmd)
    if args.run_peak_optuna:
        peak_cmd = [
            sys.executable,
            "scripts/run_peak_optuna.py",
            "--device",
            args.device,
            "--n-trials",
            str(args.peak_trials),
        ]
        if args.skip_plots:
            peak_cmd.append("--skip-plots")
        run(peak_cmd)
    if args.run_tail_optuna:
        tail_cmd = [
            sys.executable,
            "scripts/run_tail_optuna.py",
            "--device",
            args.device,
            "--n-trials",
            str(args.tail_trials),
        ]
        if args.skip_plots:
            tail_cmd.append("--skip-plots")
        run(tail_cmd)
    print("\nAll done. Main outputs:")
    print(PROJECT_ROOT / "outputs" / "processed")
    print(PROJECT_ROOT / "outputs" / "optuna")
    print(PROJECT_ROOT / "outputs" / "model_integration")
    if args.run_peak_optuna:
        print(PROJECT_ROOT / "outputs" / "peak_optuna")
    if args.run_tail_optuna:
        print(PROJECT_ROOT / "outputs" / "tail_optuna")


if __name__ == "__main__":
    main()
