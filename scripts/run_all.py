#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Colab-friendly one-click workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", nargs="+", type=int, default=[5, 10, 14, 85])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run([sys.executable, "scripts/run_feature_pipeline.py"])
    cmd = [sys.executable, "scripts/run_model_integration.py", "--device", args.device, "--top-k", *[str(k) for k in args.top_k]]
    if args.skip_plots:
        cmd.append("--skip-plots")
    run(cmd)
    print("\nAll done. Main outputs:")
    print(PROJECT_ROOT / "outputs" / "processed")
    print(PROJECT_ROOT / "outputs" / "optuna")
    print(PROJECT_ROOT / "outputs" / "model_integration")


if __name__ == "__main__":
    main()
