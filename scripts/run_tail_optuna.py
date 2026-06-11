#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tail-balanced Optuna tuning for both high-AQI peaks and low-AQI valleys.

This stage keeps the existing leak-safe feature package fixed. It searches model
hyperparameters and a two-sided persistence fusion rule on the validation split,
then evaluates the selected configuration once on the 2019 test split.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tail_optuna"
TARGET_OFFSET = 13
DEFAULT_TOP_K = [5, 10, 11, 14, 15, 20]

sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_peak_optuna import (  # noqa: E402
    configure_plot_style,
    context_sequences,
    load_baseline_val_rmse,
    load_recommended_features,
    maybe_load_current_best,
    persistence_prediction,
    save_json,
    scale_target,
    to_raw,
    trial_rows,
)
from src.feature_pipeline import get_xy, load_feature_package  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--top-k", nargs="+", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--quick", action="store_true", help="Use a tiny search space for smoke tests.")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def available_model_choices() -> List[str]:
    """Use all planned models when installed; skip optional libraries locally."""
    choices: List[str] = []
    try:
        import modeling.models  # noqa: F401

        choices.append("bilstm")
    except Exception as exc:
        print(f"[TailOptuna] modeling.models could not be imported; skipping BiLSTM trials here: {exc}")
    try:
        import lightgbm  # noqa: F401

        choices.append("lightgbm")
    except Exception:
        print("[TailOptuna] lightgbm is not installed; skipping LightGBM trials in this environment.")
    try:
        import catboost  # noqa: F401

        choices.append("catboost")
    except Exception:
        print("[TailOptuna] catboost is not installed; skipping CatBoost trials in this environment.")
    if not choices:
        print("[TailOptuna] planned model libraries are unavailable; using ridge_smoke only for local smoke tests.")
        choices.append("ridge_smoke")
    return choices


def safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def tail_metrics(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> Dict[str, float]:
    """Evaluate overall error plus both tail-specific errors.

    residual = actual - prediction. Therefore positive residual on high-AQI days
    means under-prediction, and negative residual on low-AQI days means
    over-prediction.
    """
    y_true_raw = np.asarray(y_true_raw, dtype=np.float32).flatten()
    y_pred_raw = np.asarray(y_pred_raw, dtype=np.float32).flatten()
    n = min(len(y_true_raw), len(y_pred_raw))
    y_true_raw = y_true_raw[:n]
    y_pred_raw = y_pred_raw[:n]
    mask = ~(np.isnan(y_true_raw) | np.isnan(y_pred_raw))
    y_true_raw = y_true_raw[mask]
    y_pred_raw = y_pred_raw[mask]

    residual = y_true_raw - y_pred_raw
    abs_err = np.abs(residual)
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true_raw - np.mean(y_true_raw)) ** 2))

    q25 = float(np.quantile(y_true_raw, 0.25))
    q75 = float(np.quantile(y_true_raw, 0.75))
    low25_mask = y_true_raw <= q25
    top25_mask = y_true_raw >= q75
    low50_mask = y_true_raw <= 50
    high150_mask = y_true_raw >= 150

    high_under = np.maximum(residual[high150_mask], 0.0)
    if high_under.size == 0:
        high_under = np.maximum(residual[top25_mask], 0.0)
    low_over = np.maximum(-residual[low25_mask], 0.0)

    return {
        "N_Test": int(len(y_true_raw)),
        "RMSE_AQI": float(np.sqrt(np.mean(residual**2))),
        "MAE_AQI": float(np.mean(abs_err)),
        "R2": float(1 - ss_res / (ss_tot + 1e-8)),
        "Top25_AQI_MAE": safe_mean(abs_err[top25_mask]),
        "AQI_ge_150_MAE": safe_mean(abs_err[high150_mask]),
        "AQI_ge_150_N": int(high150_mask.sum()),
        "Low25_AQI_MAE": safe_mean(abs_err[low25_mask]),
        "Low25_N": int(low25_mask.sum()),
        "AQI_le_50_MAE": safe_mean(abs_err[low50_mask]),
        "AQI_le_50_N": int(low50_mask.sum()),
        "High150_UnderPred_Bias": safe_mean(high_under),
        "Low25_OverPred_Bias": safe_mean(low_over),
        "Mean_Residual_AQI": float(np.mean(residual)),
        "Max_Abs_Error_AQI": float(np.max(abs_err)),
    }


def tail_score(metrics: Dict[str, float], baseline_val_rmse: float) -> float:
    """Balanced validation objective for high peaks and low valleys."""
    high_mae = metrics["AQI_ge_150_MAE"]
    if np.isnan(high_mae):
        high_mae = metrics["Top25_AQI_MAE"]
    low50_mae = metrics["AQI_le_50_MAE"]
    if np.isnan(low50_mae):
        low50_mae = metrics["Low25_AQI_MAE"]

    score = (
        metrics["RMSE_AQI"]
        + 0.30 * high_mae
        + 0.20 * metrics["Low25_AQI_MAE"]
        + 0.15 * low50_mae
        + 0.10 * metrics["High150_UnderPred_Bias"]
        + 0.10 * metrics["Low25_OverPred_Bias"]
    )
    if metrics["RMSE_AQI"] > baseline_val_rmse * 1.05:
        score += (metrics["RMSE_AQI"] - baseline_val_rmse * 1.05) * 5.0
    return float(score)


def make_tail_weights(y_raw: np.ndarray, high_weight: float, low_weight: float) -> np.ndarray:
    """Upweight both extremes during model fitting."""
    y_raw = np.asarray(y_raw, dtype=np.float32).flatten()
    q25 = float(np.quantile(y_raw, 0.25))
    q75 = float(np.quantile(y_raw, 0.75))
    weights = np.ones_like(y_raw, dtype=np.float32)
    weights[y_raw >= q75] = high_weight
    weights[y_raw <= q25] = low_weight
    return weights


def apply_dual_persistence_fusion(
    model_pred_std: np.ndarray,
    persistence_std: np.ndarray,
    persistence_raw: np.ndarray,
    high_alpha: float,
    high_threshold: float,
    low_beta: float,
    low_threshold: float,
) -> np.ndarray:
    """Blend with persistence separately for high-risk and low-risk days."""
    pred = np.asarray(model_pred_std, dtype=np.float32).copy()
    persistence_std = np.asarray(persistence_std, dtype=np.float32).flatten()[: len(pred)]
    persistence_raw = np.asarray(persistence_raw, dtype=np.float32).flatten()[: len(pred)]

    low_risk = persistence_raw <= low_threshold
    high_risk = persistence_raw >= high_threshold
    pred[low_risk] = low_beta * pred[low_risk] + (1 - low_beta) * persistence_std[low_risk]
    pred[high_risk] = high_alpha * pred[high_risk] + (1 - high_alpha) * persistence_std[high_risk]
    return pred


def maybe_load_peak_best_with_low_metrics() -> Dict[str, object] | None:
    path = PROJECT_ROOT / "outputs" / "peak_optuna" / "best_peak_predictions_test.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"actual_AQI", "peak_optuna_pred_AQI"}
    if not required.issubset(df.columns):
        return None
    row = {
        "Comparison": "peak_optuna_best_existing",
        "Model": "PeakOptuna_existing",
        **tail_metrics(df["actual_AQI"].to_numpy(), df["peak_optuna_pred_AQI"].to_numpy()),
    }
    return row


def train_predict_trial(
    trial: optuna.Trial,
    package: Dict[str, object],
    top_k_choices: List[int],
    model_choices: List[str],
    device: str,
    quick: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    model_name = trial.suggest_categorical("model_name", model_choices)
    top_k = trial.suggest_categorical("top_k", top_k_choices)
    high_weight = trial.suggest_categorical("high_weight", [1.0, 1.5, 2.0] if quick else [1.0, 1.5, 2.0, 3.0])
    low_weight = trial.suggest_categorical("low_weight", [1.0, 1.5, 2.0] if quick else [1.0, 1.5, 2.0, 3.0])
    high_alpha = trial.suggest_categorical("high_alpha", [0.3, 0.5, 0.7, 0.9])
    low_beta = trial.suggest_categorical("low_beta", [0.3, 0.5, 0.7, 0.9])
    high_threshold = trial.suggest_categorical("high_threshold", [120, 130, 140, 150])
    low_threshold = trial.suggest_categorical("low_threshold", [50, 60, 70, 80])

    seq_choices = [7, 14] if quick else [7, 14, 21, 30]
    seq_len = trial.suggest_categorical("seq_len", seq_choices)
    features = load_recommended_features(top_k)

    X_train, y_train_raw = get_xy(package["train"], features)
    X_val, y_val_raw = get_xy(package["val"], features)
    X_test, y_test_raw = get_xy(package["test"], features)
    y_train, y_val, y_test, target_mean, target_std = scale_target(y_train_raw, y_val_raw, y_test_raw)

    val_offset = TARGET_OFFSET
    y_val_eval = y_val[val_offset:]
    y_val_eval_raw = y_val_raw[val_offset:]
    persistence_val_std = persistence_prediction(y_train_raw, y_val_raw, target_mean, target_std)[val_offset:]
    persistence_val_raw = to_raw(persistence_val_std, target_mean, target_std)
    weights = make_tail_weights(y_train_raw, high_weight=high_weight, low_weight=low_weight)

    params: Dict[str, object] = {
        "model_name": model_name,
        "top_k": top_k,
        "feature_count": len(features),
        "seq_len": seq_len,
        "high_weight": high_weight,
        "low_weight": low_weight,
        "high_alpha": high_alpha,
        "low_beta": low_beta,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "target_mean": target_mean,
        "target_std": target_std,
    }

    t0 = time.time()
    if model_name == "bilstm":
        from modeling.models import predict_bilstm, train_bilstm

        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64] if quick else [32, 64, 96, 128])
        dropout = trial.suggest_categorical("dropout", [0.1, 0.2] if quick else [0.1, 0.2, 0.3, 0.4])
        lr = trial.suggest_categorical("lr", [0.001] if quick else [0.0005, 0.001, 0.002])
        batch_size = trial.suggest_categorical("batch_size", [32] if quick else [16, 32, 64])
        epochs = 5 if quick else 80
        patience = 2 if quick else 10
        model, _ = train_bilstm(
            X_train,
            y_train,
            X_val,
            y_val,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            device=device,
            sample_weight_train=weights,
            seed=42,
            model_label=f"TailOptuna BiLSTM trial{trial.number}",
            dropout=dropout,
        )
        X_val_seq = context_sequences(X_train, X_val, seq_len=seq_len, start_offset=val_offset)
        pred_val = predict_bilstm(model, X_val_seq, seq_len=seq_len, device=device)
        params.update({"hidden_dim": hidden_dim, "dropout": dropout, "lr": lr, "batch_size": batch_size})
    elif model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 80 if quick else 120, 160 if quick else 600, step=80),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            }
        )
        model = LGBMRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            num_leaves=int(params["num_leaves"]),
            max_depth=int(params["max_depth"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train, sample_weight=weights, eval_set=[(X_val, y_val)])
        pred_val = model.predict(X_val)[val_offset:]
    elif model_name == "catboost":
        from catboost import CatBoostRegressor

        params.update(
            {
                "iterations": trial.suggest_int("iterations", 80 if quick else 120, 180 if quick else 700, step=80),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                "depth": trial.suggest_int("depth", 4, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            }
        )
        model = CatBoostRegressor(
            iterations=int(params["iterations"]),
            learning_rate=float(params["learning_rate"]),
            depth=int(params["depth"]),
            l2_leaf_reg=float(params["l2_leaf_reg"]),
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(X_train, y_train, sample_weight=weights, eval_set=(X_val, y_val), early_stopping_rounds=30)
        pred_val = model.predict(X_val)[val_offset:]
    elif model_name == "ridge_smoke":
        from sklearn.linear_model import Ridge

        alpha = trial.suggest_categorical("ridge_alpha", [0.1, 1.0, 10.0])
        params.update({"ridge_alpha": alpha})
        model = Ridge(alpha=float(alpha), random_state=42)
        model.fit(X_train, y_train, sample_weight=weights)
        pred_val = model.predict(X_val)[val_offset:]

    pred_val = apply_dual_persistence_fusion(
        pred_val,
        persistence_val_std,
        persistence_val_raw,
        high_alpha=high_alpha,
        high_threshold=high_threshold,
        low_beta=low_beta,
        low_threshold=low_threshold,
    )
    params["train_time_s"] = time.time() - t0
    params["model_object"] = model
    params["features"] = features
    params["X_splits"] = (X_train, X_val, X_test)
    params["y_splits"] = (y_train, y_val, y_test)
    params["y_raw_splits"] = (y_train_raw, y_val_raw, y_test_raw)
    return y_val_eval, pred_val, y_val_eval_raw, params


def predict_with_best(best_params: Dict[str, object], split: str = "test") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = best_params["model_object"]
    seq_len = int(best_params["seq_len"])
    high_alpha = float(best_params["high_alpha"])
    low_beta = float(best_params["low_beta"])
    high_threshold = float(best_params["high_threshold"])
    low_threshold = float(best_params["low_threshold"])
    target_mean = float(best_params["target_mean"])
    target_std = float(best_params["target_std"])
    X_train, X_val, X_test = best_params["X_splits"]
    y_train, y_val, y_test = best_params["y_splits"]
    y_train_raw, y_val_raw, y_test_raw = best_params["y_raw_splits"]

    if split == "val":
        X_prev, X_split = X_train, X_val
        y_prev_raw, y_split_raw, y_split = y_train_raw, y_val_raw, y_val
    else:
        X_prev, X_split = X_val, X_test
        y_prev_raw, y_split_raw, y_split = y_val_raw, y_test_raw, y_test

    if best_params["model_name"] == "bilstm":
        from modeling.models import predict_bilstm

        X_seq = context_sequences(X_prev, X_split, seq_len=seq_len, start_offset=TARGET_OFFSET)
        pred_std = predict_bilstm(model, X_seq, seq_len=seq_len, device=str(best_params.get("device", "cpu")))
    else:
        pred_std = model.predict(X_split)[TARGET_OFFSET:]

    persistence_std = persistence_prediction(y_prev_raw, y_split_raw, target_mean, target_std)[TARGET_OFFSET:]
    persistence_raw = to_raw(persistence_std, target_mean, target_std)
    pred_std = apply_dual_persistence_fusion(
        pred_std,
        persistence_std,
        persistence_raw,
        high_alpha=high_alpha,
        high_threshold=high_threshold,
        low_beta=low_beta,
        low_threshold=low_threshold,
    )
    y_eval_std = y_split[TARGET_OFFSET:]
    dates = best_params[f"{split}_dates"][TARGET_OFFSET:]
    return y_eval_std, pred_std, persistence_std, dates


def plot_history(df: pd.DataFrame, save_path: Path) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(df["trial_number"], df["score"], marker="o", lw=1.2, label="每次试验")
    ax.plot(df["trial_number"], df["score"].cummin(), color="#b23a30", lw=2, label="当前最优")
    ax.set_title("双尾优化搜索历史")
    ax.set_xlabel("Trial 编号")
    ax.set_ylabel("双尾平衡目标分数")
    ax.grid(alpha=0.3)
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bin_errors(
    y_true_raw: np.ndarray,
    pred_raw: np.ndarray,
    persistence_raw: np.ndarray,
    peak_raw: np.ndarray | None,
    save_path: Path,
) -> None:
    configure_plot_style()
    bins = [0, 50, 75, 100, 150, 200, np.inf]
    labels = ["0-50", "50-75", "75-100", "100-150", "150-200", "200+"]
    data = {
        "真实AQI": y_true_raw,
        "双尾优化模型": pred_raw,
        "持久性基线": persistence_raw,
    }
    if peak_raw is not None:
        data["峰值优化模型"] = peak_raw[: len(y_true_raw)]
    df = pd.DataFrame(data)
    df["AQI区间"] = pd.cut(df["真实AQI"], bins=bins, labels=labels, right=False)
    rows = []
    for model_col in [c for c in df.columns if c not in {"真实AQI", "AQI区间"}]:
        tmp = df[["真实AQI", "AQI区间", model_col]].copy()
        tmp["绝对误差"] = np.abs(tmp["真实AQI"] - tmp[model_col])
        summary = tmp.groupby("AQI区间", observed=False)["绝对误差"].mean().reset_index()
        summary["模型"] = model_col
        rows.append(summary)
    plot_df = pd.concat(rows, ignore_index=True)
    pivot = plot_df.pivot(index="AQI区间", columns="模型", values="绝对误差")
    ax = pivot.plot(kind="bar", figsize=(10, 5.2), color=["#4c78a8", "#c08a5a", "#72a56a"][: len(pivot.columns)])
    ax.set_title("高低 AQI 分箱误差对比")
    ax.set_xlabel("真实 AQI 区间")
    ax.set_ylabel("MAE (AQI)")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_low_scatter(y_true_raw, pred_raw, persistence_raw, save_path: Path) -> None:
    configure_plot_style()
    low = y_true_raw <= 50
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.scatter(y_true_raw[~low], pred_raw[~low], s=18, alpha=0.42, label="普通日", color="#5d8aa8")
    ax.scatter(y_true_raw[low], pred_raw[low], s=38, alpha=0.9, label="AQI<=50", color="#2f7d32")
    ax.scatter(y_true_raw[low], persistence_raw[low], s=38, alpha=0.75, label="低AQI日持久性预测", color="#c08a5a", marker="x")
    lo = float(min(y_true_raw.min(), pred_raw.min(), persistence_raw.min()))
    hi = float(max(y_true_raw.max(), pred_raw.max(), persistence_raw.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.axvline(50, color="#2f7d32", ls=":", lw=1)
    ax.set_title("低 AQI 样本预测散点")
    ax.set_xlabel("真实 AQI")
    ax.set_ylabel("预测 AQI")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_high_scatter(y_true_raw, pred_raw, persistence_raw, save_path: Path) -> None:
    configure_plot_style()
    high = y_true_raw >= 150
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.scatter(y_true_raw[~high], pred_raw[~high], s=18, alpha=0.42, label="普通日", color="#5d8aa8")
    ax.scatter(y_true_raw[high], pred_raw[high], s=38, alpha=0.9, label="AQI>=150", color="#b23a30")
    ax.scatter(y_true_raw[high], persistence_raw[high], s=38, alpha=0.75, label="高AQI日持久性预测", color="#c08a5a", marker="x")
    lo = float(min(y_true_raw.min(), pred_raw.min(), persistence_raw.min()))
    hi = float(max(y_true_raw.max(), pred_raw.max(), persistence_raw.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.axvline(150, color="#b23a30", ls=":", lw=1)
    ax.set_title("高 AQI 样本预测散点")
    ax.set_xlabel("真实 AQI")
    ax.set_ylabel("预测 AQI")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_timeline(dates, y_true_raw, pred_raw, persistence_raw, save_path: Path) -> None:
    configure_plot_style()
    dates = pd.to_datetime(pd.Index(dates))
    fig, ax = plt.subplots(figsize=(15, 5.8))
    ax.plot(dates, y_true_raw, label="真实 AQI", color="#1f77b4", lw=1.3)
    ax.plot(dates, pred_raw, label="双尾优化模型", color="#b23a30", lw=1.2, ls="--")
    ax.plot(dates, persistence_raw, label="持久性基线", color="#666666", lw=1.0, alpha=0.65)
    ax.axhline(50, color="#2f7d32", lw=1, ls=":", label="AQI=50")
    ax.axhline(150, color="#b23a30", lw=1, ls=":", label="AQI=150")
    ax.set_title("预测值与真实值：双尾优化")
    ax.set_xlabel("日期")
    ax.set_ylabel("AQI")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_low_residual_timeline(dates, y_true_raw, pred_raw, save_path: Path) -> None:
    configure_plot_style()
    dates = pd.to_datetime(pd.Index(dates))
    residual = y_true_raw - pred_raw
    low = y_true_raw <= 50
    fig, ax = plt.subplots(figsize=(15, 5.4))
    ax.plot(dates, residual, color="#4c78a8", lw=1.1, label="残差：真实值-预测值")
    ax.scatter(dates[low], residual[low], color="#2f7d32", s=30, label="AQI<=50 低谷日", zorder=3)
    ax.axhline(0, color="#333333", lw=1)
    ax.set_title("残差时间序列：低谷日标注")
    ax.set_xlabel("日期")
    ax.set_ylabel("残差 (AQI)")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                print("[TailOptuna] CUDA requested but unavailable; falling back to CPU.")
                args.device = "cpu"
        except Exception:
            print("[TailOptuna] PyTorch CUDA check failed; falling back to CPU.")
            args.device = "cpu"

    top_k_choices = [k for k in args.top_k if k >= 1]
    if args.quick:
        top_k_choices = [k for k in top_k_choices if k in {5, 10, 11}][:3] or [5, 10]
    model_choices = available_model_choices()
    print(f"[TailOptuna] model search space: {model_choices}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    package = load_feature_package(PROCESSED_DIR)
    baseline_val_rmse = load_baseline_val_rmse()

    def objective(trial: optuna.Trial) -> float:
        _, pred_val, y_val_raw_eval, params = train_predict_trial(
            trial, package, top_k_choices, model_choices, args.device, args.quick
        )
        val_metrics = tail_metrics(y_val_raw_eval, to_raw(pred_val, params["target_mean"], params["target_std"]))
        score = tail_score(val_metrics, baseline_val_rmse)
        for key, value in {**params, **{f"val_{k}": v for k, v in val_metrics.items()}}.items():
            if key not in {"model_object", "features", "X_splits", "y_splits", "y_raw_splits"}:
                trial.set_user_attr(key, value)
        trial.set_user_attr("balanced_score", score)
        print(
            f"[TailOptuna] trial={trial.number} model={params['model_name']} top{params['top_k']} "
            f"score={score:.4f} val_RMSE={val_metrics['RMSE_AQI']:.4f} "
            f"val_high150_MAE={val_metrics['AQI_ge_150_MAE']:.4f} "
            f"val_low50_MAE={val_metrics['AQI_le_50_MAE']:.4f}"
        )
        return score

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)

    trials_df = trial_rows(study)
    trials_df.to_csv(OUTPUT_DIR / "tail_optuna_trials.csv", index=False, encoding="utf-8-sig")

    best_trial = study.best_trial
    _, _, _, best_params = train_predict_trial(best_trial, package, top_k_choices, model_choices, args.device, args.quick)
    best_params["device"] = args.device
    best_params["train_dates"] = package["train"].index
    best_params["val_dates"] = package["val"].index
    best_params["test_dates"] = package["test"].index

    y_test_eval, pred_test, persistence_test, test_dates = predict_with_best(best_params, split="test")
    target_mean = float(best_params["target_mean"])
    target_std = float(best_params["target_std"])
    y_test_raw = to_raw(y_test_eval, target_mean, target_std)
    pred_test_raw = to_raw(pred_test, target_mean, target_std)
    persistence_test_raw = to_raw(persistence_test, target_mean, target_std)
    test_metrics = tail_metrics(y_test_raw, pred_test_raw)
    persistence_metrics = tail_metrics(y_test_raw, persistence_test_raw)

    best_payload = {
        "selection_rule": "Optuna minimizes validation double-tail score; test split is used only once after selection.",
        "score_formula": (
            "RMSE + 0.30*AQI_ge_150_MAE + 0.20*Low25_MAE + 0.15*AQI_le_50_MAE "
            "+ 0.10*High150_UnderPred_Bias + 0.10*Low25_OverPred_Bias"
        ),
        "baseline_val_RMSE": baseline_val_rmse,
        "best_trial_number": best_trial.number,
        "best_score": study.best_value,
        "best_params": {
            k: v
            for k, v in best_params.items()
            if k
            not in {
                "model_object",
                "features",
                "X_splits",
                "y_splits",
                "y_raw_splits",
                "train_dates",
                "val_dates",
                "test_dates",
            }
        },
        "best_features": best_params["features"],
    }
    save_json(OUTPUT_DIR / "best_tail_params.json", best_payload)

    metrics_payload = {
        "optimized_model": best_params["model_name"],
        "top_k": best_params["top_k"],
        "seq_len": best_params["seq_len"],
        "test_metrics": test_metrics,
        "persistence_same_window_metrics": persistence_metrics,
        "success_rule": (
            "Success if AQI<=50 and Low25 MAE improve while AQI>=150 MAE stays controlled "
            "and RMSE_AQI remains within 5% of 28.28."
        ),
        "rmse_limit_for_success": 28.28 * 1.05,
        "target_low50_mae_below": 24.50,
        "target_low25_mae_below": 23.75,
        "target_high150_mae_at_most": 45.0,
    }
    save_json(OUTPUT_DIR / "best_tail_model_test_metrics.json", metrics_payload)

    current_best = maybe_load_current_best()
    peak_best = maybe_load_peak_best_with_low_metrics()
    comparison_rows = []
    if current_best:
        comparison_rows.append(current_best)
    if peak_best:
        comparison_rows.append(peak_best)
    comparison_rows.append({"Comparison": "persistence_same_window", "Model": "M1_Persistence", **persistence_metrics})
    comparison_rows.append(
        {
            "Comparison": "tail_optuna_best",
            "Model": f"TailOptuna_{best_params['model_name']}",
            "top_k": best_params["top_k"],
            "feature_count": best_params["feature_count"],
            **test_metrics,
        }
    )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUTPUT_DIR / "tail_model_comparison.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(OUTPUT_DIR / "tail_error_before_after.csv", index=False, encoding="utf-8-sig")

    peak_pred_raw = None
    peak_path = PROJECT_ROOT / "outputs" / "peak_optuna" / "best_peak_predictions_test.csv"
    if peak_path.exists():
        peak_df = pd.read_csv(peak_path)
        if "peak_optuna_pred_AQI" in peak_df.columns:
            peak_pred_raw = peak_df["peak_optuna_pred_AQI"].to_numpy(dtype=np.float32)[: len(y_test_raw)]

    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Index(test_dates)).strftime("%Y-%m-%d"),
            "actual_AQI": y_test_raw,
            "tail_optuna_pred_AQI": pred_test_raw,
            "persistence_pred_AQI": persistence_test_raw,
            "abs_error_tail_optuna": np.abs(y_test_raw - pred_test_raw),
            "abs_error_persistence": np.abs(y_test_raw - persistence_test_raw),
            "residual_tail_optuna": y_test_raw - pred_test_raw,
            "is_low_AQI_le_50": y_test_raw <= 50,
            "is_high_AQI_ge_150": y_test_raw >= 150,
        }
    )
    if peak_pred_raw is not None:
        predictions["peak_optuna_pred_AQI"] = peak_pred_raw
        predictions["abs_error_peak_optuna"] = np.abs(y_test_raw - peak_pred_raw)
    predictions.to_csv(OUTPUT_DIR / "tail_predictions_test.csv", index=False, encoding="utf-8-sig")

    if not args.skip_plots:
        fig_dir = OUTPUT_DIR / "figures"
        plot_history(trials_df, fig_dir / "双尾优化搜索历史.png")
        plot_bin_errors(y_test_raw, pred_test_raw, persistence_test_raw, peak_pred_raw, fig_dir / "高低AQI分箱误差对比.png")
        plot_low_scatter(y_test_raw, pred_test_raw, persistence_test_raw, fig_dir / "低AQI散点图_优化前后.png")
        plot_high_scatter(y_test_raw, pred_test_raw, persistence_test_raw, fig_dir / "高AQI散点图_优化前后.png")
        plot_prediction_timeline(test_dates, y_test_raw, pred_test_raw, persistence_test_raw, fig_dir / "预测值与真实值_双尾优化.png")
        plot_low_residual_timeline(test_dates, y_test_raw, pred_test_raw, fig_dir / "残差时间序列_低谷标注.png")

    print("\n[TailOptuna] Best validation trial:")
    print(json.dumps({k: v for k, v in best_payload.items() if k != "best_features"}, ensure_ascii=False, indent=2, default=str))
    print("\n[TailOptuna] Final test metrics:")
    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2, default=str))
    print(f"\n[TailOptuna] outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
