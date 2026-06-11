#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Peak-focused Optuna tuning on top of the leak-safe feature package."""

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
OPTUNA_DIR = PROJECT_ROOT / "outputs" / "optuna"
MODEL_DIR = PROJECT_ROOT / "outputs" / "model_integration"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "peak_optuna"
TARGET_OFFSET = 13
DEFAULT_TOP_K = [5, 10, 11, 14, 15, 20]

sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_pipeline import get_xy, load_feature_package  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--top-k", nargs="+", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--quick", action="store_true", help="Use a tiny search space for smoke tests.")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Noto Sans CJK JP",
                "Noto Sans CJK TC",
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )


def load_recommended_features(top_k: int) -> List[str]:
    path = OPTUNA_DIR / f"recommended_top{top_k}_features.csv"
    if path.exists():
        return pd.read_csv(path)["feature"].tolist()

    ranking_path = OPTUNA_DIR / "ranked_features_rf.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(
            f"Missing {path} and fallback ranking {ranking_path}. Run scripts/run_feature_pipeline.py first."
        )
    ranking = pd.read_csv(ranking_path)
    selected = ranking.head(top_k).copy()
    selected.to_csv(path, index=False, encoding="utf-8-sig")
    return selected["feature"].tolist()


def scale_target(y_train_raw: np.ndarray, y_val_raw: np.ndarray, y_test_raw: np.ndarray):
    mean = float(np.mean(y_train_raw))
    std = float(np.std(y_train_raw))
    if std == 0:
        raise ValueError("Training target standard deviation is zero.")
    return (
        ((y_train_raw - mean) / std).astype(np.float32),
        ((y_val_raw - mean) / std).astype(np.float32),
        ((y_test_raw - mean) / std).astype(np.float32),
        mean,
        std,
    )


def to_raw(y_std: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.asarray(y_std, dtype=np.float32).flatten() * std + mean


def persistence_prediction(prev_y_raw: np.ndarray, split_y_raw: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Yesterday's raw AQI predicts today's AQI, then converts to target z-score."""
    prev_y_raw = np.asarray(prev_y_raw, dtype=np.float32).flatten()
    split_y_raw = np.asarray(split_y_raw, dtype=np.float32).flatten()
    pred_raw = np.empty_like(split_y_raw, dtype=np.float32)
    pred_raw[0] = prev_y_raw[-1] if len(prev_y_raw) else split_y_raw[0]
    pred_raw[1:] = split_y_raw[:-1]
    return ((pred_raw - mean) / std).astype(np.float32)


def context_sequences(prev_X: np.ndarray, split_X: np.ndarray, seq_len: int, start_offset: int) -> np.ndarray:
    """Build split sequences with previous split rows as legal historical context."""
    prev_X = np.asarray(prev_X, dtype=np.float32)
    split_X = np.asarray(split_X, dtype=np.float32)
    history_len = seq_len - 1
    history = prev_X[-history_len:] if history_len else np.empty((0, split_X.shape[1]), dtype=np.float32)
    combined = np.vstack([history, split_X])
    seqs = []
    for target_idx in range(start_offset, len(split_X)):
        end_idx = history_len + target_idx
        start_idx = end_idx - seq_len + 1
        if start_idx < 0:
            raise ValueError(f"Not enough context for seq_len={seq_len}, target_idx={target_idx}")
        seqs.append(combined[start_idx : end_idx + 1])
    return np.asarray(seqs, dtype=np.float32)


def raw_metrics(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> Dict[str, float]:
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
    q75 = float(np.quantile(y_true_raw, 0.75))
    top25_mask = y_true_raw >= q75
    high_mask = y_true_raw >= 150
    return {
        "N_Test": int(len(y_true_raw)),
        "RMSE_AQI": float(np.sqrt(np.mean(residual**2))),
        "MAE_AQI": float(np.mean(abs_err)),
        "R2": float(1 - ss_res / (ss_tot + 1e-8)),
        "Top25_AQI_MAE": float(np.mean(abs_err[top25_mask])) if top25_mask.any() else np.nan,
        "AQI_ge_150_MAE": float(np.mean(abs_err[high_mask])) if high_mask.any() else np.nan,
        "AQI_ge_150_N": int(high_mask.sum()),
        "Mean_Residual_AQI": float(np.mean(residual)),
        "Max_Abs_Error_AQI": float(np.max(abs_err)),
    }


def peak_score(metrics: Dict[str, float], baseline_val_rmse: float) -> float:
    high = metrics["AQI_ge_150_MAE"]
    if np.isnan(high):
        high = metrics["Top25_AQI_MAE"]
    score = metrics["RMSE_AQI"] + 0.35 * high + 0.15 * metrics["Top25_AQI_MAE"]
    if metrics["RMSE_AQI"] > baseline_val_rmse * 1.05:
        score += (metrics["RMSE_AQI"] - baseline_val_rmse * 1.05) * 5.0
    return float(score)


def make_peak_weights(y_raw: np.ndarray, peak_weight: float) -> np.ndarray:
    q75 = float(np.quantile(y_raw, 0.75))
    return np.where(np.asarray(y_raw).flatten() >= q75, peak_weight, 1.0).astype(np.float32)


def apply_persistence_fusion(
    model_pred_std: np.ndarray,
    persistence_std: np.ndarray,
    persistence_raw: np.ndarray,
    alpha: float,
    threshold: float,
) -> np.ndarray:
    pred = np.asarray(model_pred_std, dtype=np.float32).copy()
    persistence_std = np.asarray(persistence_std, dtype=np.float32).flatten()[: len(pred)]
    persistence_raw = np.asarray(persistence_raw, dtype=np.float32).flatten()[: len(pred)]
    high_risk = persistence_raw >= threshold
    pred[high_risk] = alpha * pred[high_risk] + (1 - alpha) * persistence_std[high_risk]
    return pred


def load_baseline_val_rmse() -> float:
    params_path = OPTUNA_DIR / "optuna_best_params.json"
    if not params_path.exists():
        return 24.0
    with open(params_path, "r", encoding="utf-8") as f:
        return float(json.load(f).get("best_value_val_RMSE", 24.0))


def maybe_load_current_best() -> Dict[str, object] | None:
    path = MODEL_DIR / "results" / "all_model_comparison_my_features.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    row = df.sort_values("RMSE_AQI").iloc[0].to_dict()
    row["Comparison"] = "current_best_before_peak_optuna"
    return row


def train_predict_trial(
    trial: optuna.Trial,
    package: Dict[str, object],
    top_k_choices: List[int],
    device: str,
    quick: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    model_name = trial.suggest_categorical("model_name", ["bilstm", "lightgbm", "catboost"])
    top_k = trial.suggest_categorical("top_k", top_k_choices)
    peak_weight = trial.suggest_categorical("peak_weight", [1.5, 2.0, 3.0, 4.0, 5.0])
    alpha = trial.suggest_categorical("fusion_alpha", [0.3, 0.5, 0.7, 0.9])
    threshold = trial.suggest_categorical("fusion_threshold", [120, 130, 140, 150])

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
    weights = make_peak_weights(y_train_raw, peak_weight)
    params: Dict[str, object] = {
        "model_name": model_name,
        "top_k": top_k,
        "feature_count": len(features),
        "seq_len": seq_len,
        "peak_weight": peak_weight,
        "fusion_alpha": alpha,
        "fusion_threshold": threshold,
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
            model_label=f"PeakOptuna BiLSTM trial{trial.number}",
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
    else:
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

    pred_val = apply_persistence_fusion(pred_val, persistence_val_std, persistence_val_raw, alpha, threshold)
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
    alpha = float(best_params["fusion_alpha"])
    threshold = float(best_params["fusion_threshold"])
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
    pred_std = apply_persistence_fusion(pred_std, persistence_std, persistence_raw, alpha, threshold)
    y_eval_std = y_split[TARGET_OFFSET:]
    dates = best_params[f"{split}_dates"][TARGET_OFFSET:]
    return y_eval_std, pred_std, persistence_std, dates


def save_json(path: Path, payload: Dict[str, object]) -> None:
    def clean(value):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (pd.Index, pd.Series)):
            return [str(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            skip_keys = {
                "model_object",
                "X_splits",
                "y_splits",
                "y_raw_splits",
                "train_dates",
                "val_dates",
                "test_dates",
            }
            return {k: clean(v) for k, v in value.items() if k not in skip_keys}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean(payload), f, ensure_ascii=False, indent=2)


def trial_rows(study: optuna.Study) -> pd.DataFrame:
    rows = []
    for trial in study.trials:
        row = {
            "trial_number": trial.number,
            "state": str(trial.state),
            "score": trial.value,
            **trial.params,
            **trial.user_attrs,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_history(df: pd.DataFrame, save_path: Path) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(df["trial_number"], df["score"], marker="o", lw=1.2)
    best_so_far = df["score"].cummin()
    ax.plot(df["trial_number"], best_so_far, color="#b23a30", lw=2, label="当前最优")
    ax.set_title("峰值 Optuna 搜索历史")
    ax.set_xlabel("Trial 编号")
    ax.set_ylabel("平衡目标分数")
    ax.grid(alpha=0.3)
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_peak_bins(y_true_raw: np.ndarray, pred_raw: np.ndarray, persistence_raw: np.ndarray, save_path: Path) -> None:
    configure_plot_style()
    bins = [0, 50, 100, 150, 200, np.inf]
    labels = ["0-50", "50-100", "100-150", "150-200", "200+"]
    df = pd.DataFrame({"actual": y_true_raw, "optimized": pred_raw, "persistence": persistence_raw})
    df["AQI区间"] = pd.cut(df["actual"], bins=bins, labels=labels, right=False)
    rows = []
    for name in ["optimized", "persistence"]:
        tmp = df.copy()
        tmp["abs_error"] = np.abs(tmp["actual"] - tmp[name])
        summary = tmp.groupby("AQI区间", observed=False)["abs_error"].mean().reset_index()
        summary["模型"] = "峰值优化模型" if name == "optimized" else "持久性基线"
        rows.append(summary)
    plot_df = pd.concat(rows, ignore_index=True)
    pivot = plot_df.pivot(index="AQI区间", columns="模型", values="abs_error")
    ax = pivot.plot(kind="bar", figsize=(9, 5), color=["#5d8aa8", "#c08a5a"])
    ax.set_title("优化后模型与持久性基线的 AQI 分箱误差")
    ax.set_xlabel("真实 AQI 区间")
    ax.set_ylabel("MAE (AQI)")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_prediction_timeline(dates, y_true_raw, pred_raw, persistence_raw, save_path: Path) -> None:
    configure_plot_style()
    dates = pd.to_datetime(pd.Index(dates))
    fig, ax = plt.subplots(figsize=(15, 5.8))
    ax.plot(dates, y_true_raw, label="真实 AQI", color="#1f77b4", lw=1.3)
    ax.plot(dates, pred_raw, label="峰值优化模型", color="#d62728", lw=1.2, ls="--")
    ax.plot(dates, persistence_raw, label="持久性基线", color="#666666", lw=1.0, alpha=0.65)
    ax.axhline(150, color="#b23a30", lw=1, ls=":", label="AQI=150")
    ax.set_title("峰值优化后：测试集真实值与预测值对比")
    ax.set_xlabel("日期")
    ax.set_ylabel("AQI")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_high_aqi_scatter(y_true_raw, pred_raw, persistence_raw, save_path: Path) -> None:
    configure_plot_style()
    high = y_true_raw >= 150
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(y_true_raw[~high], pred_raw[~high], s=18, alpha=0.45, label="普通日", color="#5d8aa8")
    ax.scatter(y_true_raw[high], pred_raw[high], s=36, alpha=0.85, label="AQI>=150", color="#b23a30")
    ax.scatter(y_true_raw[high], persistence_raw[high], s=36, alpha=0.7, label="峰值日持久性预测", color="#c08a5a", marker="x")
    lo = float(min(y_true_raw.min(), pred_raw.min(), persistence_raw.min()))
    hi = float(max(y_true_raw.max(), pred_raw.max(), persistence_raw.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title("高 AQI 样本预测散点")
    ax.set_xlabel("真实 AQI")
    ax.set_ylabel("预测 AQI")
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
                print("[PeakOptuna] CUDA requested but unavailable; falling back to CPU.")
                args.device = "cpu"
        except Exception:
            print("[PeakOptuna] PyTorch CUDA check failed; falling back to CPU.")
            args.device = "cpu"

    top_k_choices = [k for k in args.top_k if k >= 1]
    if args.quick:
        top_k_choices = [k for k in top_k_choices if k in {5, 10, 11}][:3] or [5, 10]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    package = load_feature_package(PROCESSED_DIR)
    baseline_val_rmse = load_baseline_val_rmse()

    def objective(trial: optuna.Trial) -> float:
        y_val_eval, pred_val, y_val_raw_eval, params = train_predict_trial(
            trial, package, top_k_choices, args.device, args.quick
        )
        val_metrics = raw_metrics(y_val_raw_eval, to_raw(pred_val, params["target_mean"], params["target_std"]))
        score = peak_score(val_metrics, baseline_val_rmse)
        for key, value in {**params, **{f"val_{k}": v for k, v in val_metrics.items()}}.items():
            if key not in {"model_object", "features", "X_splits", "y_splits", "y_raw_splits"}:
                trial.set_user_attr(key, value)
        trial.set_user_attr("balanced_score", score)
        print(
            f"[PeakOptuna] trial={trial.number} model={params['model_name']} top{params['top_k']} "
            f"score={score:.4f} val_RMSE={val_metrics['RMSE_AQI']:.4f} "
            f"val_AQI>=150_MAE={val_metrics['AQI_ge_150_MAE']:.4f}"
        )
        return score

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)

    trials_df = trial_rows(study)
    trials_df.to_csv(OUTPUT_DIR / "peak_optuna_trials.csv", index=False, encoding="utf-8-sig")

    best_trial = study.best_trial
    _, _, _, best_params = train_predict_trial(best_trial, package, top_k_choices, args.device, args.quick)
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
    test_metrics = raw_metrics(y_test_raw, pred_test_raw)
    persistence_metrics = raw_metrics(y_test_raw, persistence_test_raw)

    best_payload = {
        "selection_rule": "Optuna minimizes validation balanced score; test split is used only once after selection.",
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
    save_json(OUTPUT_DIR / "best_peak_params.json", best_payload)

    metrics_payload = {
        "optimized_model": best_params["model_name"],
        "top_k": best_params["top_k"],
        "seq_len": best_params["seq_len"],
        "test_metrics": test_metrics,
        "persistence_same_window_metrics": persistence_metrics,
        "success_rule": "Success if AQI_ge_150_MAE improves while RMSE_AQI stays within 5% of 28.28.",
        "rmse_limit_for_success": 28.28 * 1.05,
    }
    save_json(OUTPUT_DIR / "best_peak_model_test_metrics.json", metrics_payload)

    current_best = maybe_load_current_best()
    comparison_rows = []
    if current_best:
        comparison_rows.append(current_best)
    persistence_row = {"Comparison": "persistence_same_window", "Model": "M1_Persistence", **persistence_metrics}
    optimized_row = {
        "Comparison": "peak_optuna_best",
        "Model": f"PeakOptuna_{best_params['model_name']}",
        "top_k": best_params["top_k"],
        "feature_count": best_params["feature_count"],
        **test_metrics,
    }
    comparison_rows.extend([persistence_row, optimized_row])
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUTPUT_DIR / "peak_model_comparison.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(OUTPUT_DIR / "peak_error_before_after.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Index(test_dates)).strftime("%Y-%m-%d"),
            "actual_AQI": y_test_raw,
            "peak_optuna_pred_AQI": pred_test_raw,
            "persistence_pred_AQI": persistence_test_raw,
            "abs_error_peak_optuna": np.abs(y_test_raw - pred_test_raw),
            "abs_error_persistence": np.abs(y_test_raw - persistence_test_raw),
        }
    ).to_csv(OUTPUT_DIR / "best_peak_predictions_test.csv", index=False, encoding="utf-8-sig")

    if not args.skip_plots:
        plot_history(trials_df, OUTPUT_DIR / "figures" / "peak_optuna_history.png")
        plot_peak_bins(y_test_raw, pred_test_raw, persistence_test_raw, OUTPUT_DIR / "figures" / "before_after_peak_error_bins.png")
        plot_prediction_timeline(
            test_dates,
            y_test_raw,
            pred_test_raw,
            persistence_test_raw,
            OUTPUT_DIR / "figures" / "before_after_prediction_vs_actual.png",
        )
        plot_high_aqi_scatter(
            y_test_raw,
            pred_test_raw,
            persistence_test_raw,
            OUTPUT_DIR / "figures" / "before_after_high_aqi_scatter.png",
        )

    print("\n[PeakOptuna] Best validation trial:")
    print(json.dumps({k: v for k, v in best_payload.items() if k != "best_features"}, ensure_ascii=False, indent=2, default=str))
    print("\n[PeakOptuna] Final test metrics:")
    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2, default=str))
    print(f"\n[PeakOptuna] outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
