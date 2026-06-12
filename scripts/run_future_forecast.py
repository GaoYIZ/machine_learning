#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create a 2020-01 scenario forecast for AQI trend reporting.

The original dataset ends at 2019-12-31 and does not provide true future
weather, emissions, or pollutant observations. This script therefore produces a
scenario forecast rather than a literal weather forecast. It uses the latest
model backtest trajectory as the winter reference pattern, then builds neutral,
better-air, and worse-air scenarios with uncertainty intervals estimated from
2019 test residuals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "future_forecast"


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


def load_reference_predictions() -> Tuple[pd.DataFrame, str, str]:
    """Prefer the balanced tail model, then fall back to peak or dimension outputs."""
    candidates = [
        (
            PROJECT_ROOT / "outputs" / "tail_optuna" / "tail_predictions_test.csv",
            "tail_optuna_pred_AQI",
            "TailOptuna 双尾均衡模型",
        ),
        (
            PROJECT_ROOT / "outputs" / "peak_optuna" / "best_peak_predictions_test.csv",
            "peak_optuna_pred_AQI",
            "PeakOptuna 峰值优化模型",
        ),
        (
            PROJECT_ROOT / "outputs" / "optuna" / "best_predictions_test.csv",
            "best_topk_pred",
            "Optuna 最佳维度随机森林模型",
        ),
    ]
    for path, pred_col, model_name in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if {"date", "actual_AQI", pred_col}.issubset(df.columns):
                return df, pred_col, model_name
    raise FileNotFoundError("No usable prediction file found. Run model/peak/tail stages first.")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return {
        "RMSE_AQI": float(np.sqrt(np.mean(residual**2))),
        "MAE_AQI": float(np.mean(np.abs(residual))),
        "Mean_Residual_AQI": float(np.mean(residual)),
        "Residual_Q10": float(np.quantile(residual, 0.10)),
        "Residual_Q90": float(np.quantile(residual, 0.90)),
        "Residual_Q025": float(np.quantile(residual, 0.025)),
        "Residual_Q975": float(np.quantile(residual, 0.975)),
    }


def build_forecast() -> Tuple[pd.DataFrame, Dict[str, object]]:
    ref, pred_col, model_name = load_reference_predictions()
    ref = ref.copy()
    ref["date"] = pd.to_datetime(ref["date"])

    # Use the first 30 available winter test predictions as a model-derived
    # reference trajectory. In the LSTM-aligned outputs this starts at 2019-01-14.
    base = ref.sort_values("date").head(30).reset_index(drop=True)
    if len(base) < 30:
        raise ValueError("Need at least 30 reference prediction rows for future scenario forecast.")

    y_true = base["actual_AQI"].to_numpy(dtype=float)
    y_pred = base[pred_col].to_numpy(dtype=float)
    residual_stats = metrics(y_true, y_pred)
    full_residual = ref["actual_AQI"].to_numpy(dtype=float) - ref[pred_col].to_numpy(dtype=float)
    q10, q90 = np.quantile(full_residual, [0.10, 0.90])
    q025, q975 = np.quantile(full_residual, [0.025, 0.975])

    future_dates = pd.date_range("2020-01-01", periods=30, freq="D")
    scenario_specs = [
        ("better_air", "改善情景：污染水平下降 10%", 0.90),
        ("neutral", "中性情景：延续相似冬季水平", 1.00),
        ("worse_air", "恶化情景：污染水平上升 10%", 1.10),
    ]
    rows = []
    for scenario, scenario_name, factor in scenario_specs:
        forecast = np.clip(y_pred * factor, 0, 300)
        for i, date in enumerate(future_dates):
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "scenario": scenario,
                    "scenario_name": scenario_name,
                    "forecast_AQI": float(forecast[i]),
                    "interval80_low": float(max(0, forecast[i] + q10)),
                    "interval80_high": float(min(300, forecast[i] + q90)),
                    "interval95_low": float(max(0, forecast[i] + q025)),
                    "interval95_high": float(min(300, forecast[i] + q975)),
                    "source_2019_date": base.loc[i, "date"].strftime("%Y-%m-%d"),
                    "source_2019_actual_AQI": float(base.loc[i, "actual_AQI"]),
                    "source_2019_model_pred_AQI": float(base.loc[i, pred_col]),
                }
            )

    forecast_df = pd.DataFrame(rows)
    summary = {
        "forecast_type": "scenario_forecast_not_weather_forecast",
        "forecast_period": "2020-01-01 to 2020-01-30",
        "reference_model": model_name,
        "reference_prediction_column": pred_col,
        "reference_period_used": f"{base['date'].min().date()} to {base['date'].max().date()}",
        "scenario_definition": {
            "neutral": "Use the model-derived winter reference trajectory.",
            "better_air": "Multiply neutral trajectory by 0.90.",
            "worse_air": "Multiply neutral trajectory by 1.10.",
        },
        "uncertainty_interval": "Residual quantiles from the 2019 test backtest.",
        "reference_backtest_metrics_first_30_days": residual_stats,
        "important_limitation": (
            "No true 2020 weather, emissions, or pollutant inputs are available. "
            "The result is for trend discussion and report completeness."
        ),
    }
    return forecast_df, summary


def plot_scenarios(forecast_df: pd.DataFrame, save_path: Path) -> None:
    configure_plot_style()
    df = forecast_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    colors = {"better_air": "#2f7d32", "neutral": "#1f77b4", "worse_air": "#b23a30"}
    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    for scenario, sub in df.groupby("scenario"):
        ax.plot(
            sub["date"],
            sub["forecast_AQI"],
            marker="o",
            ms=3.2,
            lw=1.5,
            color=colors.get(scenario, "#666666"),
            label=sub["scenario_name"].iloc[0],
        )
    ax.axhline(50, color="#2f7d32", ls=":", lw=1, label="AQI=50")
    ax.axhline(100, color="#d9a441", ls=":", lw=1, label="AQI=100")
    ax.axhline(150, color="#b23a30", ls=":", lw=1, label="AQI=150")
    ax.set_title("未来 30 天 AQI 情景预测（2020 年 1 月）")
    ax.set_xlabel("日期")
    ax.set_ylabel("AQI")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_interval(forecast_df: pd.DataFrame, save_path: Path) -> None:
    configure_plot_style()
    df = forecast_df[forecast_df["scenario"] == "neutral"].copy()
    df["date"] = pd.to_datetime(df["date"])
    x = df["date"].to_numpy()
    y = df["forecast_AQI"].to_numpy(dtype=float)
    low80 = df["interval80_low"].to_numpy(dtype=float)
    high80 = df["interval80_high"].to_numpy(dtype=float)
    low95 = df["interval95_low"].to_numpy(dtype=float)
    high95 = df["interval95_high"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    ax.fill_between(x, low95, high95, color="#9ecae1", alpha=0.28, label="95% 情景区间")
    ax.fill_between(x, low80, high80, color="#3182bd", alpha=0.28, label="80% 情景区间")
    ax.plot(x, y, color="#08519c", marker="o", ms=3.2, lw=1.6, label="中性情景预测")
    ax.axhline(100, color="#d9a441", ls=":", lw=1)
    ax.axhline(150, color="#b23a30", ls=":", lw=1)
    ax.set_title("未来预测区间与不确定性")
    ax.set_xlabel("日期")
    ax.set_ylabel("AQI")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_method_note(summary: Dict[str, object], path: Path) -> None:
    text = f"""# 未来趋势情景预测方法说明

## 预测口径

本项目原始数据截止到 2019-12-31，未提供 2020 年真实天气、排放或污染物观测。因此这里输出的是 **2020 年 1 月 AQI 情景预测**，不是严格意义上的天气预报。

## 使用的数据基础

- 参考模型：{summary['reference_model']}
- 参考测试窗口：{summary['reference_period_used']}
- 预测区间：{summary['forecast_period']}
- 不确定性区间：根据 2019 测试集残差分位数估计。

## 三种情景

- 改善情景：相对中性情景下降 10%。
- 中性情景：延续相似冬季模型预测走势。
- 恶化情景：相对中性情景上升 10%。

## 报告写法建议

建议写为：“基于 2019 年测试阶段的模型预测残差和冬季相似情景，对 2020 年 1 月 AQI 进行趋势推演。由于缺少未来天气和排放输入，该结果用于展示模型的未来情景分析能力，不等同于真实天气预报。”
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    forecast_df, summary = build_forecast()
    forecast_df.to_csv(OUTPUT_DIR / "future_forecast_scenarios.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / "future_forecast_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_method_note(summary, OUTPUT_DIR / "未来预测方法说明.md")
    plot_scenarios(forecast_df, OUTPUT_DIR / "未来30天AQI情景预测.png")
    plot_interval(forecast_df, OUTPUT_DIR / "未来预测区间与不确定性.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[FutureForecast] outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
