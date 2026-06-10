"""
=============================================================================
评估模块 (Evaluation)
=============================================================================
包含: 回归指标计算、模型对比汇总、残差分析图生成
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict
import os


def inverse_transform_target(y: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    """将标准化后的目标变量还原到原始 AQI 单位。"""
    return np.asarray(y).flatten() * target_std + target_mean


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算全部回归评估指标"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # 移除 NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = y_true[mask]
    y_p = y_pred[mask]

    if len(y_t) == 0:
        return {}

    res = y_t - y_p
    n = len(y_t)

    mae = np.mean(np.abs(res))
    mse = np.mean(res ** 2)
    rmse = np.sqrt(mse)

    # MAPE (忽略零值)
    nz = y_t != 0
    mape = np.mean(np.abs(res[nz] / y_t[nz])) * 100 if nz.sum() > 0 else np.nan

    # R²
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # 峰值精度 (AQI > 75 百分位)
    thresh = np.percentile(y_t, 75)
    pk = y_t > thresh
    peak_mae = np.mean(np.abs(res[pk])) if pk.sum() > 0 else np.nan

    # 命中率 (相对误差 < 15%)
    hit = np.mean(np.abs(res) / (y_t + 1e-8) < 0.15) * 100

    # Durbin-Watson: 残差自相关检验
    dw_num = np.sum(np.diff(res, n=1) ** 2)
    dw_den = np.sum(res ** 2)
    dw = dw_num / (dw_den + 1e-8) if dw_den > 0 else 2.0

    return {
        'RMSE':     round(rmse, 4),
        'MAE':      round(mae, 4),
        'R2':       round(r2, 4),
        'MAPE(%)':  round(mape, 1),
        'Peak_MAE': round(peak_mae, 4),
        'Hit_Rate(%)': round(hit, 1),
        'Mean_Residual': round(np.mean(res), 4),
        'Std_Residual':  round(np.std(res), 4),
        'Durbin_Watson': round(dw, 4),
    }


def calc_metrics_original(y_true: np.ndarray, y_pred: np.ndarray,
                          target_mean: float, target_std: float) -> Dict[str, float]:
    """
    在原始 AQI 单位下计算指标。

    RMSE/MAE/MAPE/Hit Rate 这类指标需要在真实 AQI 标尺上解释,
    不能直接使用标准化后的 y 值计算百分比误差。
    """
    y_true_raw = inverse_transform_target(y_true, target_mean, target_std)
    y_pred_raw = inverse_transform_target(y_pred, target_mean, target_std)

    mask = ~(np.isnan(y_true_raw) | np.isnan(y_pred_raw))
    y_t = y_true_raw[mask]
    y_p = y_pred_raw[mask]
    if len(y_t) == 0:
        return {}

    res = y_t - y_p
    mae = np.mean(np.abs(res))
    rmse = np.sqrt(np.mean(res ** 2))
    nz = y_t != 0
    mape = np.mean(np.abs(res[nz] / y_t[nz])) * 100 if nz.sum() > 0 else np.nan

    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    thresh = np.percentile(y_t, 75)
    pk = y_t > thresh
    peak_mae = np.mean(np.abs(res[pk])) if pk.sum() > 0 else np.nan
    hit15 = np.mean(np.abs(res) / (y_t + 1e-8) < 0.15) * 100
    hit20 = np.mean(np.abs(res) / (y_t + 1e-8) < 0.20) * 100

    dw_num = np.sum(np.diff(res, n=1) ** 2)
    dw_den = np.sum(res ** 2)
    dw = dw_num / (dw_den + 1e-8) if dw_den > 0 else 2.0

    return {
        'RMSE_AQI': round(rmse, 4),
        'MAE_AQI': round(mae, 4),
        'MAPE(%)': round(mape, 2),
        'R2': round(r2, 4),
        'Peak_MAE_AQI': round(peak_mae, 4),
        'Hit_Rate_15%(%)': round(hit15, 1),
        'Hit_Rate_20%(%)': round(hit20, 1),
        'Mean_Residual_AQI': round(np.mean(res), 4),
        'Std_Residual_AQI': round(np.std(res), 4),
        'Durbin_Watson': round(dw, 4),
        'N_Test': int(len(y_t)),
    }


def build_comparison_table(all_results: Dict, y_test: np.ndarray,
                           aqi_mean: float, aqi_std: float) -> pd.DataFrame:
    """构建模型对比总表。百分比误差和残差均使用原始 AQI 单位。"""
    rows = []
    for name, info in all_results.items():
        pred = info.get('test_pred')
        if pred is None:
            continue
        # Align lengths
        yt = y_test
        if '_y_test' in info:
            yt = info['_y_test']
        min_len = min(len(yt), len(pred))
        m_std = calc_metrics(yt[:min_len], pred[:min_len])
        m_raw = calc_metrics_original(yt[:min_len], pred[:min_len], aqi_mean, aqi_std)
        row = {
            'Model': name,
            'Type': info.get('type', ''),
            'Train_Time(s)': round(info.get('train_time', 0), 1),
            **m_raw,
            'RMSE_std': m_std.get('RMSE', np.nan),
            'MAE_std': m_std.get('MAE', np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('RMSE_AQI')
    return df


def make_residual_plot(y_true, y_pred, model_name, save_path,
                       target_mean=None, target_std=None, dates=None):
    """生成完整的 6 子图残差诊断面板"""
    y_t = np.asarray(y_true).flatten()
    y_p = np.asarray(y_pred).flatten()
    if target_mean is not None and target_std is not None:
        y_t = inverse_transform_target(y_t, target_mean, target_std)
        y_p = inverse_transform_target(y_p, target_mean, target_std)
    mask = ~(np.isnan(y_t) | np.isnan(y_p))
    y_t, y_p = y_t[mask], y_p[mask]
    if dates is not None:
        dates = pd.to_datetime(pd.Index(dates))[:len(mask)][mask]
    res = y_t - y_p
    n = len(res)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1. 残差 vs 预测值 (异方差性检验)
    axes[0, 0].scatter(y_p, res, alpha=0.4, s=8, c='steelblue', edgecolors='none')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', lw=1.5)
    axes[0, 0].set_xlabel('Predicted AQI'); axes[0, 0].set_ylabel('Residual AQI')
    axes[0, 0].set_title('Residuals vs Predicted')

    # 2. 残差时间序列
    x_axis = dates if dates is not None else range(n)
    axes[0, 1].plot(x_axis, res, lw=0.8, color='steelblue')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', lw=1.5)
    axes[0, 1].set_xlabel('Date' if dates is not None else 'Time Index')
    axes[0, 1].set_ylabel('Residual AQI')
    axes[0, 1].set_title('Residual Time Series')

    # 3. 残差分布 + 正态拟合
    axes[0, 2].hist(res, bins=40, density=True, alpha=0.6,
                    color='steelblue', edgecolor='white')
    xr = np.linspace(res.min(), res.max(), 100)
    axes[0, 2].plot(xr, stats.norm.pdf(xr, np.mean(res), np.std(res)),
                    'r-', lw=2)
    axes[0, 2].set_title('Residual Distribution')

    # 4. Q-Q 图
    try:
        stats.probplot(res[:min(1000, n)], dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
    except:
        axes[1, 0].text(0.5, 0.5, 'N/A', ha='center', va='center')

    # 5. 残差自相关 (ACF) — 简化版
    try:
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(res[:min(500, n)], lags=min(30, n//4), ax=axes[1, 1])
        axes[1, 1].set_title('Residual ACF')
    except:
        axes[1, 1].text(0.5, 0.5, 'ACF N/A', ha='center', va='center')

    # 6. 预测 vs 实际 (最后 60 天)
    n_show = min(60, n)
    xs = dates[-n_show:] if dates is not None else range(n_show)
    axes[1, 2].plot(xs, y_t[-n_show:], 'b-', label='Actual', lw=1.5, alpha=0.8)
    axes[1, 2].plot(xs, y_p[-n_show:], 'r--', label='Predicted', lw=1.5, alpha=0.8)
    axes[1, 2].fill_between(xs, y_t[-n_show:], y_p[-n_show:], alpha=0.3, color='gray')
    axes[1, 2].legend(); axes[1, 2].set_title(f'Last {n_show} Days')

    plt.suptitle(f'{model_name} — Residual Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_comparison(df_comp: pd.DataFrame, save_path: str,
                          title: str = 'Model Performance Comparison on 2019 Test Set'):
    """绘制模型 RMSE、MAE、R2 对比图。"""
    df = df_comp.sort_values('RMSE_AQI', ascending=True).copy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].barh(df['Model'], df['RMSE_AQI'], color='#4c78a8')
    axes[0].invert_yaxis()
    axes[0].set_xlabel('RMSE (AQI)')
    axes[0].set_title('RMSE Comparison')

    axes[1].barh(df['Model'], df['MAE_AQI'], color='#f58518')
    axes[1].invert_yaxis()
    axes[1].set_xlabel('MAE (AQI)')
    axes[1].set_title('MAE Comparison')

    axes[2].barh(df['Model'], df['R2'], color='#54a24b')
    axes[2].invert_yaxis()
    axes[2].axvline(0, color='black', lw=0.8)
    axes[2].set_xlabel('R2')
    axes[2].set_title('R2 Comparison')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()


def plot_actual_vs_predicted(dates, y_true, y_pred, model_name, save_path):
    """绘制测试集真实值和预测值完整时间序列。"""
    dates = pd.to_datetime(pd.Index(dates))[:min(len(y_true), len(y_pred))]
    y_true = np.asarray(y_true).flatten()[:len(dates)]
    y_pred = np.asarray(y_pred).flatten()[:len(dates)]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dates, y_true, label='Actual AQI', color='#1f77b4', lw=1.4)
    ax.plot(dates, y_pred, label='Predicted AQI', color='#d62728', lw=1.2, ls='--')
    ax.fill_between(dates, y_true, y_pred, color='gray', alpha=0.18, label='Absolute Error Area')
    ax.set_title(f'{model_name}: Actual vs Predicted AQI on Test Set',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('AQI')
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame, save_path: str, top_n: int = 25):
    """绘制特征重要性 Top-N。"""
    df = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(df['Feature'], df['Importance'], color='#4c78a8')
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {min(top_n, len(importance_df))} Feature Importance',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()


def plot_future_forecast(forecast_df: pd.DataFrame, save_path: str):
    """绘制未来预测与 80%/95% 预测区间。"""
    dates = pd.to_datetime(forecast_df['date'])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, forecast_df['forecast_AQI'], marker='o', color='#d62728',
            label='Forecast AQI', lw=1.8)
    ax.fill_between(dates, forecast_df['lower_95'], forecast_df['upper_95'],
                    color='#ff9896', alpha=0.25, label='95% Prediction Interval')
    ax.fill_between(dates, forecast_df['lower_80'], forecast_df['upper_80'],
                    color='#d62728', alpha=0.22, label='80% Prediction Interval')
    ax.set_title('Future AQI Forecast with Prediction Intervals',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('AQI')
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()


def make_comparison_bar(results, metric, save_path):
    """生成模型对比水平柱状图"""
    names, vals = [], []
    for name, info in results.items():
        if 'test_pred' in info and info['test_pred'] is not None:
            pred = info['test_pred']
            if not np.all(np.isnan(pred)):
                names.append(name)
                m = calc_metrics(np.ones(len(pred)), pred)  # placeholder
                vals.append(info.get(f'_metrics_{metric}', 0))

    if not names:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(names)))
    ax.barh(names, vals, color=colors, edgecolor='white')
    ax.set_xlabel(metric); ax.invert_yaxis()
    ax.set_title(f'Model Comparison — {metric}')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
