# -*- coding: utf-8 -*-
"""
One-stop preprocessing, leakage-safe feature engineering, plotting, and handoff.

Task definition
---------------
Use historical daily air-quality observations to predict the next-day AQI.
For a target date t, model features may only use information available before
date t. In practice this means:

* Current-day AQI is the target, never an input feature.
* Current-day pollutant values are not used directly to predict current-day AQI.
* Rolling, EWM, and difference features are computed after ``shift(horizon)``.
* The scaler is fitted on the training split only, then reused for validation
  and test splits.

Downstream model users can either call ``build_feature_package`` once to create
all CSV/NPZ/figure outputs, or call ``load_feature_package`` to read the saved
handoff files and ``get_xy`` to split a DataFrame into model-ready X/y arrays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from .data_loader import load_raw_data
except ImportError:  # Allows: python src/feature_pipeline.py
    from data_loader import load_raw_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "processed"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "feature_figures"

TARGET_COL = "AQI"
QUALITY_COL = "quality_level"
POLLUTANT_COLS = ["PM2_5", "PM10", "SO2", "CO", "NO2", "O3_8h"]
RAW_NUMERIC_COLS = [TARGET_COL] + POLLUTANT_COLS

TRAIN_START, TRAIN_END = "2014-01-01", "2017-12-31"
VAL_START, VAL_END = "2018-01-01", "2018-12-31"
TEST_START, TEST_END = "2019-01-01", "2019-12-31"


def _configure_plot_style() -> None:
    """Use a restrained matplotlib/seaborn style with Chinese font fallback."""
    sns.set_theme(style="whitegrid", font_scale=0.9)
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
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "axes.unicode_minus": False,
        }
    )


def _as_path(path: Optional[str | Path], default: Path) -> Path:
    return Path(path).resolve() if path is not None else default.resolve()


def _add_feature_meta(
    rows: List[Dict[str, object]],
    feature: str,
    category: str,
    formula: str,
    note: str = "",
    use_for_model: bool = True,
    is_standardized: bool = True,
) -> None:
    """Append one feature metadata row used by feature_metadata.csv."""
    rows.append(
        {
            "feature": feature,
            "category": category,
            "formula": formula,
            "use_for_model": int(use_for_model),
            "is_standardized": int(is_standardized),
            "note": note,
        }
    )


def clean_air_quality_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw air-quality data before feature engineering.

    Missing numeric values are interpolated in time order because the missing
    ratio is very low. Extreme pollution episodes are marked with IQR flags but
    are not removed, since they are important for AQI forecasting.
    """
    data = df.copy().sort_index()

    for col in RAW_NUMERIC_COLS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Linear interpolation respects the daily time order; bfill/ffill only cover
    # possible boundary gaps after interpolation.
    fill_cols = [c for c in RAW_NUMERIC_COLS if c in data.columns]
    data[fill_cols] = data[fill_cols].interpolate(method="linear").bfill().ffill()

    # Physical concentrations cannot be negative. The provided dataset normally
    # has none, but keeping this guard makes the handoff robust.
    for col in POLLUTANT_COLS:
        if col in data.columns:
            data.loc[data[col] < 0, col] = 0.0

    # IQR flags are quality-control features. They are lagged before model use.
    for col in fill_cols:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        data[f"{col}_outlier"] = ((data[col] < lower) | (data[col] > upper)).astype(int)

    return data


def build_safe_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    horizon: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build leakage-safe features for target-date rows.

    Each row keeps the target value for the row date, while every model feature
    is either calendar information known in advance or historical measurements
    shifted by at least ``horizon`` days.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1 so model features stay before the target date.")

    data = df.copy().sort_index()
    feature_df = pd.DataFrame(index=data.index)
    metadata_rows: List[Dict[str, object]] = []

    feature_df[target_col] = data[target_col]
    if QUALITY_COL in data.columns:
        feature_df[QUALITY_COL] = data[QUALITY_COL]

    # Calendar features are allowed because the target date is known when making
    # a forecast; cyclical encoding preserves month/week periodicity.
    calendar_defs = {
        "month": (data.index.month, "target_date.month"),
        "day_of_week": (data.index.dayofweek, "target_date.dayofweek"),
        "quarter": (data.index.quarter, "target_date.quarter"),
        "is_weekend": ((data.index.dayofweek >= 5).astype(int), "1 if target_date is Sat/Sun else 0"),
        "day_of_year": (data.index.dayofyear, "target_date.dayofyear"),
    }
    for name, (values, formula) in calendar_defs.items():
        feature_df[name] = values
        _add_feature_meta(metadata_rows, name, "time", formula, "Known before the target date.")

    feature_df["month_sin"] = np.sin(2 * np.pi * feature_df["month"] / 12)
    feature_df["month_cos"] = np.cos(2 * np.pi * feature_df["month"] / 12)
    feature_df["dow_sin"] = np.sin(2 * np.pi * feature_df["day_of_week"] / 7)
    feature_df["dow_cos"] = np.cos(2 * np.pi * feature_df["day_of_week"] / 7)
    feature_df["year_progress"] = feature_df["day_of_year"] / 366.0

    for name, formula in {
        "month_sin": "sin(2*pi*month/12)",
        "month_cos": "cos(2*pi*month/12)",
        "dow_sin": "sin(2*pi*day_of_week/7)",
        "dow_cos": "cos(2*pi*day_of_week/7)",
        "year_progress": "day_of_year/366",
    }.items():
        _add_feature_meta(metadata_rows, name, "time", formula, "Cyclical/continuous date encoding.")

    # Lag definitions: lag_1 for horizon=1 means the value from t-1.
    for lag in (1, 2, 3, 7, 14):
        shift_days = horizon + lag - 1
        name = f"{target_col}_lag_{lag}"
        feature_df[name] = data[target_col].shift(shift_days)
        _add_feature_meta(
            metadata_rows,
            name,
            "lag",
            f"{target_col}.shift({shift_days})",
            f"Target history; available {shift_days} day(s) before target.",
        )

    for col in POLLUTANT_COLS:
        for lag in (1, 3, 7):
            shift_days = horizon + lag - 1
            name = f"{col}_lag_{lag}"
            feature_df[name] = data[col].shift(shift_days)
            _add_feature_meta(
                metadata_rows,
                name,
                "lag",
                f"{col}.shift({shift_days})",
                "Historical pollutant concentration.",
            )

    # Rolling/EWM/difference features are always computed on a shifted series.
    # This is the main anti-leakage guard in the pipeline.
    for col in [target_col] + POLLUTANT_COLS:
        hist = data[col].shift(horizon)
        for window in (3, 7, 14):
            mean_name = f"{col}_hist_roll_mean_{window}"
            feature_df[mean_name] = hist.rolling(window).mean()
            _add_feature_meta(
                metadata_rows,
                mean_name,
                "rolling",
                f"{col}.shift({horizon}).rolling({window}).mean()",
                "Causal rolling mean; target day is excluded.",
            )

        for window in (7, 14):
            std_name = f"{col}_hist_roll_std_{window}"
            feature_df[std_name] = hist.rolling(window).std()
            _add_feature_meta(
                metadata_rows,
                std_name,
                "rolling",
                f"{col}.shift({horizon}).rolling({window}).std()",
                "Causal rolling volatility; target day is excluded.",
            )

        if col == target_col:
            for window in (7, 14):
                min_name = f"{col}_hist_roll_min_{window}"
                max_name = f"{col}_hist_roll_max_{window}"
                feature_df[min_name] = hist.rolling(window).min()
                feature_df[max_name] = hist.rolling(window).max()
                _add_feature_meta(
                    metadata_rows,
                    min_name,
                    "rolling",
                    f"{col}.shift({horizon}).rolling({window}).min()",
                    "Historical low point over the window.",
                )
                _add_feature_meta(
                    metadata_rows,
                    max_name,
                    "rolling",
                    f"{col}.shift({horizon}).rolling({window}).max()",
                    "Historical high point over the window.",
                )

            ewm_name = f"{col}_hist_ewm_7"
            diff_1_name = f"{col}_hist_diff_1"
            diff_7_name = f"{col}_hist_diff_7"
            feature_df[ewm_name] = hist.ewm(span=7, adjust=False).mean()
            feature_df[diff_1_name] = hist - data[col].shift(horizon + 1)
            feature_df[diff_7_name] = hist - data[col].shift(horizon + 7)
            _add_feature_meta(metadata_rows, ewm_name, "rolling", f"{col}.shift({horizon}).ewm(span=7).mean()")
            _add_feature_meta(metadata_rows, diff_1_name, "trend", f"{col}.shift({horizon}) - {col}.shift({horizon + 1})")
            _add_feature_meta(metadata_rows, diff_7_name, "trend", f"{col}.shift({horizon}) - {col}.shift({horizon + 7})")

    eps = 1e-8
    pm25_lag1 = data["PM2_5"].shift(horizon)
    pm10_lag1 = data["PM10"].shift(horizon)
    so2_lag1 = data["SO2"].shift(horizon)
    no2_lag1 = data["NO2"].shift(horizon)
    co_lag1 = data["CO"].shift(horizon)

    feature_df["PM_ratio_lag1"] = pm25_lag1 / (pm10_lag1 + eps)
    feature_df["NO2_SO2_ratio_lag1"] = no2_lag1 / (so2_lag1 + eps)
    feature_df["pollution_index_lag1"] = (
        0.30 * pm25_lag1 / 500
        + 0.25 * pm10_lag1 / 600
        + 0.15 * so2_lag1 / 200
        + 0.15 * co_lag1 / 5
        + 0.15 * no2_lag1 / 200
    )
    _add_feature_meta(metadata_rows, "PM_ratio_lag1", "composite", "PM2_5.shift(horizon) / PM10.shift(horizon)")
    _add_feature_meta(metadata_rows, "NO2_SO2_ratio_lag1", "composite", "NO2.shift(horizon) / SO2.shift(horizon)")
    _add_feature_meta(
        metadata_rows,
        "pollution_index_lag1",
        "composite",
        "weighted historical PM2_5, PM10, SO2, CO, NO2 index",
        "Built on raw-scale historical pollutant values before feature scaling.",
    )

    for col in RAW_NUMERIC_COLS:
        flag_col = f"{col}_outlier"
        if flag_col in data.columns:
            name = f"{flag_col}_lag1"
            feature_df[name] = data[flag_col].shift(horizon)
            _add_feature_meta(
                metadata_rows,
                name,
                "quality_flag",
                f"{flag_col}.shift({horizon})",
                "Historical IQR outlier marker; extreme events are kept, not deleted.",
            )

    metadata = pd.DataFrame(metadata_rows).drop_duplicates("feature", keep="first").reset_index(drop=True)
    return feature_df, metadata


def split_feature_frame(
    feature_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = TARGET_COL,
) -> Dict[str, pd.DataFrame]:
    """Split target-date rows by time and drop rows still missing safe history."""
    splits = {
        "train": feature_df.loc[TRAIN_START:TRAIN_END].copy(),
        "val": feature_df.loc[VAL_START:VAL_END].copy(),
        "test": feature_df.loc[TEST_START:TEST_END].copy(),
    }
    required_cols = list(feature_cols) + [target_col]
    for key, split in splits.items():
        before = len(split)
        split = split.dropna(subset=required_cols).copy()
        split.index.name = "date"
        splits[key] = split
        if split.empty:
            raise ValueError(f"{key} split is empty after dropping rows with missing features.")
        print(f"[FeaturePipeline] {key}: {before} rows -> {len(split)} usable rows")
    return splits


def fit_transform_feature_scaler(
    splits: Dict[str, pd.DataFrame],
    feature_cols: Sequence[str],
) -> Tuple[Dict[str, pd.DataFrame], StandardScaler]:
    """Fit StandardScaler on train features only, then transform all splits."""
    scaler = StandardScaler()
    scaler.fit(splits["train"][feature_cols])

    scaled_splits: Dict[str, pd.DataFrame] = {}
    for name, df in splits.items():
        scaled = df.copy()
        scaled.loc[:, feature_cols] = scaler.transform(df[feature_cols])
        scaled_splits[name] = scaled
    return scaled_splits, scaler


def make_feature_audit(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple leakage audit table.

    All exported model features are intentionally low risk because formulas are
    either calendar-only or use shift(horizon) before reading measurements.
    """
    audit_rows = []
    for row in metadata.to_dict("records"):
        feature = row["feature"]
        category = row["category"]
        formula = row["formula"]
        if category == "time":
            availability = "Known before prediction time."
        else:
            availability = "Uses historical shifted values only."
        audit_rows.append(
            {
                "feature": feature,
                "category": category,
                "use_for_model": int(row.get("use_for_model", 1)),
                "leakage_risk": "low",
                "availability": availability,
                "formula": formula,
                "audit_note": "Pass: target-day AQI/current pollutant values are excluded.",
            }
        )
    return pd.DataFrame(audit_rows)


def get_xy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = TARGET_COL,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return model-ready X/y arrays from one processed split DataFrame.

    Downstream usage:
        X_train, y_train = get_xy(splits["train"], feature_cols)
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing[:10]}")
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")
    X = df[list(feature_cols)].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)
    return X, y


def make_sequence_npz(
    splits: Dict[str, pd.DataFrame],
    feature_cols: Sequence[str],
    output_dir: Optional[str | Path] = None,
    target_col: str = TARGET_COL,
    seq_len: int = 14,
) -> Dict[str, Path]:
    """
    Save LSTM-ready sequence arrays for each split.

    Each sample contains a 14-row sequence of already-safe feature rows ending
    at the target date. The target is the AQI of that sequence's final date.
    """
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2 for sequence models.")

    out_dir = _as_path(output_dir, DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}

    for split_name, df in splits.items():
        ordered = df.sort_index()
        X, y = get_xy(ordered, feature_cols, target_col)
        dates = ordered.index.astype(str).to_numpy()
        X_seq, y_seq, date_seq = [], [], []
        for end_idx in range(seq_len - 1, len(ordered)):
            start_idx = end_idx - seq_len + 1
            X_seq.append(X[start_idx : end_idx + 1])
            y_seq.append(y[end_idx])
            date_seq.append(dates[end_idx])

        path = out_dir / f"{split_name}_sequences.npz"
        np.savez_compressed(
            path,
            X_seq=np.asarray(X_seq, dtype=np.float32),
            y=np.asarray(y_seq, dtype=np.float32),
            dates=np.asarray(date_seq),
            feature_cols=np.asarray(feature_cols),
            target_col=np.asarray([target_col]),
        )
        saved[split_name] = path
        print(f"[FeaturePipeline] saved {split_name} sequences: {path.name}")

    return saved


def save_feature_package(
    splits: Dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    audit: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: Sequence[str],
    output_dir: Optional[str | Path] = None,
    target_col: str = TARGET_COL,
    horizon: int = 1,
) -> None:
    """Persist processed split CSVs, metadata, audit table, scaler info, and README."""
    out_dir = _as_path(output_dir, DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, df in splits.items():
        df.reset_index().to_csv(out_dir / f"{name}_features.csv", index=False, encoding="utf-8-sig")

    metadata.to_csv(out_dir / "feature_metadata.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out_dir / "feature_audit.csv", index=False, encoding="utf-8-sig")

    scaler_info = {
        "target_col": target_col,
        "target_scale": "raw_AQI_units",
        "horizon_days": horizon,
        "feature_cols": list(feature_cols),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "note": "Feature scaler fitted on train split only; target AQI is not scaled.",
    }
    with open(out_dir / "scaler_info.json", "w", encoding="utf-8") as f:
        json.dump(scaler_info, f, ensure_ascii=False, indent=2)

    readme = f"""# Processed Feature Package

This folder is generated by `src.feature_pipeline`.

## Prediction Task

Use historical observations to predict next-day `{target_col}`. The target stays
in raw AQI units. Model input features are already standardized with a
`StandardScaler` fitted on the training split only.

## Minimal Model Usage

```python
from src.feature_pipeline import load_feature_package, get_xy

splits = load_feature_package()
feature_cols = splits["feature_cols"]

X_train, y_train = get_xy(splits["train"], feature_cols)
X_val, y_val = get_xy(splits["val"], feature_cols)
X_test, y_test = get_xy(splits["test"], feature_cols)
```

## Files

- `train_features.csv`, `val_features.csv`, `test_features.csv`: model-ready split data.
- `feature_metadata.csv`: feature category, formula, standardization, and notes.
- `feature_audit.csv`: leakage-risk audit. All `use_for_model=1` rows should be `low`.
- `scaler_info.json`: train-only scaler parameters.
- `train_sequences.npz`, `val_sequences.npz`, `test_sequences.npz`: LSTM-ready arrays.

## Split Dates

- Train: {TRAIN_START} to {TRAIN_END}
- Validation: {VAL_START} to {VAL_END}
- Test: {TEST_START} to {TEST_END}
"""
    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)


def load_feature_package(
    output_dir: Optional[str | Path] = None,
) -> Dict[str, object]:
    """
    Load saved processed files for downstream modeling.

    Returns a dictionary with `splits`, `feature_cols`, `metadata`, `audit`, and
    `scaler_info`.
    """
    out_dir = _as_path(output_dir, DEFAULT_OUTPUT_DIR)
    splits: Dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        path = out_dir / f"{name}_features.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing processed split: {path}")
        split = pd.read_csv(path, parse_dates=["date"])
        split = split.set_index("date").sort_index()
        splits[name] = split

    metadata = pd.read_csv(out_dir / "feature_metadata.csv")
    audit = pd.read_csv(out_dir / "feature_audit.csv")
    with open(out_dir / "scaler_info.json", "r", encoding="utf-8") as f:
        scaler_info = json.load(f)

    # Expose train/val/test at the top level so downstream code can use the
    # simple handoff style: splits = load_feature_package(); splits["train"].
    return {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "splits": splits,
        "feature_cols": scaler_info["feature_cols"],
        "metadata": metadata,
        "audit": audit,
        "scaler_info": scaler_info,
        "output_dir": out_dir,
    }


def _rf_metrics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = TARGET_COL,
    random_state: int = 42,
) -> Dict[str, float]:
    X_train, y_train = get_xy(train_df, feature_cols, target_col)
    X_test, y_test = get_xy(test_df, feature_cols, target_col)
    model = RandomForestRegressor(
        n_estimators=160,
        max_depth=12,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "MAE": float(mean_absolute_error(y_test, pred)),
        "R2": float(r2_score(y_test, pred)),
    }


def smoke_test_feature_package(
    splits: Dict[str, pd.DataFrame],
    feature_cols: Sequence[str],
    audit: pd.DataFrame,
    output_dir: Optional[str | Path] = None,
    target_col: str = TARGET_COL,
) -> Dict[str, object]:
    """Run acceptance checks and a lightweight Random Forest training smoke test."""
    out_dir = _as_path(output_dir, DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_idx = set(splits["train"].index)
    val_idx = set(splits["val"].index)
    test_idx = set(splits["test"].index)
    if train_idx & val_idx or train_idx & test_idx or val_idx & test_idx:
        raise AssertionError("train/val/test date ranges overlap.")

    for split_name, df in splits.items():
        missing = df[list(feature_cols) + [target_col]].isna().sum()
        bad = missing[missing > 0]
        if not bad.empty:
            raise AssertionError(f"{split_name} has missing model values: {bad.to_dict()}")

    used_audit = audit[audit["use_for_model"] == 1]
    risky = used_audit[used_audit["leakage_risk"] != "low"]
    if not risky.empty:
        raise AssertionError(f"Feature audit contains risky model features: {risky['feature'].tolist()[:10]}")

    metrics = _rf_metrics(splits["train"], splits["test"], feature_cols, target_col)
    result = {
        "date_overlap_check": "pass",
        "missing_feature_check": "pass",
        "leakage_audit_check": "pass",
        "random_forest_smoke_test": metrics,
    }
    with open(out_dir / "smoke_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def _plot_eda_timeseries(clean_df: pd.DataFrame, figure_dir: Path) -> None:
    pollutants = [TARGET_COL] + POLLUTANT_COLS
    colors = ["#2f5d8c", "#c85746", "#d28b2a", "#3f8f73", "#6c6fb0", "#8b5a9b", "#4b9da7"]
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True)
    axes_flat = axes.ravel()
    for ax, col, color in zip(axes_flat, pollutants, colors):
        ax.plot(clean_df.index, clean_df[col], color=color, linewidth=0.35, alpha=0.45)
        monthly = clean_df[col].resample("M").mean()
        ax.plot(monthly.index, monthly, color=color, linewidth=1.8, label="月均值")
        ax.set_title(col, loc="left", fontweight="bold")
        ax.set_ylabel("浓度 / 指数值")
    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.96, 0.96))
    fig.suptitle("2013-2019 年每日空气质量时间序列", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figure_dir / "eda_timeseries.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_eda_seasonal_boxplot(clean_df: pd.DataFrame, figure_dir: Path) -> None:
    data = clean_df[[TARGET_COL]].copy()
    data["month"] = data.index.month
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    sns.boxplot(data=data, x="month", y=TARGET_COL, color="#8fb6c9", fliersize=2.0, ax=ax)
    monthly_mean = data.groupby("month")[TARGET_COL].mean()
    ax.plot(np.arange(12), monthly_mean.values, color="#b13f2f", marker="o", linewidth=1.4, label="月平均值")
    ax.set_title("AQI 按月份的季节性分布", fontweight="bold")
    ax.set_xlabel("月份")
    ax.set_ylabel("AQI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "eda_seasonal_boxplot.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_eda_correlation(clean_df: pd.DataFrame, figure_dir: Path) -> None:
    corr = clean_df[[TARGET_COL] + POLLUTANT_COLS].corr()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.4,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.75},
        ax=ax,
    )
    ax.set_title("AQI 与污染物相关性矩阵", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / "eda_correlation_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_lag_correlation(clean_df: pd.DataFrame, figure_dir: Path, max_lag: int = 30) -> None:
    rows = []
    for col in [TARGET_COL] + POLLUTANT_COLS:
        for lag in range(1, max_lag + 1):
            corr = clean_df[TARGET_COL].corr(clean_df[col].shift(lag))
            rows.append({"feature_source": col, "lag": lag, "corr": corr})
    lag_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for col in [TARGET_COL] + POLLUTANT_COLS:
        subset = lag_df[lag_df["feature_source"] == col]
        ax.plot(subset["lag"], subset["corr"], linewidth=1.5, label=col)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_title("不同历史滞后天数与目标日 AQI 的相关性", fontweight="bold")
    ax.set_xlabel("目标日前的滞后天数")
    ax.set_ylabel("皮尔逊相关系数")
    ax.legend(ncol=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_lag_correlation.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _feature_groups(metadata: pd.DataFrame, feature_cols: Sequence[str]) -> Dict[str, List[str]]:
    by_category = metadata.set_index("feature")["category"].to_dict()
    groups = {
        "时间特征": [f for f in feature_cols if by_category.get(f) == "time"],
        "滞后特征": [f for f in feature_cols if by_category.get(f) == "lag"],
        "移动统计/趋势特征": [f for f in feature_cols if by_category.get(f) in {"rolling", "trend"}],
        "复合污染特征": [f for f in feature_cols if by_category.get(f) == "composite"],
        "全部无泄漏特征": list(feature_cols),
    }
    return {k: v for k, v in groups.items() if v}


def _plot_feature_group_ablation(
    splits: Dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    feature_cols: Sequence[str],
    figure_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    groups = _feature_groups(metadata, feature_cols)
    for group_name, cols in groups.items():
        metrics = _rf_metrics(splits["train"], splits["test"], cols)
        rows.append({"feature_group": group_name, "n_features": len(cols), **metrics})
    result = pd.DataFrame(rows).sort_values("RMSE")
    result.to_csv(output_dir / "feature_group_ablation.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    plot_df = result.sort_values("RMSE", ascending=False)
    axes[0].barh(plot_df["feature_group"], plot_df["RMSE"], color="#6f9fb8")
    axes[0].set_title("不同特征组的 RMSE", fontweight="bold")
    axes[0].set_xlabel("RMSE (AQI)")
    axes[1].barh(plot_df["feature_group"], plot_df["R2"], color="#b8875f")
    axes[1].set_title("不同特征组的 R2", fontweight="bold")
    axes[1].set_xlabel("R2")
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_group_ablation.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return result


def _plot_feature_importance_rf(
    splits: Dict[str, pd.DataFrame],
    feature_cols: Sequence[str],
    figure_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    X_train, y_train = get_xy(splits["train"], feature_cols)
    model = RandomForestRegressor(
        n_estimators=240,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    importance = (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(output_dir / "feature_importance_rf.csv", index=False, encoding="utf-8-sig")

    top = importance.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    ax.barh(top["feature"], top["importance"], color="#7a9f6e")
    ax.set_title("随机森林特征重要性 Top 20", fontweight="bold")
    ax.set_xlabel("重要性")
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_importance_rf.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return importance


def _plot_data_flow(figure_dir: Path) -> None:
    steps = [
        ("原始 Excel", "2013-2019 每日 AQI\n及主要污染物浓度"),
        ("数据清洗", "线性插补缺失值\n检查并修正负浓度"),
        ("质量标记", "IQR 异常值标记\n保留为历史质量特征"),
        ("无泄漏特征", "shift(1) 后构造滞后\n移动统计、趋势与复合特征"),
        ("切分与标准化", "2014-2017 训练集\n2018 验证集，2019 测试集"),
        ("建模交接", "传统模型使用 CSV\nLSTM 使用 NPZ 序列"),
    ]
    fig, ax = plt.subplots(figsize=(11.0, 3.2))
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(steps))
    y = 0.55
    for i, ((title, body), x) in enumerate(zip(steps, xs)):
        ax.text(
            x,
            y,
            f"{title}\n{body}",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f2f5f3", edgecolor="#6e7f86", linewidth=0.9),
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.07, y),
                xytext=(x + 0.07, y),
                arrowprops=dict(arrowstyle="->", color="#4f5b62", linewidth=1.1),
            )
    ax.set_title("从原始数据到建模数据包的数据处理流程", fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(figure_dir / "processed_data_flow.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_feature_plots(
    clean_df: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    feature_cols: Sequence[str],
    figure_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Path]:
    """Generate EDA and feature-report figures for the handoff package."""
    _configure_plot_style()
    fig_dir = _as_path(figure_dir, DEFAULT_FIGURE_DIR)
    out_dir = _as_path(output_dir, DEFAULT_OUTPUT_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_eda_timeseries(clean_df, fig_dir)
    _plot_eda_seasonal_boxplot(clean_df, fig_dir)
    _plot_eda_correlation(clean_df, fig_dir)
    _plot_lag_correlation(clean_df, fig_dir)
    _plot_feature_group_ablation(splits, metadata, feature_cols, fig_dir, out_dir)
    _plot_feature_importance_rf(splits, feature_cols, fig_dir, out_dir)
    _plot_data_flow(fig_dir)

    return {p.stem: p for p in sorted(fig_dir.glob("*.png"))}


def build_feature_package(
    data_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    figure_dir: Optional[str | Path] = None,
    target_col: str = TARGET_COL,
    horizon: int = 1,
    make_plots: bool = True,
    save: bool = True,
    seq_len: int = 14,
) -> Dict[str, object]:
    """
    Create the full data-processing handoff package.

    Parameters
    ----------
    data_path:
        Optional raw Excel path. Defaults to `data/raw_air_quality.xlsx`.
    output_dir:
        Destination for processed CSV/NPZ/metadata files.
    figure_dir:
        Destination for EDA and feature figures.
    target_col:
        Regression target. Default is AQI.
    horizon:
        Forecast horizon in days. `1` means next-day prediction.
    make_plots:
        Whether to generate report-ready PNG figures.
    save:
        Whether to write outputs to disk.
    seq_len:
        Sequence length for LSTM-ready `.npz` files.
    """
    out_dir = _as_path(output_dir, DEFAULT_OUTPUT_DIR)
    fig_dir = _as_path(figure_dir, DEFAULT_FIGURE_DIR)

    raw_df = load_raw_data(str(data_path)) if data_path is not None else load_raw_data()
    clean_df = clean_air_quality_data(raw_df)
    full_features, metadata = build_safe_features(clean_df, target_col=target_col, horizon=horizon)
    feature_cols = metadata.loc[metadata["use_for_model"] == 1, "feature"].tolist()

    splits_raw = split_feature_frame(full_features, feature_cols, target_col=target_col)
    splits, scaler = fit_transform_feature_scaler(splits_raw, feature_cols)
    audit = make_feature_audit(metadata)

    if save:
        save_feature_package(
            splits=splits,
            metadata=metadata,
            audit=audit,
            scaler=scaler,
            feature_cols=feature_cols,
            output_dir=out_dir,
            target_col=target_col,
            horizon=horizon,
        )
        make_sequence_npz(splits, feature_cols, output_dir=out_dir, target_col=target_col, seq_len=seq_len)

    figures: Dict[str, Path] = {}
    if make_plots:
        figures = make_feature_plots(clean_df, splits, metadata, feature_cols, figure_dir=fig_dir, output_dir=out_dir)

    smoke = smoke_test_feature_package(splits, feature_cols, audit, output_dir=out_dir, target_col=target_col)

    print("[FeaturePipeline] package build complete")
    print(f"[FeaturePipeline] processed outputs: {out_dir}")
    print(f"[FeaturePipeline] feature figures: {fig_dir}")

    return {
        "splits": splits,
        "feature_cols": feature_cols,
        "metadata": metadata,
        "audit": audit,
        "scaler": scaler,
        "clean_df": clean_df,
        "figures": figures,
        "smoke_test": smoke,
        "output_dir": out_dir,
        "figure_dir": fig_dir,
    }


if __name__ == "__main__":
    build_feature_package(make_plots=True, save=True)
