# -*- coding: utf-8 -*-
"""
Optuna-based feature dimension search for the AQI feature package.

Definition of "best dimension"
------------------------------
The best dimension is the top_k value in [5, 85] that minimizes validation
RMSE on the 2018 validation split, after ranking features with a training-only
RandomForestRegressor. The 2019 test split is used only once for final reporting.

This is a wrapper-style feature selection workflow: model performance is used
to evaluate candidate feature subsets, matching the feature-selection concept
from the course slides.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed"
OPTUNA_DIR = PROJECT_ROOT / "outputs" / "optuna"
FIGURE_DIR = OPTUNA_DIR / "figures"
TARGET_COL = "AQI"
RANDOM_STATE = 42


def _configure_plots() -> None:
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
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 240,
        }
    )


def _category_cn(category: str) -> str:
    """Translate feature category names for Chinese plot legends."""
    mapping = {
        "time": "时间特征",
        "lag": "滞后特征",
        "rolling": "移动统计特征",
        "trend": "趋势特征",
        "composite": "复合污染特征",
        "quality_flag": "质量标记特征",
        "unknown": "未知类别",
    }
    return mapping.get(str(category), str(category))


def load_data_package(processed_dir: Path = PROCESSED_DIR) -> Dict[str, object]:
    """Load processed train/val/test splits and feature metadata."""
    with open(processed_dir / "scaler_info.json", "r", encoding="utf-8") as f:
        scaler_info = json.load(f)
    feature_cols = scaler_info["feature_cols"]
    splits = {}
    for split in ("train", "val", "test"):
        path = processed_dir / f"{split}_features.csv"
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        splits[split] = df
    metadata = pd.read_csv(processed_dir / "feature_metadata.csv")
    return {"splits": splits, "feature_cols": feature_cols, "metadata": metadata, "scaler_info": scaler_info}


def get_xy(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = df[list(feature_cols)].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)
    return X, y


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_rf(n_estimators: int = 180, max_depth: int = 14) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def rank_features_by_rf(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Rank all features using a training-only Random Forest."""
    X_train, y_train = get_xy(train_df, feature_cols)
    model = make_rf(n_estimators=260, max_depth=16)
    model.fit(X_train, y_train)
    category_map = metadata.set_index("feature")["category"].to_dict()
    ranked = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.feature_importances_,
                "category": [category_map.get(f, "unknown") for f in feature_cols],
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def fit_eval_subset(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: Sequence[str],
    n_estimators: int = 160,
) -> Tuple[Dict[str, float], np.ndarray, RandomForestRegressor]:
    X_train, y_train = get_xy(train_df, feature_cols)
    X_eval, y_eval = get_xy(eval_df, feature_cols)
    model = make_rf(n_estimators=n_estimators, max_depth=14)
    model.fit(X_train, y_train)
    pred = model.predict(X_eval)
    return regression_metrics(y_eval, pred), pred, model


def export_feature_vector_examples(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    output_dir: Path = PROCESSED_DIR,
    n_examples: int = 3,
) -> Dict[str, Path]:
    """Export readable examples showing how one row becomes an 85-D vector."""
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = train_df.head(n_examples).copy()
    wide_cols = [TARGET_COL]
    if "quality_level" in examples.columns:
        wide_cols.append("quality_level")
    wide_cols.extend(feature_cols)
    csv_path = output_dir / "feature_vector_examples.csv"
    examples[wide_cols].reset_index().to_csv(csv_path, index=False, encoding="utf-8-sig")

    first = examples.iloc[0]
    vector = [float(first[col]) for col in feature_cols]
    payload = {
        "date": str(examples.index[0].date()),
        "target_AQI": float(first[TARGET_COL]),
        "quality_level": str(first.get("quality_level", "")),
        "feature_count": len(feature_cols),
        "feature_cols": list(feature_cols),
        "feature_vector": vector,
        "feature_name_mapping": [{"index": i, "feature": col, "value": vector[i]} for i, col in enumerate(feature_cols)],
        "note": "feature_vector values are standardized model inputs; target_AQI remains in raw AQI units.",
    }
    json_path = output_dir / "feature_vector_example.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {"csv": csv_path, "json": json_path}


def _smallest_k_covering_features(ranking: pd.DataFrame, required_features: Sequence[str], fallback: int) -> int:
    """Return the smallest top_k that contains all available required features."""
    rank_map = ranking.set_index("feature")["rank"].to_dict()
    covered_ranks = [int(rank_map[f]) for f in required_features if f in rank_map]
    if not covered_ranks:
        return fallback
    return max(max(covered_ranks), fallback)


def _smallest_k_covering_categories(ranking: pd.DataFrame, required_categories: Sequence[str], fallback: int) -> int:
    """Return the first top_k whose prefix covers the requested feature families."""
    required = set(required_categories)
    for k in range(1, len(ranking) + 1):
        categories = set(ranking.head(k)["category"].tolist())
        if required.issubset(categories):
            return max(k, fallback)
    return fallback


def build_recommended_dimension_plan(
    results_df: pd.DataFrame,
    ranking: pd.DataFrame,
    best_top_k: int,
) -> pd.DataFrame:
    """
    Build the handoff dimension plan for downstream models.

    The selected dimensions are not another hidden test-set search. They are a
    modeling handoff plan: one validation-best low-dimensional subset, one
    compact one-day pollutant state, one broader near-best feature-family subset,
    and the complete 85-D engineered-feature baseline.
    """
    all_feature_count = len(ranking)
    best_rmse = float(results_df["val_RMSE"].min())

    one_day_required = [
        "AQI_lag_1",
        "AQI_hist_diff_1",
        "pollution_index_lag1",
        "PM2_5_lag_1",
        "PM10_lag_1",
        "SO2_lag_1",
        "CO_lag_1",
        "NO2_lag_1",
        "O3_8h_lag_1",
    ]
    one_day_k = _smallest_k_covering_features(ranking, one_day_required, fallback=max(best_top_k + 1, 5))
    balanced_k = _smallest_k_covering_categories(
        ranking,
        required_categories=["lag", "rolling", "time", "trend", "composite"],
        fallback=max(one_day_k + 1, 12),
    )

    candidates = [
        {
            "role": "A_optuna_best",
            "top_k": int(best_top_k),
            "name": "验证集最优低维方案",
            "theory_basis": "验证集 RMSE 最低，代表当前 RF 排序规则下的最低噪声输入。",
            "lstm_use": "作为 LSTM 的第一组强基线，输入形状为 [样本数, 14, top_k]。",
        },
        {
            "role": "B_one_day_state",
            "top_k": int(one_day_k),
            "name": "一天历史污染状态方案",
            "theory_basis": "覆盖 AQI 昨日状态、AQI 变化量、综合污染指数以及主要污染物 lag_1，保留最直接的短期记忆。",
            "lstm_use": "适合检验 LSTM 是否仅靠紧邻历史污染状态即可学习趋势。",
        },
        {
            "role": "C_balanced_family",
            "top_k": int(balanced_k),
            "name": "特征家族均衡方案",
            "theory_basis": "覆盖 lag、rolling、time、trend、composite 多类特征，在很小的维度增加下补充周期性和移动统计信息。",
            "lstm_use": "适合检验 LSTM 在低维基础上加入季节性和窗口统计后是否更稳定。",
        },
        {
            "role": "D_full_baseline",
            "top_k": int(all_feature_count),
            "name": "完整 85 维基线方案",
            "theory_basis": "保留全部无泄漏特征，作为后续模型判断“是否需要降维”的完整参照。",
            "lstm_use": "适合配合 dropout、早停或正则化，检验深度模型能否从完整特征空间中受益。",
        },
    ]

    rows: List[Dict[str, object]] = []
    seen_topks = set()
    for item in candidates:
        top_k = int(min(max(item["top_k"], 5), all_feature_count))
        if top_k in seen_topks:
            continue
        seen_topks.add(top_k)
        metric_row = results_df.loc[results_df["top_k"].astype(int) == top_k]
        if metric_row.empty:
            continue
        metric = metric_row.iloc[0]
        selected = ranking.head(top_k)
        category_counts = selected["category"].value_counts().to_dict()
        rows.append(
            {
                **item,
                "top_k": top_k,
                "val_RMSE": float(metric["val_RMSE"]),
                "val_MAE": float(metric["val_MAE"]),
                "val_R2": float(metric["val_R2"]),
                "pct_worse_than_best_val_RMSE": float((float(metric["val_RMSE"]) / best_rmse - 1.0) * 100.0),
                "category_counts_json": json.dumps(category_counts, ensure_ascii=False),
                "first_10_features": ", ".join(selected["feature"].head(10).tolist()),
                "feature_file": f"recommended_top{top_k}_features.csv",
                "lstm_sequence_dir": str(PROCESSED_DIR / "lstm_dimension_sets" / f"top{top_k}"),
            }
        )
    return pd.DataFrame(rows)


def export_recommended_dimension_artifacts(
    recommendations: pd.DataFrame,
    ranking: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
    train_val: pd.DataFrame,
    output_dir: Path,
    seq_len: int = 14,
) -> pd.DataFrame:
    """Export feature lists, LSTM sequence files, and final test metrics for recommended dimensions."""
    try:
        from src.feature_pipeline import make_sequence_npz
    except ImportError:  # pragma: no cover - supports direct script execution from src/
        from feature_pipeline import make_sequence_npz

    rows = []
    for _, row in recommendations.iterrows():
        top_k = int(row["top_k"])
        selected_features = ranking.head(top_k)["feature"].tolist()

        feature_file = output_dir / f"recommended_top{top_k}_features.csv"
        ranking.head(top_k).to_csv(feature_file, index=False, encoding="utf-8-sig")

        seq_dir = PROCESSED_DIR / "lstm_dimension_sets" / f"top{top_k}"
        make_sequence_npz(splits, selected_features, output_dir=seq_dir, seq_len=seq_len)
        with open(seq_dir / "feature_cols.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "top_k": top_k,
                    "seq_len": seq_len,
                    "feature_count": len(selected_features),
                    "feature_cols": selected_features,
                    "shape_note": f"Each split stores X_seq with shape [samples, {seq_len}, {top_k}].",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        test_metrics, _, _ = fit_eval_subset(train_val, splits["test"], selected_features, n_estimators=260)
        out = row.to_dict()
        out.update(
            {
                "test_RMSE_train_val": test_metrics["RMSE"],
                "test_MAE_train_val": test_metrics["MAE"],
                "test_R2_train_val": test_metrics["R2"],
                "feature_file": str(feature_file),
                "lstm_sequence_dir": str(seq_dir),
            }
        )
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_dir / "recommended_dimension_results.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "recommended_dimension_plan.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return out_df


def run_optuna_search(
    n_min: int = 5,
    n_max: int | None = None,
    output_dir: Path = OPTUNA_DIR,
) -> Dict[str, object]:
    """Enumerate top_k dimensions with Optuna GridSampler and export all outputs."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _configure_plots()
    output_dir.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    package = load_data_package()
    splits: Dict[str, pd.DataFrame] = package["splits"]
    all_features: List[str] = list(package["feature_cols"])
    metadata: pd.DataFrame = package["metadata"]
    n_max = n_max or len(all_features)
    if n_min < 1 or n_max > len(all_features) or n_min > n_max:
        raise ValueError(f"Invalid top_k range: {n_min}..{n_max} for {len(all_features)} features")

    vector_paths = export_feature_vector_examples(splits["train"], all_features)

    ranking = rank_features_by_rf(splits["train"], all_features, metadata)
    ranking.to_csv(output_dir / "ranked_features_rf.csv", index=False, encoding="utf-8-sig")
    ranked_features = ranking["feature"].tolist()

    candidate_topks = list(range(n_min, n_max + 1))
    trial_rows: List[Dict[str, object]] = []
    search_space = {"top_k": candidate_topks}
    sampler = optuna.samplers.GridSampler(search_space, seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="topk_feature_dimension_search")

    def objective(trial: optuna.Trial) -> float:
        top_k = int(trial.suggest_categorical("top_k", candidate_topks))
        selected = ranked_features[:top_k]
        val_metrics, _, _ = fit_eval_subset(splits["train"], splits["val"], selected)
        trial.set_user_attr("val_MAE", val_metrics["MAE"])
        trial.set_user_attr("val_R2", val_metrics["R2"])
        return val_metrics["RMSE"]

    study.optimize(objective, n_trials=len(candidate_topks), show_progress_bar=False)

    for trial in study.trials:
        if trial.value is None:
            continue
        trial_rows.append(
            {
                "trial_number": trial.number,
                "top_k": int(trial.params["top_k"]),
                "val_RMSE": float(trial.value),
                "val_MAE": float(trial.user_attrs["val_MAE"]),
                "val_R2": float(trial.user_attrs["val_R2"]),
            }
        )

    trials_df = pd.DataFrame(trial_rows).sort_values("trial_number")
    results_df = trials_df.sort_values("top_k").reset_index(drop=True)
    trials_df.to_csv(output_dir / "optuna_trials.csv", index=False, encoding="utf-8-sig")
    results_df.to_csv(output_dir / "topk_dimension_results.csv", index=False, encoding="utf-8-sig")

    best_row = results_df.loc[results_df["val_RMSE"].idxmin()]
    best_top_k = int(best_row["top_k"])
    best_features = ranked_features[:best_top_k]
    best_subset = ranking.head(best_top_k).copy()
    best_subset.to_csv(output_dir / "best_feature_subset.csv", index=False, encoding="utf-8-sig")

    # Final fair test comparison: train on train+val after top_k is selected,
    # evaluate once on 2019 test.
    train_val = pd.concat([splits["train"], splits["val"]]).sort_index()
    best_test_metrics, best_test_pred, best_model = fit_eval_subset(train_val, splits["test"], best_features, n_estimators=260)
    full_test_metrics, full_test_pred, full_model = fit_eval_subset(train_val, splits["test"], ranked_features, n_estimators=260)
    recommendations = build_recommended_dimension_plan(results_df, ranking, best_top_k)
    recommendations = export_recommended_dimension_artifacts(
        recommendations=recommendations,
        ranking=ranking,
        splits=splits,
        train_val=train_val,
        output_dir=output_dir,
        seq_len=14,
    )

    y_test = splits["test"][TARGET_COL].to_numpy(dtype=np.float32)
    dates_test = splits["test"].index.astype(str).tolist()
    pred_df = pd.DataFrame(
        {
            "date": dates_test,
            "actual_AQI": y_test,
            "best_topk_pred": best_test_pred,
            "full85_pred": full_test_pred,
            "best_residual": y_test - best_test_pred,
            "full85_residual": y_test - full_test_pred,
        }
    )
    pred_df.to_csv(output_dir / "best_predictions_test.csv", index=False, encoding="utf-8-sig")

    final_metrics = {
        "selection_rule": "Choose top_k with minimum validation RMSE from all integer values 5..85.",
        "feature_ranking": "RandomForestRegressor feature_importances_ fitted on training split only.",
        "best_top_k": best_top_k,
        "best_validation_metrics_train_only": {
            "RMSE": float(best_row["val_RMSE"]),
            "MAE": float(best_row["val_MAE"]),
            "R2": float(best_row["val_R2"]),
        },
        "final_test_metrics_train_val_best_topk": best_test_metrics,
        "final_test_metrics_train_val_full85": full_test_metrics,
        "all_feature_count": len(all_features),
        "recommended_dimensions": recommendations.to_dict(orient="records"),
        "recommended_dimension_note": (
            "These dimensions are handoff candidates for downstream models. "
            "They are chosen from validation behavior and feature-family coverage; "
            "test metrics are reported after selection and are not used by Optuna."
        ),
        "test_split_note": "The 2019 test split is not used by Optuna; it is used only after top_k is selected.",
        "feature_vector_examples": {k: str(v) for k, v in vector_paths.items()},
    }
    with open(output_dir / "optuna_best_params.json", "w", encoding="utf-8") as f:
        json.dump({"best_top_k": best_top_k, "best_value_val_RMSE": float(best_row["val_RMSE"])}, f, ensure_ascii=False, indent=2)
    with open(output_dir / "final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    plot_all_outputs(
        results_df=results_df,
        trials_df=trials_df,
        ranking=ranking,
        best_subset=best_subset,
        metadata=metadata,
        best_top_k=best_top_k,
        best_test_metrics=best_test_metrics,
        full_test_metrics=full_test_metrics,
        y_test=y_test,
        best_pred=best_test_pred,
        full_pred=full_test_pred,
        test_dates=splits["test"].index,
        train_df=splits["train"],
        all_features=all_features,
        recommendations=recommendations,
        output_dir=output_dir,
    )

    print(f"[Optuna] Best top_k={best_top_k}, validation RMSE={float(best_row['val_RMSE']):.4f}")
    print(f"[Optuna] Final test RMSE(best top_k, train+val)={best_test_metrics['RMSE']:.4f}")
    print(f"[Optuna] Outputs: {output_dir}")
    return {
        "best_top_k": best_top_k,
        "best_validation_rmse": float(best_row["val_RMSE"]),
        "best_test_metrics": best_test_metrics,
        "full_test_metrics": full_test_metrics,
        "recommended_dimensions": recommendations.to_dict(orient="records"),
        "output_dir": output_dir,
    }


def plot_all_outputs(
    results_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    ranking: pd.DataFrame,
    best_subset: pd.DataFrame,
    metadata: pd.DataFrame,
    best_top_k: int,
    best_test_metrics: Dict[str, float],
    full_test_metrics: Dict[str, float],
    y_test: np.ndarray,
    best_pred: np.ndarray,
    full_pred: np.ndarray,
    test_dates: pd.DatetimeIndex,
    train_df: pd.DataFrame,
    all_features: Sequence[str],
    recommendations: pd.DataFrame,
    output_dir: Path,
) -> None:
    _configure_plots()
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. top_k vs validation RMSE
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(results_df["top_k"], results_df["val_RMSE"], color="#2f6f9f", linewidth=1.8)
    ax.scatter([best_top_k], [results_df.loc[results_df["top_k"] == best_top_k, "val_RMSE"].iloc[0]], color="#b23a30", zorder=5)
    ax.axvline(best_top_k, color="#b23a30", linestyle="--", linewidth=1)
    ax.set_title("Optuna 特征维度搜索：验证集 RMSE")
    ax.set_xlabel("Top-K 特征维度")
    ax.set_ylabel("2018 验证集 RMSE")
    ax.text(best_top_k, ax.get_ylim()[0], f" 最优维度={best_top_k}", color="#b23a30", va="bottom")
    fig.tight_layout()
    fig.savefig(fig_dir / "topk_rmse_curve.png", bbox_inches="tight")
    plt.close(fig)

    # 2. MAE and R2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(results_df["top_k"], results_df["val_MAE"], color="#6f8f4e", linewidth=1.6)
    axes[0].axvline(best_top_k, color="#b23a30", linestyle="--", linewidth=1)
    axes[0].set_title("验证集 MAE")
    axes[0].set_xlabel("Top-K")
    axes[0].set_ylabel("MAE")
    axes[1].plot(results_df["top_k"], results_df["val_R2"], color="#8d63a9", linewidth=1.6)
    axes[1].axvline(best_top_k, color="#b23a30", linestyle="--", linewidth=1)
    axes[1].set_title("验证集 R2")
    axes[1].set_xlabel("Top-K")
    axes[1].set_ylabel("R2")
    fig.tight_layout()
    fig.savefig(fig_dir / "topk_mae_r2_curve.png", bbox_inches="tight")
    plt.close(fig)

    # 2b. Recommended handoff dimensions
    if recommendations is not None and not recommendations.empty:
        rec = recommendations.copy()
        rec["label"] = rec["top_k"].apply(lambda k: f"top{k}")
        fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.0))
        metric_specs = [
            ("val_RMSE", "验证集 RMSE", "#5d8aa8"),
            ("test_RMSE_train_val", "最终测试集 RMSE", "#c08a5a"),
            ("test_MAE_train_val", "最终测试集 MAE", "#7aa37a"),
        ]
        for ax, (metric, title, color) in zip(axes, metric_specs):
            ax.bar(rec["label"], rec[metric], color=color)
            ax.set_title(title)
            ax.set_xlabel("推荐维度")
            ax.tick_params(axis="x", rotation=20)
            for idx, value in enumerate(rec[metric]):
                ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        fig.suptitle("面向下游模型的推荐特征维度对比")
        fig.tight_layout()
        fig.savefig(fig_dir / "recommended_dimension_comparison.png", bbox_inches="tight")
        plt.close(fig)

        category_rows = []
        for _, row in rec.iterrows():
            counts = json.loads(row["category_counts_json"])
            for category, count in counts.items():
                category_rows.append({"dimension": f"top{int(row['top_k'])}", "category": category, "count": int(count)})
        cat_df = pd.DataFrame(category_rows)
        if not cat_df.empty:
            pivot = cat_df.pivot_table(index="dimension", columns="category", values="count", aggfunc="sum", fill_value=0)
            pivot = pivot.reindex([f"top{int(k)}" for k in rec["top_k"]])
            fig, ax = plt.subplots(figsize=(8.8, 4.8))
            palette = {
                "lag": "#5d8aa8",
                "rolling": "#7aa37a",
                "time": "#c08a5a",
                "trend": "#8d63a9",
                "composite": "#b23a30",
                "quality_flag": "#7f8c8d",
            }
            bottom = np.zeros(len(pivot))
            for category in pivot.columns:
                values = pivot[category].to_numpy()
                ax.bar(pivot.index, values, bottom=bottom, label=_category_cn(category), color=palette.get(category, "#999999"))
                bottom += values
            ax.set_title("推荐维度中的特征类别覆盖情况")
            ax.set_xlabel("推荐维度")
            ax.set_ylabel("特征数量")
            ax.legend(ncol=3, fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / "recommended_dimension_category_coverage.png", bbox_inches="tight")
            plt.close(fig)

    # 3. Optuna optimization history
    ordered = trials_df.sort_values("trial_number").copy()
    ordered["best_so_far"] = ordered["val_RMSE"].cummin()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(ordered["trial_number"], ordered["val_RMSE"], s=18, alpha=0.65, color="#6a92b8", label="单次试验 RMSE")
    ax.plot(ordered["trial_number"], ordered["best_so_far"], color="#b23a30", linewidth=1.8, label="截至当前最优")
    ax.set_title("Optuna 搜索历史")
    ax.set_xlabel("试验编号")
    ax.set_ylabel("验证集 RMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "optuna_optimization_history.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Best vs full metrics
    metric_rows = []
    for label, metrics in [("最佳 Top-K", best_test_metrics), ("完整 85 维", full_test_metrics)]:
        for metric in ["RMSE", "MAE", "R2"]:
            metric_rows.append({"model": label, "metric": metric, "value": metrics[metric]})
    metric_df = pd.DataFrame(metric_rows)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8))
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        sub = metric_df[metric_df["metric"] == metric]
        ax.bar(sub["model"], sub["value"], color=["#5d8aa8", "#c08a5a"])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("最终测试集指标：最佳 Top-K 与完整 85 维对比")
    fig.tight_layout()
    fig.savefig(fig_dir / "best_vs_full_metrics.png", bbox_inches="tight")
    plt.close(fig)

    # 5. Best category distribution
    cat_counts = best_subset["category"].value_counts().reset_index()
    cat_counts.columns = ["category", "count"]
    cat_counts["category_cn"] = cat_counts["category"].map(_category_cn)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.bar(cat_counts["category_cn"], cat_counts["count"], color="#7aa37a")
    ax.set_title(f"最佳特征子集类别分布（top_k={best_top_k}）")
    ax.set_xlabel("特征类别")
    ax.set_ylabel("数量")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(fig_dir / "best_feature_category_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 6. Best subset importances from final model
    train_val = pd.concat([load_data_package()["splits"]["train"], load_data_package()["splits"]["val"]]).sort_index()
    best_features = best_subset["feature"].tolist()
    _, _, final_model = fit_eval_subset(train_val, load_data_package()["splits"]["test"], best_features, n_estimators=260)
    final_imp = (
        pd.DataFrame({"feature": best_features, "importance": final_model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    final_imp.to_csv(output_dir / "best_subset_final_importance.csv", index=False, encoding="utf-8-sig")
    top_imp = final_imp.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_imp["feature"], top_imp["importance"], color="#6a92b8")
    ax.set_title("最佳 Top-K 子集特征重要性 Top 20")
    ax.set_xlabel("重要性")
    fig.tight_layout()
    fig.savefig(fig_dir / "best_feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    # 7. Prediction vs actual
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(test_dates, y_test, label="真实 AQI", color="#1f2933", linewidth=1.5)
    ax.plot(test_dates, best_pred, label=f"最佳 Top-{best_top_k}", color="#2f6f9f", linewidth=1.4)
    ax.plot(test_dates, full_pred, label="完整 85 维", color="#c08a5a", linewidth=1.1, alpha=0.8)
    ax.set_title("2019 测试集预测对比：真实值 vs 预测值")
    ax.set_xlabel("日期")
    ax.set_ylabel("AQI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "best_prediction_vs_actual.png", bbox_inches="tight")
    plt.close(fig)

    # 8. Residual analysis
    residuals = y_test - best_pred
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    axes[0, 0].scatter(best_pred, residuals, s=12, alpha=0.6, color="#5d8aa8")
    axes[0, 0].axhline(0, color="#b23a30", linestyle="--")
    axes[0, 0].set_title("残差与预测值关系")
    axes[0, 0].set_xlabel("预测 AQI")
    axes[0, 0].set_ylabel("残差")
    axes[0, 1].plot(test_dates, residuals, color="#5d8aa8", linewidth=1.0)
    axes[0, 1].axhline(0, color="#b23a30", linestyle="--")
    axes[0, 1].set_title("残差随时间变化")
    axes[0, 1].set_xlabel("日期")
    axes[0, 1].set_ylabel("残差")
    axes[1, 0].hist(residuals, bins=35, color="#7aa37a", alpha=0.8, edgecolor="white")
    axes[1, 0].set_title("残差分布")
    axes[1, 0].set_xlabel("残差")
    abs_err = np.abs(residuals)
    axes[1, 1].plot(test_dates, abs_err, color="#c08a5a", linewidth=1.0)
    axes[1, 1].set_title("绝对误差随时间变化")
    axes[1, 1].set_xlabel("日期")
    axes[1, 1].set_ylabel("|误差|")
    fig.tight_layout()
    fig.savefig(fig_dir / "best_residual_analysis.png", bbox_inches="tight")
    plt.close(fig)

    # 9. Feature vector example visualization
    first = train_df.iloc[0]
    values = np.array([float(first[col]) for col in all_features])
    top_idx = np.argsort(np.abs(values))[-25:]
    vec_df = pd.DataFrame({"feature": np.array(all_features)[top_idx], "value": values[top_idx]})
    vec_df = vec_df.sort_values("value")
    fig, ax = plt.subplots(figsize=(8, 6.2))
    colors = ["#b23a30" if v < 0 else "#2f6f9f" for v in vec_df["value"]]
    ax.barh(vec_df["feature"], vec_df["value"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("单个样本特征向量：标准化绝对值 Top 25")
    ax.set_xlabel("标准化后的特征值")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_vector_example.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_optuna_search()
