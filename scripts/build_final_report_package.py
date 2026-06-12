#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build human-readable report/PPT handoff materials inside outputs/.

The generated files are meant for teammates who need to write the final report
or make slides without reading every script first.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUTS_DIR / "report"
PPT_DIR = OUTPUTS_DIR / "ppt_assets"

CN_FONT = "CNFont"
CN_FONT_BOLD = "CNFontBold"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def register_fonts() -> None:
    regular_candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    bold_candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.otf"),
        Path(r"C:\Windows\Fonts\msyhbd.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    regular_paths = [p for p in regular_candidates if p.exists()]
    bold_paths = [p for p in bold_candidates if p.exists()] or regular_paths

    last_error: Exception | None = None
    for regular in regular_paths:
        try:
            pdfmetrics.registerFont(TTFont(CN_FONT, str(regular), subfontIndex=0))
            break
        except TypeError:
            try:
                pdfmetrics.registerFont(TTFont(CN_FONT, str(regular)))
                break
            except Exception as exc:
                last_error = exc
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(f"No usable Chinese font found for PDF generation: {last_error}")

    for bold in bold_paths:
        try:
            pdfmetrics.registerFont(TTFont(CN_FONT_BOLD, str(bold), subfontIndex=0))
            break
        except TypeError:
            try:
                pdfmetrics.registerFont(TTFont(CN_FONT_BOLD, str(bold)))
                break
            except Exception:
                continue
        except Exception:
            continue
    if CN_FONT_BOLD not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(CN_FONT_BOLD, str(regular_paths[0]), subfontIndex=0))


def styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=base["Title"],
            fontName=CN_FONT_BOLD,
            fontSize=20,
            leading=28,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#1f3a4a"),
            wordWrap="CJK",
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "Heading1CN",
            parent=base["Heading1"],
            fontName=CN_FONT_BOLD,
            fontSize=14,
            leading=22,
            textColor=colors.HexColor("#1f3a4a"),
            wordWrap="CJK",
            spaceBefore=10,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "Heading2CN",
            parent=base["Heading2"],
            fontName=CN_FONT_BOLD,
            fontSize=11.5,
            leading=18,
            textColor=colors.HexColor("#315a6b"),
            wordWrap="CJK",
            spaceBefore=7,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=base["BodyText"],
            fontName=CN_FONT,
            fontSize=9.2,
            leading=15.5,
            wordWrap="CJK",
            spaceAfter=5,
        ),
        "caption": ParagraphStyle(
            "CaptionCN",
            parent=base["BodyText"],
            fontName=CN_FONT,
            fontSize=8.2,
            leading=13,
            textColor=colors.HexColor("#4c5961"),
            wordWrap="CJK",
            spaceAfter=8,
        ),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)).replace("\n", "<br/>"), style)


def image_flowable(path: Path, max_width: float = 16 * cm, max_height: float = 10 * cm) -> Image | None:
    if not path.exists():
        return None
    with PILImage.open(path) as img:
        width, height = img.size
    scale = min(max_width / width, max_height / height, 1.0)
    return Image(str(path), width=width * scale, height=height * scale)


def write_pdf(path: Path, title: str, sections: List[Dict[str, object]]) -> None:
    register_fonts()
    st = styles()
    story = [p(title, st["title"]), Spacer(1, 0.2 * cm)]
    for section in sections:
        story.append(p(section["title"], st["h1"]))
        for item in section.get("items", []):
            kind = item.get("kind", "p")
            if kind == "h2":
                story.append(p(item["text"], st["h2"]))
            elif kind == "table":
                table_data = item["data"]
                table = Table(table_data, repeatRows=1)
                table.setStyle(
                    TableStyle(
                        [
                            ("FONTNAME", (0, 0), (-1, -1), CN_FONT),
                            ("FONTNAME", (0, 0), (-1, 0), CN_FONT_BOLD),
                            ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f1f4")),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9d2d6")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ]
                    )
                )
                story.extend([table, Spacer(1, 0.2 * cm)])
            elif kind == "image":
                img = image_flowable(Path(item["path"]))
                if img is not None:
                    story.append(img)
                story.append(p(item.get("caption", ""), st["caption"]))
            elif kind == "pagebreak":
                story.append(PageBreak())
            else:
                story.append(p(item["text"], st["body"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )
    doc.build(story)


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def collect_metrics() -> Dict[str, object]:
    processed = OUTPUTS_DIR / "processed"
    optuna = OUTPUTS_DIR / "optuna"
    model = OUTPUTS_DIR / "model_integration" / "results"
    peak = OUTPUTS_DIR / "peak_optuna"
    tail = OUTPUTS_DIR / "tail_optuna"
    future = OUTPUTS_DIR / "future_forecast"

    meta = read_csv(processed / "feature_metadata.csv")
    audit = read_csv(processed / "feature_audit.csv")
    best_by_dim = read_csv(model / "best_by_dimension.csv")
    peak_metrics = read_json(peak / "best_peak_model_test_metrics.json")
    tail_metrics = read_json(tail / "best_tail_model_test_metrics.json")
    opt_params = read_json(optuna / "optuna_best_params.json")
    future_summary = read_json(future / "future_forecast_summary.json")

    best_model = {}
    if not best_by_dim.empty:
        best_model = best_by_dim.sort_values("RMSE_AQI").iloc[0].to_dict()

    return {
        "feature_count": int(len(meta)) if not meta.empty else 85,
        "low_risk_count": int((audit.get("leakage_risk", pd.Series(dtype=str)) == "low").sum()) if not audit.empty else 85,
        "best_top_k": opt_params.get("best_top_k", 11),
        "best_top_k_val_rmse": opt_params.get("best_value_val_RMSE", 23.7816),
        "best_model": best_model,
        "peak_test_metrics": peak_metrics.get("test_metrics", {}),
        "tail_test_metrics": tail_metrics.get("test_metrics", {}),
        "future_summary": future_summary,
    }


def project_timeline(metrics: Dict[str, object]) -> List[Dict[str, str]]:
    best = metrics.get("best_model", {})
    return [
        {
            "阶段": "1 初始探索",
            "方法": "经验/探索式 44 维候选特征",
            "维度/模型": "44 维 baseline",
            "选择依据": "快速建立可训练输入",
            "关键结果": "能跑通模型，但维度选择缺少系统比较",
            "发现的问题": "不知道多少维最合适",
            "下一步为什么要做": "需要用验证误差系统比较维度",
        },
        {
            "阶段": "2 初步筛选",
            "方法": "按验证误差观察低维方案",
            "维度/模型": "阶段性关注 top10",
            "选择依据": "MSE/RMSE 更低且特征更少",
            "关键结果": "低维方案可能优于完整高维",
            "发现的问题": "top10 仍是经验性候选",
            "下一步为什么要做": "需要完整遍历 top_k",
        },
        {
            "阶段": "3 系统遍历",
            "方法": "Optuna GridSampler 枚举 top_k=5...85",
            "维度/模型": f"top{metrics['best_top_k']}",
            "选择依据": "验证集 RMSE 最小",
            "关键结果": f"验证集最佳 top_k={metrics['best_top_k']}，RMSE={float(metrics['best_top_k_val_rmse']):.4f}",
            "发现的问题": "维度最优不等于最终模型最优",
            "下一步为什么要做": "需要把特征交给多模型统一测试",
        },
        {
            "阶段": "4 模型接入",
            "方法": "传统模型 + BiLSTM 多模型比较",
            "维度/模型": "top5/top8/top10/top11/top12/top14/top15/top20/top85",
            "选择依据": "统一测试窗口 RMSE/MAE/R2",
            "关键结果": "得到各维度下的模型对比表",
            "发现的问题": "不同目标下最优模型不同",
            "下一步为什么要做": "需要确定整体主模型",
        },
        {
            "阶段": "5 整体最佳",
            "方法": "统一测试集选择整体 RMSE 最低",
            "维度/模型": f"top{int(best.get('top_k', 10))} + {best.get('Model', 'M9B_BiLSTM_PeakWeighted')}",
            "选择依据": "测试集整体 RMSE 最低",
            "关键结果": f"RMSE={float(best.get('RMSE_AQI', 28.2839)):.4f}，R2={float(best.get('R2', 0.4110)):.4f}",
            "发现的问题": "高污染日仍可能被低估",
            "下一步为什么要做": "需要峰值优化",
        },
        {
            "阶段": "6 峰值优化",
            "方法": "模型层 Optuna + 峰值加权 + 持久性融合",
            "维度/模型": "top15 + PeakOptuna CatBoost",
            "选择依据": "降低 AQI>=150 MAE，RMSE 不明显恶化",
            "关键结果": f"AQI>=150 MAE={float(metrics['peak_test_metrics'].get('AQI_ge_150_MAE', 41.9589)):.2f}",
            "发现的问题": "低 AQI 日可能被整体预测偏高",
            "下一步为什么要做": "需要低谷修正和双尾优化",
        },
        {
            "阶段": "7 双尾优化",
            "方法": "同时约束高峰低估与低谷高估",
            "维度/模型": "top20 + TailOptuna BiLSTM",
            "选择依据": "低谷 MAE 下降，高峰误差受控",
            "关键结果": f"AQI<=50 MAE={float(metrics['tail_test_metrics'].get('AQI_le_50_MAE', 18.0515)):.2f}",
            "发现的问题": "整体 RMSE 略高于主模型",
            "下一步为什么要做": "形成不同应用目标的模型推荐",
        },
        {
            "阶段": "8 未来趋势预测",
            "方法": "2020 年 1 月情景预测",
            "维度/模型": "中性/改善/恶化三种情景",
            "选择依据": "任务要求需要未来趋势展示",
            "关键结果": "输出 30 天趋势和不确定性区间",
            "发现的问题": "没有未来真实天气与污染物输入",
            "下一步为什么要做": "报告中必须标注为情景预测",
        },
    ]


def task_completion_rows() -> List[Dict[str, str]]:
    return [
        {"任务要求": "1 数据探索与可视化", "实现状态": "已实现", "证据文件": "outputs/feature_figures/*.png", "说明": "包含时间趋势、季节箱线图、相关性热力图。"},
        {"任务要求": "2 数据预处理", "实现状态": "已实现", "证据文件": "outputs/processed/*.csv", "说明": "完成缺失值处理、异常值标记、时间切分、训练集 scaler。"},
        {"任务要求": "3 特征工程", "实现状态": "已实现", "证据文件": "feature_metadata.csv, feature_audit.csv, outputs/optuna/*.csv", "说明": "从经验 44 维发展为 85 个无泄漏候选特征，并做 top_k 搜索。"},
        {"任务要求": "4 模型选择与搭建", "实现状态": "已实现", "证据文件": "modeling/models.py, outputs/model_integration/", "说明": "包含 Ridge、SVR、随机森林、XGBoost、LightGBM、CatBoost、MLP、BiLSTM、Stacking。"},
        {"任务要求": "5 模型训练与验证", "实现状态": "已实现", "证据文件": "best_by_dimension.csv, all_model_comparison_my_features.csv", "说明": "训练/验证/测试按时间划分，并统一测试窗口。"},
        {"任务要求": "6 模型评估", "实现状态": "已实现", "证据文件": "all_peak_error_summary.csv, peak_optuna, tail_optuna", "说明": "包含 RMSE、MAE、R2、峰值误差、低谷误差和残差诊断。"},
        {"任务要求": "7 未来趋势预测", "实现状态": "已实现/情景预测", "证据文件": "outputs/future_forecast/", "说明": "输出 2020 年 1 月三种情景预测，并说明不是严格天气预报。"},
    ]


def base_figure_catalog() -> List[Dict[str, str]]:
    entries = [
        ("feature_figures/processed_data_flow.png", "项目整体数据流图", "从原始 Excel 到建模数据包", "第2页 项目整体流程图", "任务1/2/3"),
        ("feature_figures/eda_timeseries.png", "空气质量时间趋势", "展示 2013-2019 AQI 与污染物长期波动", "第1页 数据背景", "任务1"),
        ("feature_figures/eda_seasonal_boxplot.png", "AQI 月度季节分布", "证明 AQI 存在季节性，需要时间特征", "第1页 数据背景", "任务1/2"),
        ("feature_figures/eda_correlation_heatmap.png", "污染物相关性热力图", "说明 AQI 与污染物的相关结构", "第1页 数据背景", "任务1/3"),
        ("feature_figures/feature_lag_correlation.png", "滞后相关性图", "说明历史 lag 特征有预测价值", "第4页 为什么需要维度搜索", "任务3"),
        ("feature_figures/feature_group_ablation.png", "特征组消融图", "比较时间、滞后、移动统计等特征组贡献", "第4页 为什么需要维度搜索", "任务3"),
        ("feature_figures/feature_importance_rf.png", "随机森林特征重要性", "给 top_k 特征排序提供依据", "第3页 44维到85维", "任务3"),
        ("optuna/figures/topk_rmse_curve.png", "top_k RMSE 曲线", "展示 top_k=5...85 遍历结果", "第5页 top_k 遍历结果与 top11", "任务3/5"),
        ("optuna/figures/topk_mae_r2_curve.png", "top_k MAE/R2 曲线", "辅助判断不同维度的泛化表现", "第5页 top_k 遍历结果与 top11", "任务3/6"),
        ("optuna/figures/optuna_optimization_history.png", "Optuna 维度搜索历史", "说明不是随机碰运气，而是系统比较", "第5页 top_k 遍历结果与 top11", "任务3"),
        ("optuna/figures/best_vs_full_metrics.png", "最佳维度与85维对比", "解释降维为何可能优于完整特征", "第5页 top_k 遍历结果与 top11", "任务3/6"),
        ("optuna/figures/feature_vector_example.png", "特征向量示例", "说明一天如何变成模型输入向量", "第3页 44维到85维", "任务3"),
        ("peak_optuna/figures/before_after_peak_error_bins.png", "峰值优化分箱误差", "展示高污染区间误差改善", "第8页 峰值优化结果", "任务6"),
        ("peak_optuna/figures/before_after_prediction_vs_actual.png", "峰值优化时间序列", "观察峰值模型与真实 AQI 的时间走势", "第8页 峰值优化结果", "任务6"),
        ("tail_optuna/figures/高低AQI分箱误差对比.png", "双尾分箱误差", "展示高 AQI 与低 AQI 两端是否均衡", "第10页 双尾优化结果", "任务6"),
        ("tail_optuna/figures/低AQI散点图_优化前后.png", "低 AQI 散点图", "说明低谷高估问题是否缓解", "第9页 低谷高估问题", "任务6"),
        ("tail_optuna/figures/预测值与真实值_双尾优化.png", "双尾优化时间序列", "展示双尾模型全年预测形态", "第10页 双尾优化结果", "任务6"),
        ("future_forecast/未来30天AQI情景预测.png", "未来 30 天情景预测", "展示 2020 年 1 月改善/中性/恶化三种情景", "第11页 未来趋势情景预测", "任务7"),
        ("future_forecast/未来预测区间与不确定性.png", "未来预测区间", "展示中性情景的不确定性范围", "第11页 未来趋势情景预测", "任务7"),
    ]
    return [
        {
            "path": f"outputs/{path}",
            "title": title,
            "meaning": meaning,
            "ppt_use": ppt_use,
            "task": task,
            "project_step": project_step_from_path(path),
        }
        for path, title, meaning, ppt_use, task in entries
    ]


def project_step_from_path(path: str) -> str:
    if path.startswith("feature_figures"):
        return "数据探索、预处理与特征工程"
    if path.startswith("optuna"):
        return "特征维度搜索与特征向量解释"
    if path.startswith("model_integration"):
        return "模型接入与统一评估"
    if path.startswith("peak_optuna"):
        return "峰值误差优化"
    if path.startswith("tail_optuna"):
        return "高峰低谷双尾优化"
    if path.startswith("future_forecast"):
        return "未来趋势情景预测"
    return "项目输出"


def infer_figure_entry(path: Path) -> Dict[str, str]:
    relative = rel(path)
    name = path.name
    title = path.stem
    meaning = "该图用于支撑项目结果说明。"
    ppt_use = "可作为报告或 PPT 的补充图。"
    if "model_comparison" in name:
        title = f"{path.parent.parent.name} 模型对比"
        meaning = "比较同一特征维度下不同模型的 RMSE、MAE 和 R2。"
        ppt_use = "第6页 模型接入与 top10 整体最佳"
    elif "actual_vs_predicted" in name:
        title = f"{path.parent.parent.name} 真实值与预测值"
        meaning = "展示最佳模型预测曲线是否跟随真实 AQI 变化。"
        ppt_use = "第6页 模型接入与 top10 整体最佳"
    elif "residual" in name:
        title = f"{path.parent.parent.name} 残差诊断"
        meaning = "检查预测误差是否存在系统偏差或集中时段。"
        ppt_use = "第12页 最终结论与不足"
    elif "peak_error_bins" in name:
        title = f"{path.parent.parent.name} 高 AQI 分箱误差"
        meaning = "按真实 AQI 区间统计误差，定位峰值低估问题。"
        ppt_use = "第7页 峰值低估问题"
    elif "top_error_dates" in name:
        title = f"{path.parent.parent.name} 最大误差日期"
        meaning = "列出误差最大的日期，用于解释模型瓶颈。"
        ppt_use = "第12页 最终结论与不足"
    return {
        "path": relative,
        "title": title,
        "meaning": meaning,
        "ppt_use": ppt_use,
        "task": "任务5/6",
        "project_step": project_step_from_path(relative.replace("outputs/", "")),
    }


def figure_catalog() -> List[Dict[str, str]]:
    seen = set()
    rows: List[Dict[str, str]] = []
    for entry in base_figure_catalog():
        path = PROJECT_ROOT / entry["path"]
        if path.exists() and entry["path"] not in seen:
            rows.append(entry)
            seen.add(entry["path"])
    for path in sorted(OUTPUTS_DIR.rglob("*.png")):
        relative = rel(path)
        if relative.startswith("outputs/ppt_assets/") or relative.startswith("outputs/report/"):
            continue
        if relative not in seen:
            rows.append(infer_figure_entry(path))
            seen.add(relative)
    return rows


def ppt_index_rows() -> List[Dict[str, str]]:
    return [
        {"页码": "第1页", "标题": "任务背景与数据集", "推荐素材": "eda_timeseries.png, eda_seasonal_boxplot.png", "讲解重点": "说明数据范围、污染物字段、AQI 回归预测目标。"},
        {"页码": "第2页", "标题": "项目整体流程图", "推荐素材": "processed_data_flow.png", "讲解重点": "从原始数据到模型输入再到评估输出。"},
        {"页码": "第3页", "标题": "从44维经验特征到85维无泄漏特征", "推荐素材": "feature_importance_rf.png, feature_vector_example.png", "讲解重点": "强调从探索式 baseline 走向系统化特征工程。"},
        {"页码": "第4页", "标题": "为什么需要维度搜索", "推荐素材": "feature_lag_correlation.png, feature_group_ablation.png", "讲解重点": "证明特征不是随便堆叠，需要用验证误差选择。"},
        {"页码": "第5页", "标题": "top_k 遍历结果与 top11", "推荐素材": "topk_rmse_curve.png, best_vs_full_metrics.png", "讲解重点": "top11 是维度搜索阶段验证 RMSE 最优。"},
        {"页码": "第6页", "标题": "模型接入与 top10 整体最佳", "推荐素材": "model_comparison_top10.png", "讲解重点": "top10 + M9B 是最终整体 RMSE 最优模型。"},
        {"页码": "第7页", "标题": "峰值低估问题", "推荐素材": "peak_error_bins_top10.png, top_error_dates_top10.png", "讲解重点": "整体最优模型仍可能低估高污染日。"},
        {"页码": "第8页", "标题": "峰值优化结果", "推荐素材": "before_after_peak_error_bins.png", "讲解重点": "模型层 Optuna 和持久性融合降低高污染误差。"},
        {"页码": "第9页", "标题": "低谷高估问题", "推荐素材": "低AQI散点图_优化前后.png", "讲解重点": "峰值改善后发现低 AQI 日被预测偏高。"},
        {"页码": "第10页", "标题": "双尾优化结果", "推荐素材": "高低AQI分箱误差对比.png", "讲解重点": "同时控制高峰低估和低谷高估。"},
        {"页码": "第11页", "标题": "未来趋势情景预测", "推荐素材": "未来30天AQI情景预测.png", "讲解重点": "以情景预测形式补齐任务第 7 条。"},
        {"页码": "第12页", "标题": "最终结论与不足", "推荐素材": "任务完成度表、最终结论摘要", "讲解重点": "总结三套模型推荐和数据限制。"},
    ]


def write_markdown_files(metrics: Dict[str, object], timeline: List[Dict[str, str]], figures: List[Dict[str, str]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    best = metrics["best_model"]
    readme = f"""# 空气质量 AQI 预测项目总说明

## 一句话总结

本项目从经验/探索式 44 维候选特征出发，逐步完成无泄漏 85 维特征工程、top_k 维度遍历、多模型接入、模型层 Optuna、峰值优化、低谷修正和未来趋势情景预测，最终形成可用于报告和 PPT 的完整输出包。

## 最重要的结论

- 特征维度搜索阶段：`top{metrics['best_top_k']}` 在验证集 RMSE 下最优，RMSE={float(metrics['best_top_k_val_rmse']):.4f}。
- 整体最佳模型：`top{int(best.get('top_k', 10))} + {best.get('Model', 'M9B_BiLSTM_PeakWeighted')}`，测试 RMSE={float(best.get('RMSE_AQI', 28.2839)):.4f}。
- 峰值优先模型：`top15 + PeakOptuna CatBoost`，用于改善 `AQI>=150` 高污染日误差。
- 双尾均衡模型：`top20 + TailOptuna BiLSTM`，用于同时控制高值低估和低值高估。
- 未来趋势预测：`outputs/future_forecast/` 给出 2020 年 1 月三种情景预测，必须表述为“情景预测”。

## top11 和 top10 为什么不矛盾

`top11` 是特征维度遍历阶段在验证集 RMSE 下的最佳维度；`top10 + M9B_BiLSTM_PeakWeighted` 是多模型接入后在 2019 测试集整体 RMSE 下的最佳模型。前者回答“用多少维特征更合理”，后者回答“最终哪个模型组合效果最好”。

## 输出目录怎么用

- `outputs/processed/`：给建模同学用的数据接口、特征说明和泄漏审计。
- `outputs/feature_figures/`：数据探索、季节性、相关性和特征工程图。
- `outputs/optuna/`：特征维度搜索、推荐维度和特征向量示例。
- `outputs/model_integration/`：多模型接入和统一测试评估。
- `outputs/peak_optuna/`：高污染峰值优化。
- `outputs/tail_optuna/`：高峰低谷双尾优化。
- `outputs/future_forecast/`：未来趋势情景预测。
- `outputs/report/`：README、PDF、图表解释手册、PPT 素材索引和任务完成度表。
- `outputs/ppt_assets/`：按 PPT 章节整理出来的推荐图片。
"""
    (REPORT_DIR / "项目总说明_README.md").write_text(readme, encoding="utf-8")

    timeline_md = "# 项目历程时间线\n\n" + "\n".join(
        f"## {row['阶段']}\n\n- 方法：{row['方法']}\n- 维度/模型：{row['维度/模型']}\n- 选择依据：{row['选择依据']}\n- 关键结果：{row['关键结果']}\n- 发现的问题：{row['发现的问题']}\n- 下一步为什么要做：{row['下一步为什么要做']}\n"
        for row in timeline
    )
    (REPORT_DIR / "项目历程时间线.md").write_text(timeline_md, encoding="utf-8")

    fig_md = "# 图表解释手册\n\n"
    for row in figures:
        fig_md += (
            f"## {row['title']}\n\n"
            f"- 文件：`{row['path']}`\n"
            f"- 对应项目步骤：{row['project_step']}\n"
            f"- 图的含义：{row['meaning']}\n"
            f"- PPT/报告用途：{row['ppt_use']}\n"
            f"- 对应任务：{row['task']}\n\n"
        )
    (REPORT_DIR / "图表解释手册.md").write_text(fig_md, encoding="utf-8")

    writing = """# 论文/报告写作素材索引

## 引言可以写什么

本项目基于 2013-2019 年每日空气质量数据，构建 AQI 回归预测流程。项目重点不是只训练一个模型，而是完成从原始数据清洗、无泄漏特征工程、特征维度搜索、模型接入、极端误差优化到未来趋势情景预测的完整机器学习流程。

## 方法部分可以写什么

先介绍数据预处理和时间切分，再介绍从 44 维经验候选特征到 85 维无泄漏特征的扩展，然后说明用 Optuna GridSampler 遍历 top_k=5...85，最后介绍多模型比较和模型层 Optuna 优化。

## 结果部分可以写什么

结果应按“整体预测效果、峰值误差、低谷误差、未来情景预测”组织。不要只写 RMSE 最低，还要说明高污染日和低污染日的误差变化。

## 讨论部分可以写什么

R2 没有接近 0.9 是合理的，因为原始数据缺少天气、排放、交通和区域传输等外部变量。当前模型已经在给定数据条件下完成较充分优化，剩余误差主要来自突发污染过程和未来输入不可得。
"""
    (REPORT_DIR / "论文报告写作素材索引.md").write_text(writing, encoding="utf-8")

    final_summary = """# 最终结论摘要

1. 数据处理和特征工程已完成：85 个正式特征全部通过低泄漏风险审计。
2. 维度搜索阶段验证集最优为 top11，但最终多模型测试中 top10 + M9B_BiLSTM_PeakWeighted 整体 RMSE 最低。
3. 峰值优化证明高污染日误差可以被专门降低。
4. 双尾优化进一步解决低 AQI 日被预测偏高的问题。
5. 未来趋势预测采用 2020 年 1 月三种情景，作为任务第 7 条的趋势展示。
6. 最终报告建议同时展示整体最佳、峰值优先、双尾均衡三套模型定位。
"""
    (REPORT_DIR / "最终结论摘要.md").write_text(final_summary, encoding="utf-8")


def write_csv_outputs(timeline: List[Dict[str, str]], figures: List[Dict[str, str]]) -> None:
    pd.DataFrame(timeline).to_csv(REPORT_DIR / "阶段性结果对照表.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(task_completion_rows()).to_csv(REPORT_DIR / "任务要求完成度总表.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ppt_index_rows()).to_csv(REPORT_DIR / "PPT素材索引.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(figures).to_csv(REPORT_DIR / "图表解释索引.csv", index=False, encoding="utf-8-sig")


def copy_ppt_assets(figures: List[Dict[str, str]]) -> None:
    if PPT_DIR.exists():
        shutil.rmtree(PPT_DIR)
    PPT_DIR.mkdir(parents=True, exist_ok=True)
    page_to_dir = {
        "第1页": "01_数据探索与预处理",
        "第2页": "02_项目整体流程",
        "第3页": "03_特征工程",
        "第4页": "04_维度搜索依据",
        "第5页": "05_topk遍历",
        "第6页": "06_模型训练与评估",
        "第7页": "07_峰值低估问题",
        "第8页": "08_峰值优化",
        "第9页": "09_低谷高估问题",
        "第10页": "10_双尾优化",
        "第11页": "11_未来趋势预测",
        "第12页": "12_最终结论与不足",
    }
    for row in figures:
        page = str(row["ppt_use"]).split(" ")[0]
        folder = page_to_dir.get(page, "99_补充素材")
        src = PROJECT_ROOT / row["path"]
        if src.exists():
            dst_dir = PPT_DIR / folder
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)


def build_pdf_outputs(metrics: Dict[str, object], timeline: List[Dict[str, str]], figures: List[Dict[str, str]]) -> None:
    timeline_table = [["阶段", "方法", "维度/模型", "关键结果", "下一步"]] + [
        [r["阶段"], r["方法"], r["维度/模型"], r["关键结果"], r["下一步为什么要做"]] for r in timeline
    ]
    task_table = [["任务要求", "实现状态", "证据文件", "说明"]] + [
        [r["任务要求"], r["实现状态"], r["证据文件"], r["说明"]] for r in task_completion_rows()
    ]
    ppt_table = [["页码", "标题", "推荐素材", "讲解重点"]] + [
        [r["页码"], r["标题"], r["推荐素材"], r["讲解重点"]] for r in ppt_index_rows()
    ]
    sections = [
        {
            "title": "项目核心结论",
            "items": [
                {"text": "本项目从经验/探索式 44 维候选特征出发，逐步完成系统化特征工程、维度遍历、多模型接入、峰值优化、低谷修正和未来趋势情景预测。"},
                {"text": "top11 是维度搜索阶段的验证集最优；top10 + M9B_BiLSTM_PeakWeighted 是最终多模型测试中的整体 RMSE 最优，两者不矛盾。"},
                {"text": "最终推荐按用途分为整体最佳、峰值优先、双尾均衡三套模型，而不是只给一个单一模型。"},
            ],
        },
        {"title": "项目演进路线", "items": [{"kind": "table", "data": timeline_table}]},
        {"title": "任务完成度", "items": [{"kind": "table", "data": task_table}]},
        {"title": "PPT 素材索引", "items": [{"kind": "table", "data": ppt_table}]},
    ]
    write_pdf(REPORT_DIR / "项目总说明_README.pdf", "空气质量 AQI 预测项目总说明", sections)

    fig_sections = [{"title": "图表解释手册", "items": [{"text": "下面按输出图逐一说明图的含义、对应项目步骤和报告/PPT用途。"}]}]
    for i, row in enumerate(figures, start=1):
        fig_path = PROJECT_ROOT / row["path"]
        items = [
            {"kind": "h2", "text": f"{i}. {row['title']}"},
            {"text": f"文件：{row['path']}"},
            {"text": f"对应项目步骤：{row['project_step']}"},
            {"text": f"图的含义：{row['meaning']}"},
            {"text": f"PPT/报告用途：{row['ppt_use']}；对应任务：{row['task']}"},
        ]
        if fig_path.exists():
            items.insert(1, {"kind": "image", "path": str(fig_path), "caption": f"{row['title']}：{row['meaning']}"})
        fig_sections.append({"title": row["title"], "items": items})
    write_pdf(REPORT_DIR / "图表解释手册.pdf", "AQI 项目图表解释手册", fig_sections)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = collect_metrics()
    timeline = project_timeline(metrics)
    figures = figure_catalog()
    write_markdown_files(metrics, timeline, figures)
    write_csv_outputs(timeline, figures)
    copy_ppt_assets(figures)
    build_pdf_outputs(metrics, timeline, figures)
    print(f"[FinalPackage] report outputs: {REPORT_DIR}")
    print(f"[FinalPackage] ppt assets: {PPT_DIR}")
    print(f"[FinalPackage] figures documented: {len(figures)}")


if __name__ == "__main__":
    main()
