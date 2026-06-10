# 空气质量 AQI 预测：特征工程与模型对接项目

这是一个面向 Google Colab 的小组作业项目版本，包含：

- 2013-2019 年每日空气质量原始数据；
- 无泄漏数据处理与特征工程；
- Optuna 特征维度搜索；
- `top5/top10/top_best/top14/top85` 推荐对接维度；
- 模型同学的多模型接入实验；
- 中文图表与评估结果输出。

## Colab 推荐用法

本仓库不放 notebook。请使用本地文件：

```text
E:\aq_project\AQI_Colab_Run.ipynb
```

把它手动上传到 Google Colab 后，从上到下运行即可。notebook 已包含：

- GitHub 拉取项目；
- apt 安装中文字体；
- pip 安装依赖；
- 生成特征工程图、Optuna 图、模型评估图；
- 展示评估表和任务完成度表；
- 打包下载 `outputs/`。

如果不用 notebook，也可以在 Colab 单元格里手动执行：

```python
!apt-get -qq update
!apt-get -qq install -y fonts-noto-cjk
!rm -rf ~/.cache/matplotlib

!git clone https://github.com/GaoYIZ/machine_learning.git
%cd machine_learning
!pip -q install -r requirements-colab.txt
!python scripts/run_all.py --device cuda
```

如果 Colab 没有分配 GPU，脚本会自动回退到 CPU。

## 主要输出

运行完成后会生成：

```text
outputs/processed/          处理后 CSV、特征说明、泄漏审计、LSTM 序列
outputs/feature_figures/    数据处理与特征工程中文图
outputs/optuna/             Optuna 维度搜索结果和图
outputs/model_integration/  模型接入结果、预测图、残差图
```

关键结果文件：

```text
outputs/optuna/recommended_dimension_results.csv
outputs/model_integration/results/best_by_dimension.csv
outputs/model_integration/results/all_model_comparison_my_features.csv
```

## 输入形状

传统模型使用：

```text
X = [样本数, top_k]
```

LSTM 序列数据使用：

```text
X_seq = [样本数, 14, top_k]
```

其中 `top_k` 是 Optuna 和特征排序后推荐的特征维度。

## 单独运行命令

只生成特征数据包和 Optuna 结果：

```bash
python scripts/run_feature_pipeline.py
```

只运行模型接入实验，例如本次 Optuna 最优维度为 `top11` 时：

```bash
python scripts/run_model_integration.py --device cuda --top-k 5 10 11 14 85
```

一键运行全部流程：

```bash
python scripts/run_all.py --device cuda
```

`run_all.py` 默认会先读取 Optuna 的 `best_top_k`，再自动跑 `top5/top10/top_best/top14/top85`。
