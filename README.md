# AQI Prediction: Feature Engineering + Model Integration

This repository is a Colab-friendly version of the AQI prediction group project.

It includes:

- raw daily AQI dataset from 2013 to 2019;
- leakage-safe feature engineering;
- Optuna top-k feature dimension search;
- teammate model benchmark integration;
- scripts that can run end-to-end on Google Colab.

## Quick Start In Colab

Use the local notebook provided separately by the project owner, or run these commands in a Colab cell:

```python
!git clone https://github.com/GaoYIZ/machine_learning.git
%cd machine_learning
!pip -q install -r requirements-colab.txt
!python scripts/run_all.py --device cuda --top-k 5 10 14 85
```

If Colab GPU is unavailable, the script falls back to CPU for BiLSTM.

## Outputs

After running:

```text
outputs/processed/
outputs/feature_figures/
outputs/optuna/
outputs/model_integration/
```

Important files:

```text
outputs/optuna/recommended_dimension_results.csv
outputs/model_integration/results/best_by_dimension.csv
outputs/model_integration/results/all_model_comparison_my_features.csv
```

## Recommended Dimensions

The pipeline evaluates these feature dimensions:

```text
top5   validation-best low-dimensional feature subset
top10  one-day pollution state subset
top14  balanced feature-family subset
top85  full engineered feature baseline
```

Traditional models use:

```text
X = [samples, top_k]
```

Sequence models use:

```text
X_seq = [samples, 14, top_k]
```

## Main Commands

Build feature package only:

```bash
python scripts/run_feature_pipeline.py
```

Run teammate model benchmark with generated feature package:

```bash
python scripts/run_model_integration.py --device cuda --top-k 5 10 14 85
```

Run everything:

```bash
python scripts/run_all.py --device cuda --top-k 5 10 14 85
```
