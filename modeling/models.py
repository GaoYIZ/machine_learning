"""
=============================================================================
模型模块 (Models)
=============================================================================
包含全部模型的实现:

  [统计基准] M1: 持久性模型 (Persistence)
  [线性方法] M2: 岭回归 (Ridge)
  [核方法]   M3: 支持向量回归 (SVR)
  [树集成]   M4: 随机森林 (RF)
  [树集成]   M5: XGBoost
  [树集成]   M6: LightGBM
  [树集成]   M7: CatBoost
  [神经网络] M8: 多层感知机 (MLP)
  [深度学习] M9: BiLSTM + Attention
  [深度学习] M9B: Peak-weighted BiLSTM + Attention
  [集成融合] M10: Stacking 集成 (改进模型)
=============================================================================
"""

import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from typing import Dict, Tuple, Optional


def set_torch_seed(seed: int = 42) -> None:
    """Keep neural-network runs reproducible across Colab executions."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# M1: 持久性模型 (Persistence / Naive Forecast)
# =============================================================================
class PersistenceModel:
    """
    持久性模型: y_hat(t+1) = y(t)

    最简基准 — 明天的 AQI 等于今天的 AQI。
    其 RMSE 等于 AQI 日间变化量的标准差, 量化了
    "仅靠 AQI 自相关" 能解释的上限。

    任何有意义的模型必须超越这个基准。
    """
    def __init__(self):
        self.name = 'M1_Persistence'

    def fit(self, X=None, y=None):
        """无需训练"""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        # 持久性模型: 用今天的 AQI 预测明天的 AQI → y_hat(t+1) = AQI(t)
        if hasattr(X, 'columns'):
            if 'AQI' in X.columns:
                return X['AQI'].values
            for col in ['AQI_lag_1', 'AQI_lag_1.0']:
                if col in X.columns:
                    return X[col].values
        if len(X.shape) == 2:
            return X[:, 0]
        return X.flatten()


# =============================================================================
# M2: 岭回归 (Ridge Regression)
# =============================================================================
def create_ridge(alpha: float = 1.0) -> Ridge:
    """
    岭回归: 在线性回归的 MSE 损失函数中加入 L2 正则化项。

    损失: ||y - Xw||^2 + alpha * ||w||^2

    为什么选它?
    - 代表线性方法的天花板 (比普通线性回归更能处理共线性)
    - 44 个工程特征中可能存在多重共线性 (如 lag_1 和 roll_mean_3)
    - L2 正则化通过 "收缩" 系数来处理共线性, 而非直接删除特征
    """
    return Ridge(alpha=alpha, random_state=42)


# =============================================================================
# M3: 支持向量回归 (SVR)
# =============================================================================
def create_svr(C: float = 10.0, epsilon: float = 0.1) -> SVR:
    """
    SVR (RBF 核): 在特征空间中寻找宽度为 2ε 的 "管道",
    尽量使训练点落在管道内。

    核函数: K(x,x') = exp(-γ||x-x'||²)
    - 隐式地将数据映射到高维空间以处理非线性
    - 代表 "核方法" 这一方法论流派

    局限性:
    - 计算复杂度 O(n²)~O(n³), 大数据集受限
    - 隐式特征映射缺乏可解释性 (与树模型形成对比)
    """
    return SVR(kernel='rbf', C=C, epsilon=epsilon,
               gamma='scale', cache_size=500)


# =============================================================================
# M4: 随机森林 (Random Forest)
# =============================================================================
def create_rf(n_estimators: int = 200, max_depth: int = 15) -> RandomForestRegressor:
    """
    随机森林: 并行构建 N 棵决策树, 每棵树在 Bootstrap 样本 +
    随机特征子集上独立训练, 最终取均值。

    方差降低原理:
    单棵树的预测误差 = 偏差² + 方差
    N 棵树的平均误差  = 偏差² + 方差/N  (在误差独立假设下)
    → 200 棵树的方差 = 单棵树的 0.5%
    """
    return RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=5, min_samples_leaf=2,
        max_features='sqrt', random_state=42, n_jobs=-1
    )


# =============================================================================
# M5: XGBoost
# =============================================================================
def create_xgb(n_estimators: int = 300, lr: float = 0.05,
               max_depth: int = 6) -> XGBRegressor:
    """
    XGBoost (Extreme Gradient Boosting):
    序列式构建树 — 第 k 棵树专门拟合前 k-1 棵树的累积残差。

    核心创新:
    - 使用损失函数的二阶泰勒展开做分裂决策 (比一阶梯度更精确)
    - 内置 L1+L2 正则化 (reg_alpha + reg_lambda)
    - 支持列采样 (colsample_bytree) 和学习率衰减

    数学形式:
    y_hat_i^(k) = y_hat_i^(k-1) + η · f_k(x_i)
    """
    return XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0
    )


# =============================================================================
# M6: LightGBM
# =============================================================================
def create_lgb(n_estimators: int = 300, lr: float = 0.05) -> LGBMRegressor:
    """
    LightGBM: 与 XGBoost 同属 Gradient Boosting, 但使用:
    (1) Leaf-wise 生长 — 优先分裂增益最大的叶子节点
    (2) GOSS 采样 — 保留高梯度样本, 对低梯度样本降采样

    与 XGBoost 的对比目的:
    - XGBoost Level-wise 生长 → 天然隐式正则化
    - LightGBM Leaf-wise → 更高效但可能在小数据上过拟合
    - 两者对比 → "哪种生长策略更适合本任务?"
    """
    return LGBMRegressor(
        n_estimators=n_estimators, learning_rate=lr,
        num_leaves=31, max_depth=8,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1
    )


# =============================================================================
# M7: CatBoost
# =============================================================================
def create_cat(n_iterations: int = 500, lr: float = 0.05) -> CatBoostRegressor:
    """
    CatBoost (Categorical Boosting):
    Yandex 开发的 GBDT 变体, 两大核心创新:

    (1) Ordered Boosting:
        使用排序后的数据子集计算梯度, 天然抗过拟合,
        在小数据集 (1000-5000 样本) 上经常不调参就超越 XGBoost

    (2) 对称树 (Symmetric Trees):
        每层所有节点使用相同的分裂条件 → 推理速度极快

    为什么加入 CatBoost?
    - 本任务有 1,448 个训练样本, 恰好是 CatBoost 的优势区间
    - GBDT 三巨头 (XGBoost/LightGBM/CatBoost) 的完整对比
    """
    return CatBoostRegressor(
        iterations=n_iterations, learning_rate=lr,
        depth=6, l2_leaf_reg=3.0,
        random_seed=42, verbose=False, allow_writing_files=False
    )


# =============================================================================
# M8: 多层感知机 (MLP)
# =============================================================================
def create_mlp() -> MLPRegressor:
    """
    简单前馈神经网络: 3 层全连接 (256→128→64→1), ReLU 激活。

    为什么加入 MLP?
    - 与树模型使用同样的 44 维工程特征 → 公平对比
    - 回答: "非线性函数逼近 (NN) vs 分段常数逼近 (树), 谁更适合这个特征空间?"
    - 之前的 BiLSTM 用的是 7 维原始序列, 与树模型的对比不等价
    """
    return MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu', solver='adam',
        alpha=0.001, batch_size=64,
        learning_rate_init=0.001, max_iter=500,
        early_stopping=True, validation_fraction=0.1,
        random_state=42, verbose=False
    )


# =============================================================================
# M9: BiLSTM + Attention (深度学习)
# =============================================================================
class BiLSTMAttention(nn.Module):
    """
    双向 LSTM + 加性注意力 (Additive Attention)

    架构: Input(seq_len × top_k_features) → BiLSTM(64, 2层) → Attention → Dense(64→32→1)

    设计要点:
    - BiLSTM: 双向编码, 同时捕获过去和未来的上下文 (在已知序列内)
    - Attention: 自适应加权 → 模型自动关注最重要的历史时间步
    - 输入为 top_k 工程特征序列，与传统模型使用同一套无泄漏特征，便于公平比较
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if num_layers > 1 else 0)
        self.attn_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_v = nn.Parameter(torch.rand(hidden_dim))
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """x: [batch, seq_len, top_k_features]"""
        lstm_out, _ = self.bilstm(x)  # [B, 14, 128]

        # Attention: e_t = v^T · tanh(W · h_t),  α = softmax(e)
        energy = torch.tanh(self.attn_fc(lstm_out))
        energy = energy.permute(0, 2, 1)
        v = self.attn_v.repeat(x.size(0), 1).unsqueeze(1)
        attn_w = torch.softmax(torch.bmm(v, energy).squeeze(1), dim=1)
        context = torch.bmm(attn_w.unsqueeze(1), lstm_out).squeeze(1)

        return self.fc(context).squeeze(-1), attn_w


def prepare_sequence_X(X: np.ndarray, seq_len: int = 14) -> np.ndarray:
    """
    Build LSTM input sequences from a 2-D feature matrix.

    The sequence ending at row t predicts y[t]. This matches the feature package:
    every row's model features are already leak-safe through shift/lag logic.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"prepare_sequence_X expects a 2-D array, got shape {X.shape}")
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    if len(X) < seq_len:
        raise ValueError(f"Need at least seq_len={seq_len} rows, got {len(X)}")

    sequences = []
    for end_idx in range(seq_len - 1, len(X)):
        start_idx = end_idx - seq_len + 1
        sequences.append(X[start_idx : end_idx + 1])
    return np.asarray(sequences, dtype=np.float32)


def prepare_sequence_xy(X: np.ndarray, y: np.ndarray, seq_len: int = 14):
    """Build `[samples, seq_len, features]` inputs and aligned targets."""
    X_seq = prepare_sequence_X(X, seq_len=seq_len)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(y) != len(X):
        raise ValueError(f"X/y length mismatch: len(X)={len(X)}, len(y)={len(y)}")
    return X_seq, y[seq_len - 1 :].astype(np.float32)


def prepare_sequences(data: np.ndarray, seq_len: int = 14):
    """Backward-compatible wrapper for older callers."""
    data = np.asarray(data, dtype=np.float32)
    return prepare_sequence_X(data, seq_len=seq_len), data[seq_len - 1 :, 0].astype(np.float32)


def weighted_huber_loss(pred, target, sample_weight=None, delta: float = 1.0):
    """Huber loss with optional per-sample weights."""
    error = pred - target
    abs_error = torch.abs(error)
    quadratic = torch.minimum(abs_error, torch.tensor(delta, device=pred.device, dtype=pred.dtype))
    linear = abs_error - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    if sample_weight is not None:
        loss = loss * sample_weight
    return loss.mean()


def train_bilstm(X_train, y_train, X_val, y_val,
                 seq_len=14, hidden_dim=64, epochs=200,
                 lr=0.001, batch_size=32, patience=20,
                 device='cpu', sample_weight_train=None,
                 seed: int = 42, model_label: str = 'M9 BiLSTM',
                 dropout: float = 0.2) -> Tuple[BiLSTMAttention, dict]:
    """BiLSTM + Attention 模型训练器"""
    set_torch_seed(seed)
    print(f"\n  [{model_label}] training (device={device}, seed={seed})...")

    # 准备序列
    if len(X_train.shape) == 2:
        X_train, y_train = prepare_sequence_xy(X_train, y_train, seq_len)
        X_val, y_val = prepare_sequence_xy(X_val, y_val, seq_len)

    if sample_weight_train is not None:
        sample_weight_train = np.asarray(sample_weight_train, dtype=np.float32).reshape(-1)
        if len(sample_weight_train) != len(y_train):
            sample_weight_train = sample_weight_train[seq_len - 1 :]
        if len(sample_weight_train) != len(y_train):
            raise ValueError(
                f"sample_weight_train length mismatch: {len(sample_weight_train)} vs {len(y_train)}"
            )

    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.FloatTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.FloatTensor(y_val).to(device)
    wt = torch.FloatTensor(sample_weight_train).to(device) if sample_weight_train is not None else None

    print(f"  [{model_label}] sequence shapes: X_train={tuple(Xt.shape)}, X_val={tuple(Xv.shape)}")

    model = BiLSTMAttention(X_train.shape[2], hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8
    )

    t0 = time.time()
    best_loss = float('inf')
    best_state = None
    no_improve = 0
    h = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        train_losses = []
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            pred, _ = model(Xt[idx])
            batch_weight = wt[idx] if wt is not None else None
            loss = weighted_huber_loss(pred, yt[idx], sample_weight=batch_weight, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            vp, _ = model(Xv)
            vloss = weighted_huber_loss(vp, yv, delta=1.0).item()

        h['train_loss'].append(np.mean(train_losses))
        h['val_loss'].append(vloss)
        scheduler.step(vloss)

        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 40 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | "
                  f"Train={h['train_loss'][-1]:.4f} | Val={vloss:.4f}")
        if no_improve >= patience:
            print(f"    Early Stop @ Epoch {epoch+1}")
            break

    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  [{model_label}] done ({train_time:.1f}s) | Best Val Loss={best_loss:.4f}")
    return model, {'train_time': train_time, 'best_val_loss': best_loss, 'history': h}


def predict_bilstm(model, X, seq_len=14, device='cpu'):
    """BiLSTM 预测"""
    if len(X.shape) == 2:
        X = prepare_sequence_X(X, seq_len)
    model.eval()
    with torch.no_grad():
        pred, _ = model(torch.FloatTensor(X).to(device))
    return pred.cpu().numpy()


# =============================================================================
# M10: Stacking 集成 (改进模型)
# =============================================================================
def create_stacking_model() -> StackingRegressor:
    """
    Stacking 集成: 两层模型架构

    第一层 (基学习器): XGBoost + LightGBM + RF + CatBoost
      - 四个模型从不同角度学习 AQI 的预测规律
      - 它们的预测误差通常不相关 → 互补

    第二层 (元学习器): Ridge 回归
      - 学习如何组合四个基模型的输出
      - 使用 TimeSeriesSplit (5折) 进行交叉验证 → 严格避免数据泄漏

    为什么这很可能超越 XGBoost?
    - Kaggle 竞赛和文献一致证明: Stacking 几乎总是优于单一最优模型
    - 四个基模型覆盖 Bagging + 两种 Boosting → 误差模式互补
    - 元学习器自动学习最优组合权重
    """
    base_models = [
        ('xgb', create_xgb(n_estimators=300, lr=0.05, max_depth=6)),
        ('lgb', create_lgb(n_estimators=300, lr=0.05)),
        ('rf',  create_rf(n_estimators=200, max_depth=15)),
        ('cat', create_cat(n_iterations=500, lr=0.05)),
    ]
    return StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=1.0),
        cv=5,          # 5折交叉验证
        n_jobs=-1,
        passthrough=False
    )


class ValidationStackingEnsemble:
    """
    使用独立验证集训练元学习器的 Stacking/Blending 集成。

    流程:
    1. 基学习器只用训练集拟合。
    2. 基学习器在验证集上预测, 形成验证集元特征。
    3. Ridge 元学习器使用验证集元特征和验证集真实 y 拟合。
    4. 基学习器可再用 train+val 重拟合, 用于最终测试/未来预测。

    这样 2018 年验证集会被明确用于模型融合权重学习, 避免普通 KFold
    在时间序列场景中的解释风险。
    """

    def __init__(self, estimators=None, final_estimator=None,
                 refit_base_on_train_val: bool = True, random_state: int = 42):
        self.estimators = estimators
        self.final_estimator = final_estimator or Ridge(alpha=1.0)
        self.refit_base_on_train_val = refit_base_on_train_val
        self.random_state = random_state

    def _default_estimators(self):
        return [
            ('xgb', create_xgb(n_estimators=300, lr=0.05, max_depth=6)),
            ('lgb', create_lgb(n_estimators=300, lr=0.05)),
            ('rf',  create_rf(n_estimators=200, max_depth=15)),
            ('cat', create_cat(n_iterations=500, lr=0.05)),
        ]

    def fit(self, X_train, y_train, X_val, y_val):
        estimators = self.estimators or self._default_estimators()

        train_models = []
        val_preds = []
        for name, estimator in estimators:
            model = clone(estimator)
            model.fit(X_train, y_train)
            train_models.append((name, model))
            val_preds.append(model.predict(X_val))

        self.meta_features_val_ = np.column_stack(val_preds)
        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(self.meta_features_val_, y_val)

        if self.refit_base_on_train_val:
            X_full = np.vstack([X_train, X_val])
            y_full = np.concatenate([y_train, y_val])
            self.base_models_ = []
            for name, estimator in estimators:
                model = clone(estimator)
                model.fit(X_full, y_full)
                self.base_models_.append((name, model))
        else:
            self.base_models_ = train_models

        self.base_model_names_ = [name for name, _ in self.base_models_]
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            model.predict(X) for _, model in self.base_models_
        ])
        return self.final_estimator_.predict(meta_features)


# =============================================================================
# 统一训练接口
# =============================================================================
def train_all_models(X_train, y_train, X_val, y_val, X_test, y_test,
                     device='cpu', seq_len: int = 14) -> Dict:
    """
    训练全部模型并返回测试集上的预测结果。

    Returns
    -------
    Dict: {模型名: {'test_pred': np.array, 'test_metrics': dict, 'train_time': float, ...}}
    """
    results = {}
    target = 'AQI'

    # ---- M1: Persistence ----
    print("\n[M1] Persistence (Baseline)...")
    m1 = PersistenceModel()
    pred = m1.predict(X_test)
    results['M1_Persistence'] = {
        'test_pred': pred, 'train_time': 0, 'type': 'Baseline', 'model': m1
    }

    # ---- M2: Ridge ----
    print("[M2] Ridge Regression...")
    t0 = time.time()
    m2 = create_ridge(alpha=1.0)
    m2.fit(X_train, y_train)
    results['M2_Ridge'] = {
        'test_pred': m2.predict(X_test),
        'train_time': time.time() - t0, 'type': 'Linear', 'model': m2
    }

    # ---- M3: SVR ----
    print("[M3] SVR (RBF)...")
    t0 = time.time()
    m3 = create_svr()
    n_svr = min(3000, len(X_train))  # SVR 在 >3000 样本上很慢
    m3.fit(X_train[:n_svr], y_train[:n_svr])
    results['M3_SVR'] = {
        'test_pred': m3.predict(X_test),
        'train_time': time.time() - t0, 'type': 'Kernel', 'model': m3
    }

    # ---- M4: Random Forest ----
    print("[M4] Random Forest...")
    t0 = time.time()
    m4 = create_rf()
    m4.fit(X_train, y_train)
    results['M4_RandomForest'] = {
        'test_pred': m4.predict(X_test),
        'train_time': time.time() - t0, 'type': 'Tree-Bagging', 'model': m4
    }

    # ---- M5: XGBoost ----
    print("[M5] XGBoost...")
    t0 = time.time()
    m5 = create_xgb()
    m5.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    results['M5_XGBoost'] = {
        'test_pred': m5.predict(X_test),
        'train_time': time.time() - t0, 'type': 'Tree-Boosting', 'model': m5
    }

    # ---- M6: LightGBM ----
    print("[M6] LightGBM...")
    t0 = time.time()
    m6 = create_lgb()
    m6.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    results['M6_LightGBM'] = {
        'test_pred': m6.predict(X_test),
        'train_time': time.time() - t0, 'type': 'Tree-Boosting', 'model': m6
    }

    # ---- M7: CatBoost ----
    print("[M7] CatBoost...")
    t0 = time.time()
    m7 = create_cat()
    m7.fit(X_train, y_train, eval_set=(X_val, y_val),
           verbose=False, early_stopping_rounds=50)
    results['M7_CatBoost'] = {
        'test_pred': m7.predict(X_test),
        'train_time': time.time() - t0, 'type': 'Tree-Boosting', 'model': m7
    }

    # ---- M8: MLP ----
    print("[M8] MLP (Neural Network)...")
    t0 = time.time()
    m8 = create_mlp()
    m8.fit(X_train, y_train)
    results['M8_MLP'] = {
        'test_pred': m8.predict(X_test),
        'train_time': time.time() - t0, 'type': 'NeuralNetwork', 'model': m8
    }

    # ---- M9: BiLSTM + Attention ----
    print("[M9] BiLSTM + Attention...")
    y_test_seq = y_test[seq_len - 1 :]
    try:
        m9, m9_hist = train_bilstm(
            X_train, y_train, X_val, y_val,
            seq_len=seq_len, device=device, seed=42, model_label='M9 BiLSTM'
        )
        pred_m9 = predict_bilstm(m9, X_test, seq_len=seq_len, device=device)
        results['M9_BiLSTM'] = {
            'test_pred': pred_m9,
            'train_time': m9_hist['train_time'],
            'type': 'DeepLearning',
            '_y_test': y_test_seq,
            '_eval_offset': seq_len - 1,
            'model': m9
        }
    except Exception as e:
        print(f"  [WARNING] BiLSTM 训练失败: {e}")
        results['M9_BiLSTM'] = {
            'test_pred': np.full(len(y_test_seq), np.nan),
            'train_time': 0, 'type': 'DeepLearning',
            '_y_test': y_test_seq,
            '_eval_offset': seq_len - 1,
        }

    # ---- M9B: Peak-weighted BiLSTM + Attention ----
    print("[M9B] Peak-weighted BiLSTM + Attention...")
    try:
        q75 = float(np.quantile(y_train, 0.75))
        peak_weights = np.where(y_train >= q75, 2.0, 1.0).astype(np.float32)
        m9b, m9b_hist = train_bilstm(
            X_train, y_train, X_val, y_val,
            seq_len=seq_len,
            device=device,
            sample_weight_train=peak_weights,
            seed=42,
            model_label='M9B PeakWeighted BiLSTM',
        )
        pred_m9b = predict_bilstm(m9b, X_test, seq_len=seq_len, device=device)
        results['M9B_BiLSTM_PeakWeighted'] = {
            'test_pred': pred_m9b,
            'train_time': m9b_hist['train_time'],
            'type': 'DeepLearning',
            '_y_test': y_test_seq,
            '_eval_offset': seq_len - 1,
            'model': m9b
        }
    except Exception as e:
        print(f"  [WARNING] Peak-weighted BiLSTM 训练失败: {e}")
        results['M9B_BiLSTM_PeakWeighted'] = {
            'test_pred': np.full(len(y_test_seq), np.nan),
            'train_time': 0, 'type': 'DeepLearning',
            '_y_test': y_test_seq,
            '_eval_offset': seq_len - 1,
        }

    # ---- M10: Stacking Ensemble ----
    print("[M10] Validation Stacking Ensemble (改进模型)...")
    t0 = time.time()
    m10 = ValidationStackingEnsemble(random_state=42)
    m10.fit(X_train, y_train, X_val, y_val)
    results['M10_Stacking'] = {
        'test_pred': m10.predict(X_test),
        'train_time': time.time() - t0, 'type': 'Ensemble', 'model': m10
    }

    return results
