"""
数据加载模块
负责读取原始Excel数据，进行初步的格式化和基本信息输出
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(data_path: str = None) -> pd.DataFrame:
    """
    加载原始空气质量数据

    Parameters:
    -----------
    data_path : str, optional
        数据文件路径，默认使用项目data目录下的文件

    Returns:
    --------
    pd.DataFrame: 包含日期索引的原始数据
    """
    if data_path is None:
        project_root = Path(__file__).parent.parent
        data_path = project_root / 'data' / 'raw_air_quality.xlsx'

    print(f"[DataLoader] 加载数据: {data_path}")
    df = pd.read_excel(data_path)

    # 重命名列（处理编码问题）
    df.columns = ['date', 'AQI', 'quality_level', 'PM2_5', 'PM10', 'SO2', 'CO', 'NO2', 'O3_8h']

    # 日期转换
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    print(f"[DataLoader] 数据形状: {df.shape}")
    print(f"[DataLoader] 时间范围: {df.index.min()} ~ {df.index.max()}")
    print(f"[DataLoader] 列名: {list(df.columns)}")

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    生成数据摘要统计

    Parameters:
    -----------
    df : pd.DataFrame
        加载后的数据

    Returns:
    --------
    dict: 包含数据统计信息的字典
    """
    numeric_cols = ['AQI', 'PM2_5', 'PM10', 'SO2', 'CO', 'NO2', 'O3_8h']

    summary = {
        'n_samples': len(df),
        'n_features': len(numeric_cols),
        'date_start': df.index.min(),
        'date_end': df.index.max(),
        'date_range_days': (df.index.max() - df.index.min()).days,
        'null_counts': df.isnull().sum().to_dict(),
        'null_ratio': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'describe': df[numeric_cols].describe().to_dict(),
        'quality_distribution': df['quality_level'].value_counts().to_dict(),
    }

    return summary


def print_data_summary(df: pd.DataFrame):
    """打印数据摘要信息"""
    print("\n" + "="*60)
    print("数据集摘要")
    print("="*60)
    print(f"样本数: {len(df)}")
    print(f"特征数: {len(df.columns)}")
    print(f"时间范围: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")
    print(f"总天数: {(df.index.max() - df.index.min()).days} 天")

    print("\n--- 缺失值统计 ---")
    null_counts = df.isnull().sum()
    for col in df.columns:
        if null_counts[col] > 0:
            print(f"  {col}: {null_counts[col]} ({null_counts[col]/len(df)*100:.2f}%)")
        else:
            print(f"  {col}: 0")

    print("\n--- 描述性统计 ---")
    numeric_cols = ['AQI', 'PM2_5', 'PM10', 'SO2', 'CO', 'NO2', 'O3_8h']
    print(df[numeric_cols].describe().round(2).to_string())

    print("\n--- 空气质量等级分布 ---")
    print(df['quality_level'].value_counts().to_string())


if __name__ == '__main__':
    df = load_raw_data()
    print_data_summary(df)
