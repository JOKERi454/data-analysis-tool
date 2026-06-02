"""
传感器数据清洗模块 - Data Cleaner
====================================
功能: 加载原始CSV/TXT数据, 解析, 清洗异常值, 输出干净的Pandas DataFrame
适用: Intel Berkeley Lab 传感器数据集 (温度/湿度/光照/电压)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


# ============================================
# 1. 配置区 - 可按需修改
# ============================================
DATA_PATH = r"dfae3-main\传感器数据集\传感器数据集\data完整.txt"
POS_PATH  = r"dfae3-main\传感器数据集\传感器数据集\位置信息.txt"

# 列名 (原始数据无表头)
COLUMN_NAMES = ["date", "time", "epoch", "moteid", "temperature", "humidity", "light", "voltage"]

# 各变量合理范围 (用于标记/过滤异常值)
VALID_RANGES = {
    "temperature": (-10, 50),    # 摄氏度, 实验室环境
    "humidity":    (0, 100),     # 相对湿度百分比
    "light":       (0, 2000),    # 光照 lux
    "voltage":     (2.0, 3.5),   # 电池电压 V
}


# ============================================
# 2. 数据加载
# ============================================
def load_raw_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    加载原始传感器数据文件
    参数:
        data_path: 数据文件路径
    返回:
        pd.DataFrame with columns: [datetime, epoch, moteid, temperature, humidity, light, voltage]
    """
    print(f"[加载] 读取文件: {data_path}")

    # 读取空白分隔的文本文件 (无表头)
    df = pd.read_csv(
        data_path,
        sep=r"\s+",               # 任意空白字符分隔
        header=None,              # 无表头
        names=COLUMN_NAMES,       # 指定列名
        dtype={
            "date": str, "time": str, "epoch": int, "moteid": int,
            "temperature": float, "humidity": float, "light": float, "voltage": float
        },
        engine="c",               # C引擎更快
        na_values=["NaN", "nan", "", "NA"],  # 识别缺失值标记
        low_memory=False
    )

    print(f"  原始行数: {len(df):,}")
    print(f"  内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    return df


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """将date+time列合并解析为datetime, 并设为索引"""
    print("[解析] 合并日期时间...")
    datetime_str = df["date"].str.cat(df["time"], sep=" ")
    # 使用 format='mixed' 处理时间戳格式不一致的问题
    # (部分行有微秒 .757551, 部分行没有如 "07:46:05")
    df["datetime"] = pd.to_datetime(datetime_str, format="mixed")
    df.drop(columns=["date", "time"], inplace=True)

    # 按时间排序
    df.sort_values("datetime", inplace=True)
    df.set_index("datetime", inplace=True)  # drop=True (默认), 避免列与索引同名冲突
    print(f"  时间范围: {df.index.min()} → {df.index.max()}")
    return df


def load_position_data(pos_path: str = POS_PATH) -> pd.DataFrame:
    """加载传感器位置信息 (X, Y坐标)"""
    print(f"[加载] 传感器位置信息: {pos_path}")
    df_pos = pd.read_csv(
        pos_path, sep=r"\s+", header=None, names=["moteid", "x", "y"],
        skipfooter=1, engine="python"  # 最后一行是表头说明, 跳过
    )
    print(f"  {len(df_pos)} 个传感器有位置信息")
    return df_pos


# ============================================
# 3. 数据质量分析
# ============================================
def diagnose_data(df: pd.DataFrame) -> dict:
    """
    数据质量诊断: 缺失值、异常值、重复值统计
    返回诊断报告字典
    """
    print("\n" + "="*60)
    print("  数据质量诊断报告")
    print("="*60)

    n = len(df)
    report = {"total_rows": n}

    # 缺失值
    missing = df.isnull().sum()
    missing_pct = (missing / n * 100).round(3)
    print(f"\n[缺失值统计]")
    missing_found = False
    for col in ["temperature", "humidity", "light", "voltage"]:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]:,} ({missing_pct[col]:.2f}%)")
            missing_found = True
    if not missing_found:
        print("  ✓ 无缺失值")
    report["missing"] = missing.to_dict()

    # 异常值 (基于合理范围)
    print(f"\n[异常值统计] (超出合理范围)")
    outlier_counts = {}
    for col, (lo, hi) in VALID_RANGES.items():
        outliers = ((df[col] < lo) | (df[col] > hi)).sum()
        pct = outliers / n * 100
        if outliers > 0:
            print(f"  {col}: {outliers:,} 条 ({pct:.2f}%) - 范围应为 [{lo}, {hi}]")
        else:
            print(f"  {col}: ✓ 无异常")
        outlier_counts[col] = outliers
    report["outlier_counts"] = outlier_counts

    # 重复值
    dupes = df.duplicated(subset=["epoch", "moteid"]).sum()
    print(f"\n[重复值]")
    if dupes > 0:
        print(f"  基于 (epoch, moteid) 的重复: {dupes:,} 条 ({dupes/n*100:.2f}%)")
    else:
        print("  ✓ 无重复")
    report["duplicates"] = dupes

    # 传感器统计
    sensors = df["moteid"].unique()
    print(f"\n[传感器] 共 {len(sensors)} 个节点: {sorted(sensors)[:10]}...")
    report["sensor_count"] = len(sensors)
    report["sensor_ids"] = sorted(sensors.tolist())

    # 每个传感器的数据量分布
    sensor_counts = df["moteid"].value_counts().describe()
    print(f"\n[数据量分布] (每传感器记录数)")
    print(f"  min={sensor_counts['min']:,.0f}  max={sensor_counts['max']:,.0f}")
    print(f"  mean={sensor_counts['mean']:,.0f}  std={sensor_counts['std']:,.0f}")

    print("="*60 + "\n")
    return report


# ============================================
# 4. 数据清洗
# ============================================
def clean_data(df: pd.DataFrame,
               remove_outliers: bool = True,
               interpolate_missing: bool = True) -> pd.DataFrame:
    """
    清洗传感器数据
    参数:
        df: 原始DataFrame
        remove_outliers: 是否移除非物理范围异常值
        interpolate_missing: 是否线性插值填补缺失
    返回:
        清洗后的DataFrame
    """
    print("[清洗] 开始数据清洗...")
    n_before = len(df)
    clean_ops = []

    # --- 步骤1: 去除重复值 ---
    dupes = df.duplicated(subset=["epoch", "moteid"])
    if dupes.sum() > 0:
        df = df[~dupes].copy()  # .copy() 避免后续 SettingWithCopyWarning
        clean_ops.append(f"删除重复 {(dupes.sum()):,} 条")

    # --- 步骤2: 标记/剔除异常值 ---
    if remove_outliers:
        total_outliers = 0
        for col, (lo, hi) in VALID_RANGES.items():
            outlier_mask = (df[col] < lo) | (df[col] > hi)
            if outlier_mask.sum() > 0:
                df.loc[outlier_mask, col] = np.nan
                total_outliers += outlier_mask.sum()
        if total_outliers > 0:
            clean_ops.append(f"标记异常值为NaN {total_outliers:,} 条")

    # --- 步骤3: 插值缺失 ---
    if interpolate_missing:
        na_before = df[["temperature", "humidity", "light", "voltage"]].isnull().sum().sum()
        if na_before > 0:
            # 按传感器分组做时间序列线性插值
            for col in ["temperature", "humidity", "light", "voltage"]:
                df[col] = df.groupby("moteid")[col].transform(
                    lambda s: s.interpolate(method="linear", limit_direction="both")
                )
            na_after = df[["temperature", "humidity", "light", "voltage"]].isnull().sum().sum()
            clean_ops.append(f"线性插值填补 {na_before:,} → 剩余 {na_after:,} 个缺失")

    # --- 步骤4: 类型优化以节省内存 ---
    for col in ["temperature", "humidity", "light", "voltage"]:
        df[col] = pd.to_numeric(df[col], downcast="float")
    df["moteid"] = df["moteid"].astype("int16")
    df["epoch"] = df["epoch"].astype("int32")

    n_after = len(df)
    print(f"  清洗前: {n_before:,} 条 → 清洗后: {n_after:,} 条")
    for op in clean_ops:
        print(f"  ✓ {op}")

    mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"  内存占用: {mem:.1f} MB")

    return df


# ============================================
# 5. 主加载函数 (一键执行全流程)
# ============================================
def load_and_clean(data_path: str = DATA_PATH,
                   pos_path: str = POS_PATH,
                   return_positions: bool = True) -> tuple:
    """
    一键加载、解析、诊断、清洗数据
    参数:
        data_path: 数据文件路径
        pos_path: 位置信息文件路径
        return_positions: 是否返回位置信息
    返回:
        (df_clean, df_pos) 或 df_clean
    """
    # 1. 加载
    df = load_raw_data(data_path)

    # 2. 解析日期
    df = parse_datetime(df)

    # 3. 诊断
    diagnosis = diagnose_data(df)

    # 4. 清洗
    df_clean = clean_data(df)

    if return_positions:
        try:
            df_pos = load_position_data(pos_path)
            return df_clean, df_pos
        except FileNotFoundError:
            print("[警告] 位置文件不存在, 仅返回传感器数据")
            return df_clean

    return df_clean


# ============================================
# 6. 便捷统计函数
# ============================================
def quick_stats(df: pd.DataFrame) -> pd.DataFrame:
    """按传感器输出关键统计量"""
    stats = df.groupby("moteid")[["temperature", "humidity", "light", "voltage"]].agg([
        "count", "mean", "std", "min", "max",
        ("Q1", lambda x: x.quantile(0.25)),
        ("median", lambda x: x.quantile(0.50)),
        ("Q3", lambda x: x.quantile(0.75)),
    ]).round(3)
    return stats


def resample_sensor(df: pd.DataFrame, moteid: int, freq: str = "10T") -> pd.DataFrame:
    """
    提取单个传感器并按频率重采样
    参数:
        df: 清洗后的DataFrame
        moteid: 传感器ID
        freq: 重采样频率 (如 '1T'=1分钟, '10T'=10分钟, '1H'=1小时)
    """
    sensor_df = df[df["moteid"] == moteid].copy()
    resampled = sensor_df.resample(freq).mean(numeric_only=True)
    resampled["moteid"] = moteid
    return resampled


if __name__ == "__main__":
    # 独立运行时: 执行完整清洗流程
    print("=" * 60)
    print("  传感器数据清洗工具")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    df, df_pos = load_and_clean()

    # 快速预览
    print("\n[预览] 清洗后数据前5行:")
    print(df.head())
    print(f"\n[统计] 每个传感器关键统计量 (前5个):")
    stats = quick_stats(df)
    print(stats.head(10))

    print("\n✓ 清洗完成! df (DataFrame) 和 df_pos 可用于后续分析和可视化。")
