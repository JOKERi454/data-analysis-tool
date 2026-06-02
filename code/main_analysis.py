"""
传感器数据分析 - 主运行脚本
==============================
一键完成: 数据加载 → 清洗 → 统计分析 → 可视化报告
适用课程实验, 无需逐行手写代码

使用方法:
    python main_analysis.py                  # 完整分析
    python main_analysis.py --quick          # 快速模式 (采样, 更快)
    python main_analysis.py --sensor 10      # 仅分析指定传感器
    python main_analysis.py --export-csv     # 导出清洗后的CSV
"""

import pandas as pd
import numpy as np
import sys
import os
import argparse
from datetime import datetime

# 导入本地模块
from data_cleaner import load_and_clean, quick_stats, resample_sensor, VALID_RANGES, COLUMN_NAMES
from visualizer import generate_full_report, plot_single_sensor, plot_overview_dashboard


def export_clean_csv(df: pd.DataFrame, filename: str = "cleaned_sensor_data.csv"):
    """导出清洗后的数据为CSV"""
    path = os.path.join("output_figures", filename)
    print(f"[导出] 正在保存CSV到 {path} ...")
    # 只导出前100万行到单个CSV (完整数据太大, 建议按需筛选)
    df_export = df.copy().reset_index(drop=True)
    if len(df_export) > 1_000_000:
        print(f"  数据量较大({len(df_export):,}行), 导出前100万行...")
        df_export = df_export.head(1_000_000)
    df_export.to_csv(path, index=False, encoding="utf-8-sig")
    file_size = os.path.getsize(path) / 1024**2
    print(f"  ✓ 已导出: {path} ({file_size:.1f} MB)")
    return path


def detailed_statistics(df: pd.DataFrame):
    """输出详细统计分析"""
    print("\n" + "="*70)
    print("                    详细统计分析")
    print("="*70)

    for var in ["temperature", "humidity", "light", "voltage"]:
        data = df[var].dropna()
        lo, hi = VALID_RANGES[var]
        print(f"\n{'─'*50}")
        print(f"  {var.upper()}  |  范围: [{lo}, {hi}]")
        print(f"{'─'*50}")
        print(f"  样本数:   {len(data):,}")
        print(f"  均值:     {data.mean():.4f}")
        print(f"  中位数:   {data.median():.4f}")
        print(f"  标准差:   {data.std():.4f}")
        print(f"  最小值:   {data.min():.4f}")
        print(f"  最大值:   {data.max():.4f}")
        print(f"  25%分位:  {data.quantile(0.25):.4f}")
        print(f"  75%分位:  {data.quantile(0.75):.4f}")
        print(f"  偏度:     {data.skew():.4f}")
        print(f"  峰度:     {data.kurtosis():.4f}")
        # IQR异常值检测
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        iqr_outliers = ((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)).sum()
        print(f"  IQR异常值: {iqr_outliers:,} ({iqr_outliers/len(data)*100:.2f}%)")

    # 每个传感器摘要
    print(f"\n{'─'*50}")
    print(f"  传感器数据量 Top 10:")
    print(f"{'─'*50}")
    top10 = df["moteid"].value_counts().head(10)
    for sid, cnt in top10.items():
        print(f"    Sensor {int(sid):2d}: {cnt:,} 条记录")

    print("\n" + "="*70)


def run_quick_mode():
    """快速模式: 采样数据, 快速出图验证代码"""
    print("[快速模式] 使用20%%采样数据以加速...\n")
    df_full, df_pos = load_and_clean()

    # 随机采样20% (分层按传感器保持比例)
    df = df_full.groupby("moteid", group_keys=False).apply(
        lambda g: g.sample(frac=0.2, random_state=42)
    ).reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    print(f"  采样后: {len(df):,} 条 (原始 {len(df_full):,} 条)")

    # 简化统计
    detailed_statistics(df)

    # 简化可视化
    print("\n[可视化] 生成核心图表...")
    plot_overview_dashboard(df)
    from visualizer import plot_sensor_boxplots, plot_correlation_matrix, plot_distributions, plot_daily_pattern

    plot_sensor_boxplots(df)
    plot_distributions(df)
    plot_correlation_matrix(df)
    plot_daily_pattern(df)

    if df_pos is not None:
        from visualizer import plot_spatial_heatmap
        plot_spatial_heatmap(df, df_pos)

    print("\n✓ 快速模式完成!")


def main():
    parser = argparse.ArgumentParser(
        description="传感器数据分析工具箱",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main_analysis.py                    # 完整分析
  python main_analysis.py --quick            # 快速模式
  python main_analysis.py --sensor 10        # 仅分析传感器#10
  python main_analysis.py --export-csv       # 导出清洗数据
        """
    )
    parser.add_argument("--quick", action="store_true", help="快速模式 (20%%采样)")
    parser.add_argument("--sensor", type=int, default=None, help="仅分析指定传感器ID")
    parser.add_argument("--export-csv", action="store_true", help="导出清洗数据为CSV")
    parser.add_argument("--no-viz", action="store_true", help="跳过可视化, 仅统计分析")
    args = parser.parse_args()

    start_time = datetime.now()
    print("╔" + "═"*58 + "╗")
    print("║" + "  传感器数据分析工具箱  —  Sensor Data Analysis Toolbox".center(48) + "║")
    print("║" + f"  开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}".center(53) + "║")
    print("╚" + "═"*58 + "╝")

    # ---- 快速模式 ----
    if args.quick:
        run_quick_mode()
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n总耗时: {elapsed:.1f} 秒")
        return

    # ---- 完整流程 ----
    # Step 1: 加载+清洗
    print("\n[Step 1/4] 加载并清洗数据...")
    df, df_pos = load_and_clean()

    # Step 2: 统计分析
    print("[Step 2/4] 执行统计分析...")
    detailed_statistics(df)

    # 可选: 单传感器分析
    if args.sensor is not None:
        print(f"\n[Step 3/4] 聚焦传感器 #{args.sensor}...")
        sensor_df = df[df["moteid"] == args.sensor]
        if sensor_df.empty:
            print(f"  [错误] 传感器 {args.sensor} 不存在!")
        else:
            print(f"  数据量: {len(sensor_df):,} 条")
            print(f"  温度: mean={sensor_df['temperature'].mean():.2f}°C, std={sensor_df['temperature'].std():.2f}")
            print(f"  湿度: mean={sensor_df['humidity'].mean():.2f}%,  std={sensor_df['humidity'].std():.2f}")
            print(f"  光照: mean={sensor_df['light'].mean():.2f}lux, std={sensor_df['light'].std():.2f}")
            print(f"  电压: mean={sensor_df['voltage'].mean():.3f}V,  std={sensor_df['voltage'].std():.3f}")
            plot_single_sensor(df, args.sensor)

    # Step 4: 可视化报告
    if not args.no_viz:
        print("[Step 4/4] 生成可视化报告...")
        generate_full_report(df, df_pos)
    else:
        print("[Step 4/4] 跳过可视化 (--no-viz)")

    # 导出CSV
    if args.export_csv:
        export_clean_csv(df)

    # 完成
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n╔" + "═"*58 + "╗")
    print(f"║  分析完成! 总耗时: {elapsed:.1f} 秒".ljust(55) + "║")
    print(f"║  图表输出: output_figures/".ljust(55) + "║")
    print("╚" + "═"*58 + "╝")


if __name__ == "__main__":
    main()
