"""
传感器数据可视化模块 - Visualizer
==================================
功能: 基于清洗后的数据生成多维度可视化图表
包含: 时序图、分布图、空间热力图、相关性矩阵、传感器对比等
依赖: matplotlib, seaborn, numpy, pandas
"""

import matplotlib
matplotlib.use("Agg")  # 非交互后端, 适合脚本批量出图
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pandas as pd
import os
from datetime import datetime

# ============================================
# 全局样式设置
# ============================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 15,
    "legend.fontsize": 8,
    "font.family": "sans-serif",
})

# 统一配色
COLORS = {
    "temperature": "#E74C3C",  # 红
    "humidity":    "#3498DB",  # 蓝
    "light":       "#F39C12",  # 橙
    "voltage":     "#2ECC71",  # 绿
}
OUTPUT_DIR = "output_figures"


def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_or_show(fig, filename: str, show: bool = False):
    """保存图表, 可选是否显示"""
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"  → 已保存: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================
# 1. 数据概览仪表板
# ============================================
def plot_overview_dashboard(df: pd.DataFrame):
    """
    综合概览: 4个子图 - 温度/湿度/光照/电压的时间序列 (所有传感器叠加)
    """
    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

    variables = ["temperature", "humidity", "light", "voltage"]
    titles = ["温度 Temperature (°C)", "湿度 Humidity (%)",
              "光照 Light (lux)", "电压 Voltage (V)"]

    for ax, var, title, color in zip(axes, variables, titles, COLORS.values()):
        # 对每个传感器画半透明线 (采样以提高性能)
        for moteid, grp in df.groupby("moteid"):
            # 如果点太多, 采样到最多5000个点
            sample = grp[var] if len(grp) <= 5000 else grp[var].iloc[::max(1, len(grp)//5000)]
            ax.plot(sample.index, sample.values,
                    alpha=0.15, linewidth=0.3, color=color)

        # 画所有传感器均值线
        mean_series = df.groupby(level=0)[var].mean()  # 按时间索引聚合所有传感器
        if len(mean_series) > 5000:
            mean_series = mean_series.iloc[::len(mean_series)//5000]
        ax.plot(mean_series.index, mean_series.values,
                color="black", linewidth=1.5, alpha=0.8, label="均值")

        ax.set_ylabel(title)
        ax.legend(loc="upper right", framealpha=0.8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("时间")
    fig.suptitle("传感器数据总览 — 所有节点时序叠加", fontweight="bold", y=1.01)
    fig.autofmt_xdate()
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.tight_layout()

    save_or_show(fig, "01_overview_dashboard.png")
    return fig


# ============================================
# 2. 单传感器详细时序图
# ============================================
def plot_single_sensor(df: pd.DataFrame, moteid: int):
    """
    单个传感器4变量详细时序图
    参数:
        df: 数据
        moteid: 传感器ID (1-58)
    """
    sensor_df = df[df["moteid"] == moteid].copy()
    if sensor_df.empty:
        print(f"[错误] 传感器 {moteid} 无数据")
        return None

    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)
    variables = ["temperature", "humidity", "light", "voltage"]
    titles = [
        f"温度 Temperature — Sensor {moteid}",
        f"湿度 Humidity — Sensor {moteid}",
        f"光照 Light — Sensor {moteid}",
        f"电压 Voltage — Sensor {moteid}",
    ]

    for ax, var, title, color in zip(axes, variables, titles, COLORS.values()):
        ax.plot(sensor_df.index, sensor_df[var].values,
                linewidth=0.5, color=color, alpha=0.9)

        # 添加移动平均
        window = max(10, len(sensor_df) // 200)
        if len(sensor_df) > window:
            rolling = sensor_df[var].rolling(window=window, center=True).mean()
            ax.plot(rolling.index, rolling.values,
                    linewidth=1.5, color="black", alpha=0.6,
                    label=f"移动平均 (window={window})")

        ax.set_ylabel(titles[0].split(" ")[0])
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper right", framealpha=0.7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("时间")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.suptitle(f"传感器 #{moteid} 详细时序", fontweight="bold")
    plt.tight_layout()

    save_or_show(fig, f"02_sensor_{moteid:02d}_timeseries.png")
    return fig


# ============================================
# 3. 所有传感器对比 — 箱线图
# ============================================
def plot_sensor_boxplots(df: pd.DataFrame):
    """
    所有传感器的箱线图对比 — 一目了然看各节点差异
    """
    fig, axes = plt.subplots(2, 2, figsize=(22, 12))

    variables = ["temperature", "humidity", "light", "voltage"]
    titles = ["温度 Temperature (°C)", "湿度 Humidity (%)",
              "光照 Light (lux)", "电压 Voltage (V)"]

    for ax, var, title, color in zip(axes.flat, variables, titles, COLORS.values()):
        # 准备箱线图数据
        sensor_ids = sorted(df["moteid"].unique())
        data_by_sensor = [df[df["moteid"] == sid][var].dropna().values for sid in sensor_ids]

        bp = ax.boxplot(data_by_sensor, labels=sensor_ids, patch_artist=True,
                        widths=0.7, showfliers=False,  # 隐藏极端离群点以聚焦主体
                        medianprops={"color": "black", "linewidth": 1},
                        boxprops={"facecolor": color, "alpha": 0.6},
                        whiskerprops={"alpha": 0.6},
                        capprops={"alpha": 0.6})

        ax.set_title(title)
        ax.set_xlabel("传感器 ID")
        ax.set_ylabel(title.split("(")[0].strip())
        ax.tick_params(axis="x", labelsize=7, rotation=90)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("传感器对比 — 箱线图 (隐藏离群点)", fontweight="bold")
    plt.tight_layout()

    save_or_show(fig, "03_sensor_boxplots.png")
    return fig


# ============================================
# 4. 空间分布热力图 (基于X,Y位置)
# ============================================
def plot_spatial_heatmap(df: pd.DataFrame, df_pos: pd.DataFrame):
    """
    基于传感器位置的可视化:
    - 左: 传感器布局 + 温度着色
    - 右: 传感器布局 + 各变量均值柱状图
    """
    if df_pos is None or df_pos.empty:
        print("[跳过] 无位置数据, 无法绘制空间图")
        return None

    # 计算每个传感器的均值
    sensor_means = df.groupby("moteid")[["temperature", "humidity", "light", "voltage"]].mean()
    sensor_means = sensor_means.join(df_pos.set_index("moteid"), how="inner")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    variables = ["temperature", "humidity", "light", "voltage"]
    titles = ["平均温度 (°C)", "平均湿度 (%)", "平均光照 (lux)", "平均电压 (V)"]

    for ax, var, title, color in zip(axes.flat, variables, titles, COLORS.values()):
        sc = ax.scatter(
            sensor_means["x"], sensor_means["y"],
            c=sensor_means[var], cmap="RdYlGn_r" if var != "light" else "viridis",
            s=sensor_means[var] * 5 + 30,  # 气泡大小反映数值
            alpha=0.85, edgecolors="black", linewidth=0.5
        )
        # 标注传感器ID
        for _, row in sensor_means.iterrows():
            ax.annotate(
                str(int(row.name)), (row["x"], row["y"]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7, alpha=0.8
            )
        ax.set_xlabel("X 坐标")
        ax.set_ylabel("Y 坐标")
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, shrink=0.8, label=title.split("(")[0].strip())

    fig.suptitle("实验室传感器空间分布热力图", fontweight="bold")
    plt.tight_layout()

    save_or_show(fig, "04_spatial_heatmap.png")
    return fig


# ============================================
# 5. 变量相关性矩阵
# ============================================
def plot_correlation_matrix(df: pd.DataFrame):
    """
    四个传感器变量之间的相关性热力图
    """
    # 使用采样数据加速
    sample = df.iloc[::max(1, len(df)//50000)]

    corr = sample[["temperature", "humidity", "light", "voltage"]].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r",
                vmin=-1, vmax=1, center=0,
                mask=mask, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8, "label": "相关系数"},
                ax=ax,
                annot_kws={"fontsize": 12})

    ax.set_title("传感器变量相关性矩阵 (Pearson)", fontweight="bold", fontsize=14)

    plt.tight_layout()
    save_or_show(fig, "05_correlation_matrix.png")
    return fig


# ============================================
# 6. 分布直方图
# ============================================
def plot_distributions(df: pd.DataFrame):
    """
    四个变量的分布直方图 + KDE密度曲线
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    variables = ["temperature", "humidity", "light", "voltage"]
    titles = ["温度 Temperature (°C)", "湿度 Humidity (%)",
              "光照 Light (lux)", "电压 Voltage (V)"]

    for ax, var, title, color in zip(axes.flat, variables, titles, COLORS.values()):
        data = df[var].dropna()

        # 采样加速
        if len(data) > 200000:
            data = data.sample(200000, random_state=42)

        ax.hist(data, bins=80, density=True, alpha=0.6, color=color, edgecolor="white", linewidth=0.3)

        # KDE (如果seaborn可用)
        try:
            from scipy.stats import gaussian_kde
            kde_x = np.linspace(data.quantile(0.01), data.quantile(0.99), 500)
            kde = gaussian_kde(data)
            ax.plot(kde_x, kde(kde_x), color="black", linewidth=2, alpha=0.8, label="KDE 密度")
        except Exception as e:
            # Fallback to seaborn if scipy not available
            try:
                sns.kdeplot(data, ax=ax, color="black", linewidth=2, label="KDE 密度")
            except Exception:
                pass

        ax.axvline(data.median(), color="red", linestyle="--", linewidth=1.5, alpha=0.8, label=f"中位数={data.median():.2f}")
        ax.axvline(data.mean(), color="blue", linestyle="--", linewidth=1.5, alpha=0.8, label=f"均值={data.mean():.2f}")

        ax.set_title(title)
        ax.set_xlabel(title.split("(")[0].strip())
        ax.set_ylabel("密度")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("传感器数据分布直方图", fontweight="bold")
    plt.tight_layout()

    save_or_show(fig, "06_distributions.png")
    return fig


# ============================================
# 7. 日周期模式 (光照 & 温度按小时)
# ============================================
def plot_daily_pattern(df: pd.DataFrame):
    """
    按小时聚合, 查看温度和光照的日周期规律
    """
    df_copy = df.copy()
    df_copy["hour"] = df_copy.index.hour
    df_copy["dayofweek"] = df_copy.index.dayofweek  # 0=周一

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 左上: 温度按小时箱线图
    ax = axes[0, 0]
    hour_data = [df_copy[df_copy["hour"] == h]["temperature"].dropna().values for h in range(24)]
    bp = ax.boxplot(hour_data, labels=range(24), patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["temperature"])
        patch.set_alpha(0.5)
    ax.set_title("温度日周期分布 (按小时)")
    ax.set_xlabel("小时")
    ax.set_ylabel("温度 (°C)")
    ax.grid(True, alpha=0.3, axis="y")

    # 右上: 光照按小时箱线图
    ax = axes[0, 1]
    hour_data_light = [df_copy[df_copy["hour"] == h]["light"].dropna().values for h in range(24)]
    bp = ax.boxplot(hour_data_light, labels=range(24), patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["light"])
        patch.set_alpha(0.5)
    ax.set_title("光照日周期分布 (按小时)")
    ax.set_xlabel("小时")
    ax.set_ylabel("光照 (lux)")
    ax.grid(True, alpha=0.3, axis="y")

    # 左下: 温度按星期几
    ax = axes[1, 0]
    dow_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    dow_data = [df_copy[df_copy["dayofweek"] == d]["temperature"].dropna().values for d in range(7)]
    bp = ax.boxplot(dow_data, labels=dow_names, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["temperature"])
        patch.set_alpha(0.5)
    ax.set_title("温度按星期分布")
    ax.set_ylabel("温度 (°C)")
    ax.grid(True, alpha=0.3, axis="y")

    # 右下: 温度-湿度散点密度图
    ax = axes[1, 1]
    sample = df_copy.sample(min(30000, len(df_copy)), random_state=42)
    hb = ax.hexbin(sample["temperature"], sample["humidity"],
                   gridsize=50, cmap="YlOrRd", mincnt=1, alpha=0.9)
    ax.set_xlabel("温度 (°C)")
    ax.set_ylabel("湿度 (%)")
    ax.set_title("温度 vs 湿度 二维密度")
    plt.colorbar(hb, ax=ax, label="样本数")
    ax.grid(True, alpha=0.3)

    fig.suptitle("传感器数据日周期与关联模式", fontweight="bold")
    plt.tight_layout()

    save_or_show(fig, "07_daily_patterns.png")
    return fig


# ============================================
# 8. 综合报告生成器
# ============================================
def generate_full_report(df: pd.DataFrame, df_pos: pd.DataFrame = None,
                         sample_sensors: list = None):
    """
    一键生成全套可视化报告
    参数:
        df: 清洗后的DataFrame
        df_pos: 传感器位置DataFrame (可选)
        sample_sensors: 要生成单传感器详细图的ID列表 (默认选3个)
    """
    print("\n" + "="*60)
    print("  开始生成可视化报告...")
    print("="*60)

    # 挑选数据量最多的传感器做详细展示
    if sample_sensors is None:
        top_sensors = df["moteid"].value_counts().head(3).index.tolist()
    else:
        top_sensors = sample_sensors

    charts = []

    # 1. 综合概览
    print("\n[1/7] 生成综合概览仪表板...")
    plot_overview_dashboard(df)
    charts.append("概览仪表板")

    # 2. 箱线图对比
    print("[2/7] 生成传感器箱线图对比...")
    plot_sensor_boxplots(df)
    charts.append("箱线图对比")

    # 3. 分布直方图
    print("[3/7] 生成分布直方图...")
    plot_distributions(df)
    charts.append("分布直方图")

    # 4. 相关性矩阵
    print("[4/7] 生成相关性矩阵...")
    plot_correlation_matrix(df)
    charts.append("相关性矩阵")

    # 5. 日周期分析
    print("[5/7] 生成日周期模式图...")
    plot_daily_pattern(df)
    charts.append("日周期模式")

    # 6. 空间热力图 (需要位置数据)
    if df_pos is not None and not df_pos.empty:
        print("[6/7] 生成空间热力图...")
        plot_spatial_heatmap(df, df_pos)
        charts.append("空间热力图")
    else:
        print("[6/7] 跳过空间热力图 (无位置数据)")

    # 7. 代表性单传感器详细图
    print(f"[7/7] 生成代表性单传感器详细图 (IDs: {top_sensors})...")
    for sid in top_sensors:
        plot_single_sensor(df, sid)
        charts.append(f"传感器{sid}详细图")

    print("\n" + "="*60)
    print(f"  报告生成完成! 共 {len(charts)} 张图表")
    print(f"  输出目录: {OUTPUT_DIR}/")
    for c in charts:
        print(f"    ✓ {c}")
    print("="*60)


# ============================================
# 独立运行入口
# ============================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    # 测试: 如果没有导入外部数据, 使用模拟数据
    try:
        from data_cleaner import load_and_clean
        print("加载真实数据...")
        df, df_pos = load_and_clean()
        generate_full_report(df, df_pos, sample_sensors=[1, 25, 50])
    except Exception as e:
        print(f"无法加载真实数据 ({e}), 使用模拟数据演示...")
        # 生成模拟数据测试可视化
        np.random.seed(42)
        dates = pd.date_range("2004-02-28", "2004-04-02", periods=10000)
        n = len(dates)
        df_sim = pd.DataFrame({
            "datetime": dates,
            "moteid": np.random.choice(range(1, 55), n),
            "temperature": 20 + 5 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.randn(n) * 2,
            "humidity": 40 + 15 * np.sin(np.linspace(0, 3*np.pi, n) + 1) + np.random.randn(n) * 5,
            "light": np.clip(500 + 300 * np.sin(np.linspace(0, 2*np.pi, n)) + np.random.randn(n) * 100, 0, None),
            "voltage": 2.7 - np.linspace(0, 0.3, n) + np.random.randn(n) * 0.1,
        })
        df_sim.set_index("datetime", inplace=True)
        generate_full_report(df_sim, sample_sensors=[1, 2, 3])
