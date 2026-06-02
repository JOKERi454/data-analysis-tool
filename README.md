# Data Analysis Toolbox — 传感器数据分析工具箱

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Pandas](https://img.shields.io/badge/Pandas-1.x+-150458.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x+-11557c.svg)](https://matplotlib.org/)

一个基于 Python 的**传感器数据一键分析工具箱**，覆盖从数据加载、清洗、统计分析到多维度可视化报告生成的全流程。

> Dataset: Intel Berkeley Research Lab sensor data (Feb 28 – Apr 2, 2004)  
> 54 个无线传感器节点 · ~200 万条记录 · 温度 / 湿度 / 光照 / 电压

---

## 功能概览

- **数据清洗流水线** — 缺失值检测、异常值标记、去重、线性插值填补、内存优化
- **统计分析** — 分组聚合统计、偏度/峰度、IQR 异常值计数、传感器排行
- **7 种可视化图表** — 概览面板、单传感器详情、箱线图、空间热力图、相关性矩阵、分布直方图、日/周周期分析
- **一键报告生成** — `python main_analysis.py` 即可走完加载→清洗→统计→出图全流程
- **CLI 灵活控制** — 支持快速模式、单传感器聚焦、CSV 导出、跳过可视化

---

## 项目结构

```
Data_Analysis_Toolbox/
├── data_cleaner.py          # 数据加载、解析、诊断与清洗
├── visualizer.py            # 多图表可视化引擎
├── main_analysis.py         # CLI 入口 / 流水线编排
├── __init__.py              # 包初始化
├── output_figures/          # 生成的图表输出目录
├── dfae3-main/              # 数据集目录
│   └── 传感器数据集/
│       ├── data完整.txt      # 原始传感器数据 (~200 万行)
│       └── 位置信息.txt      # 传感器 (X, Y) 坐标
└── README.md
```

## 快速开始

### 环境要求

- Python 3.8+
- pandas ≥ 1.0
- numpy ≥ 1.18
- matplotlib ≥ 3.0
- seaborn ≥ 0.11

### 安装依赖

```bash
pip install pandas numpy matplotlib seaborn
```

### 运行

```bash
# 全流程分析（加载 → 清洗 → 统计 → 出图）
python main_analysis.py

# 快速模式（20% 分层采样，适合快速预览）
python main_analysis.py --quick

# 聚焦单个传感器
python main_analysis.py --sensor 22

# 仅统计，不生成图表
python main_analysis.py --no-viz

# 导出清洗后的数据为 CSV
python main_analysis.py --export-csv

# 组合使用
python main_analysis.py --quick --export-csv
```

生成的图表保存在 `output_figures/` 目录下。

---

## 模块说明

### `data_cleaner.py` — 数据清洗

| 函数 | 功能 |
|---|---|
| `load_and_clean()` | 一键流水线：加载 → 解析日期 → 诊断 → 清洗 |
| `load_raw_data()` | 读取空格分隔的无表头文本文件 |
| `clean_data()` | 四步清洗：去重 → 异常值置 NaN → 线性插值 → 类型压缩 |
| `diagnose_data()` | 数据质量报告（缺失值、异常值、重复行、传感器分布） |
| `quick_stats()` | 按传感器分组聚合统计 |

### `visualizer.py` — 可视化

| 函数 | 图表类型 |
|---|---|
| `plot_overview_dashboard()` | 全传感器概览面板（4 变量时间序列） |
| `plot_single_sensor()` | 单传感器详情（含滑动平均线） |
| `plot_sensor_boxplots()` | 传感器间箱线图对比 |
| `plot_spatial_heatmap()` | 基于 (X,Y) 坐标的空间热力图 |
| `plot_correlation_matrix()` | 变量相关性热力图 |
| `plot_distributions()` | 分布直方图 + KDE + 均值/中位数标记 |
| `plot_daily_pattern()` | 日/周周期规律分析 |
| `generate_full_report()` | 一键生成全部图表 |

### `main_analysis.py` — 入口

| 参数 | 说明 |
|---|---|
| `--quick` | 20% 分层随机采样，加速执行 |
| `--sensor <ID>` | 仅分析指定传感器 |
| `--export-csv` | 导出清洗数据（UTF-8 BOM，上限 100 万行） |
| `--no-viz` | 跳过可视化，仅输出统计信息 |

---

## 数据集

数据来自 **Intel Berkeley Research Lab** 的 54 个无线传感器节点，部署于实验室内部。

- **时间范围**: 2004-02-28 ~ 2004-04-02
- **数据量**: 约 200 万条记录
- **变量**: 温度 (°C)、湿度 (%)、光照 (lux)、电压 (V)
- **原始来源**: http://db.lcs.mit.edu/labdata/labdata.html
- **许可**: MIT License

数据格式（空格分隔，无表头）：

```
2004-02-28 00:59:16.02785 3 1 19.9884 37.0933 45.08 2.69964
```

各列依次为：`date time epoch moteid temperature humidity light voltage`

位置信息格式（`moteid x y`）：

```
1 21.5 23
2 24.5 20
...
```

---

## 示例输出

运行 `python main_analysis.py` 后，`output_figures/` 目录将生成以下图表：

| 文件 | 内容 |
|---|---|
| `01_overview_dashboard.png` | 全传感器 4 变量时间序列概览 |
| `02_sensor_XX_timeseries.png` | 数据量最大的 3 个传感器详情 |
| `03_sensor_boxplots.png` | 各传感器数据分布箱线图 |
| `04_spatial_heatmap.png` | 传感器空间位置热力分布 |
| `05_correlation_matrix.png` | 4 变量 Pearson 相关性矩阵 |
| `06_distributions.png` | 各变量分布直方图 + KDE |
| `07_daily_patterns.png` | 按小时 / 按星期几的周期规律 |

---

## 许可

本项目基于 MIT License 开源。数据集部分遵循其原始 MIT 许可。
