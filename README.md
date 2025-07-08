# AIS-DSTGT：AIS 海事数据处理框架

**高级轨迹分析与深度学习预处理管道**

[![CI](https://github.com/EvanYzl/AIS_D-STGT/workflows/CI/badge.svg)](https://github.com/EvanYzl/AIS_D-STGT/actions)
[![codecov](https://codecov.io/gh/EvanYzl/AIS_D-STGT/branch/main/graph/badge.svg)](https://codecov.io/gh/EvanYzl/AIS_D-STGT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 项目简介

AIS-DSTGT 是一个面向 **海事场景认知与轨迹预测** 的 AIS（Automatic Identification System）数据处理框架，提供从原始 CSV 文件到深度学习模型训练所需特征的 **端到端预处理流水线**。

主要特点如下：

- **数据摄取**：稳健的 CSV 解析与字段校验
- **多层预处理**：数据清洗、坐标转换、异常检测
- **轨迹平滑**：基于 **卡尔曼滤波 + RTS 平滑** 的高精度轨迹估计
- **轨迹分割**：结合速度/航向变化与港口访问检测的行为分段
- **场景聚合**：多船舶交互识别与导航场景分类
- **风险评估**：向量化 **TCPA/DCPA** 计算，高效碰撞风险评估
- **PyTorch 数据集**：按需加载、动态图结构，支持大规模训练

---

## 目录结构

```text
ais_dstgt/
├── data/
│   ├── ingestion/            # 数据导入与解析
│   ├── preprocessing/        # 预处理核心模块
│   │   ├── trajectory_smoother.py      # 轨迹平滑
│   │   ├── trajectory_segmenter.py     # 轨迹分割
│   │   ├── scene_aggregator.py         # 场景聚合
│   │   ├── tcpa_dcpa_calculator.py     # 碰撞风险
│   │   ├── scene_dataset.py            # PyTorch 数据集
│   │   └── ...
│   ├── processor.py          # 主处理流水线
│   └── validation/           # 数据质量验证
└── examples/                 # 使用示例脚本
```

---

## 预处理模块详解

### 1. 轨迹平滑 (`trajectory_smoother.py`)

- **本地坐标投影**：采用横轴墨卡托（Transverse Mercator）实现高精度局部投影；
- **卡尔曼滤波**：常速度模型预测，融合位置观测；
- **RTS 平滑**：后向递归优化，提高轨迹平滑度；
- **间隙插值**：对 ≤1 小时数据缺口进行物理一致性插值；
- **质量评估**：输出位置/速度不确定度及综合质量分。

### 2. 轨迹分割 (`trajectory_segmenter.py`)

#### 时间划分策略

1. **行为分段**
   - 速度阈值变化（默认 2 节）
   - 航向显著变化（默认 30°）
   - 分析窗口：300 秒
2. **港口访问检测**
   - 半径 5 km 内判定靠港
   - 停留时间 ≥30 分钟记为一次访问
3. **航次分割**
   - 以连续两次港口访问为界，提取航次片段
4. **导航状态识别**
   - 锚泊 / 航行 / 操纵 / 靠泊

### 3. 场景聚合 (`scene_aggregator.py`)

- **滑动窗口**：600 秒窗口、300 秒步长
- **交互识别**：超车、交叉、并航等典型场景
- **交通密度**：基于单位面积船舶数量评估拥堵
- **水域/港口判别**：支持多边形海区与港口范围自定义

### 4. 碰撞风险 (`tcpa_dcpa_calculator.py`)

- 向量化 **TCPA**（最近相遇时间）/ **DCPA**（最近相遇距离）矩阵
- 预测时域：默认 1800 秒
- 支持动态阈值筛选，生成风险图与邻接矩阵

### 5. PyTorch 数据集 (`scene_dataset.py`)

- **按需加载**：JIT 处理，降低内存占用
- **动态图**：基于距离/风险实时计算邻接
- **自定义 collate_fn**：支持变长批次

---

## 快速上手

```python
from ais_dstgt.data.processor import AISDataProcessor

processor = AISDataProcessor()

df_raw = processor.load_data("data/raw/ais_sample.csv")
result = processor.process_data(df_raw)

print(result.keys())
# ['smoothed_trajectories', 'trajectory_segments', 'scene_analysis', 'risk_assessment']
```

### 高级示例：轨迹分割

```python
from ais_dstgt.data.preprocessing.trajectory_segmenter import TrajectorySegmenter

segmenter = TrajectorySegmenter(
    speed_change_threshold=2.0,
    course_change_threshold=30.0,
    min_segment_duration=300.0,
    port_radius=5000.0,
    min_port_duration=1800.0,
)

segments = segmenter.segment_trajectories(df_raw)
```

---

## 配置示例

```python
# 场景聚合器
scene_cfg = {
    "scene_window_size": 600,
    "scene_step_size": 300,
    "interaction_distance": 10000.0,
    "congestion_threshold": 5,
}
```

---

## 性能指标

- **位置均方误差 (RMS)**
- **速度误差 (knots)**
- **加速度平滑度 (m/s²)**
- **插值比例 / 异常剔除比例**
- **场景交互检测准确率**

---

## 依赖

```
Python >= 3.12
numpy >= 1.21
pandas >= 1.3
scipy >= 1.7
torch >= 1.9
geopy >= 2.2
scikit-learn >= 1.0
```

---

## 安装

```bash
# 克隆仓库
git clone https://github.com/EvanYzl/AIS_D-STGT.git
cd AIS_D-STGT

# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e .
```

---

## 许可证

本项目基于 MIT 许可证发布，详见 `LICENSE` 文件。

---

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@software{ais_dstgt,
  title  = {AIS-DSTGT: AIS 数据处理与深度轨迹分析框架},
  author = {Evan Yzl},
  year   = {2024},
  url    = {https://github.com/EvanYzl/AIS_D-STGT}
}
```

---

## 致谢

- 感谢开源社区提供的优秀工具与库；
- 感谢海事研究领域的专家学者提供的技术参考与数据支持。
