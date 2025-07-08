# AIS 数据处理与海事场景分析系统 (AIS D-STGT)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目概述

AIS D-STGT 是一个专为海事研究设计的船舶自动识别系统（AIS）数据处理与分析平台。该系统实现了从原始AIS数据清洗到复杂海事场景构建的完整流程，支持轨迹预测、碰撞风险评估、交通流分析等多种海事研究应用。

### 核心功能

- **数据预处理**：AIS数据清洗、异常检测、坐标变换
- **轨迹处理**：卡尔曼滤波、轨迹平滑、分段处理
- **场景构建**：多船交互场景聚合、时空窗口分析
- **风险评估**：TCPA/DCPA计算、碰撞风险分析
- **机器学习**：PyTorch数据集构建、图神经网络支持

## 系统架构

```
ais_dstgt/
├── data/                    # 数据处理核心模块
│   ├── ingestion/          # 数据接入
│   ├── preprocessing/      # 数据预处理
│   ├── validation/         # 数据验证
│   └── processor.py        # 主处理器
├── examples/               # 使用示例
├── tests/                  # 单元测试
└── docs/                   # 文档
```

## 核心模块详解

### 1. 数据预处理模块 (`data/preprocessing/`)

#### 1.1 数据清洗器 (`data_cleaner.py`)
- **功能**：清洗原始AIS数据，去除无效记录
- **关键参数**：
  - `speed_threshold`: 最大合理速度（默认50节）
  - `position_accuracy`: 位置精度阈值（默认0.001度）
  - `time_gap_threshold`: 时间间隔阈值（默认3600秒）

#### 1.2 异常检测器 (`anomaly_detector.py`)
- **功能**：检测轨迹中的异常点和离群值
- **算法**：基于统计学方法和机器学习的异常检测
- **参数设置**：
  - `outlier_threshold`: 异常值检测阈值（默认3个标准差）
  - `min_trajectory_length`: 最小轨迹长度（默认10个点）

#### 1.3 卡尔曼滤波器 (`kalman_filter.py`)
- **功能**：轨迹平滑和状态估计
- **参数依据**：
  - `process_noise`: 过程噪声（默认1.0 m/s²），基于船舶运动特性
  - `measurement_noise`: 观测噪声（默认10.0 m），基于AIS精度规范

#### 1.4 轨迹分段器 (`trajectory_segmenter.py`)
- **功能**：将单船轨迹按行为模式分段
- **分段类型**：
  - `VOYAGE`: 航行段
  - `PORT_VISIT`: 港口停靠
  - `ANCHORING`: 锚泊
  - `MANEUVERING`: 操纵
  - `TRANSIT`: 过境

**关键参数设置依据**：
```python
# 基于海事规则和实际操作经验
speed_threshold = 2.0          # 2节，区分航行与操纵
course_threshold = 30.0        # 30度，显著航向变化
time_gap_threshold = 3600.0    # 1小时，数据断点判定
anchoring_time_threshold = 1800.0  # 30分钟，锚泊最短时间
```

### 2. 场景聚合模块 (`scene_aggregator.py`)

#### 2.1 核心概念
场景聚合是将清洗后的单船轨迹片段组合成多船交互场景的过程。一个场景包含在同一时空窗口内共同出现的N艘船的轨迹片段。

#### 2.2 关键参数及科研依据

| 参数 | 默认值 | 科研依据 |
|------|--------|----------|
| `scene_radius_km` | 7.0 | 基于VHF通信范围（~5-10海里），覆盖典型船舶交互距离 |
| `time_window_minutes` | 5.0 | 平衡实时性与稳定性，符合IMO建议的6分钟评估周期 |
| `min_scene_duration_minutes` | 10.0 | 过滤瞬时聚簇，确保场景有意义的交互时间 |
| `max_scene_duration_minutes` | 90.0 | 避免过长场景包含多种行为模式 |
| `congestion_density_threshold` | 4.0 | 基于主要港口统计数据（vessels/km²） |
| `interaction_distance_threshold` | 1.852 | 1海里，符合COLREGS避碰规则 |
| `port_approach_distance_km` | 12.0 | 覆盖典型港口引航区域 |
| `speed_threshold_knots` | 0.8 | 区分锚泊与航行状态 |

#### 2.3 场景类型分类

```python
class SceneType(Enum):
    PORT_APPROACH = "port_approach"    # 港口接近
    PORT_DEPARTURE = "port_departure"  # 港口离开
    TRANSIT = "transit"                # 过境航行
    ANCHORING = "anchoring"            # 锚泊区域
    CONGESTION = "congestion"          # 交通拥堵
    INTERACTION = "interaction"        # 船舶交互
    WATERWAY = "waterway"             # 航道航行
    OPEN_WATER = "open_water"         # 开阔水域
```

### 3. 场景数据集 (`scene_dataset.py`)

#### 3.1 时间窗口参数

**历史轨迹与预测轨迹的时间配置**：

| 参数 | 默认值 | 含义 | 科研依据 |
|------|--------|------|----------|
| `history_length` | 20 | 历史时间步数 | 20×60s=20分钟，覆盖两轮避碰决策循环 |
| `future_length` | 10 | 预测时间步数 | 10×60s=10分钟，符合轨迹预测文献标准 |
| `time_step_seconds` | 60 | 时间步长 | 与AIS平均采样频率对齐 |

**重要说明**：
- 场景聚合的5分钟时间窗口用于"同时出现"的船舶聚类
- 数据集的20分钟历史+10分钟预测用于机器学习模型训练
- 两者服务于不同目的，相互独立

#### 3.2 图网络参数

```python
# 节点特征（船舶）
node_features = [
    "speed",              # 速度
    "course",             # 航向
    "acceleration",       # 加速度
    "course_change",      # 航向变化率
    "distance_to_center", # 到场景中心距离
    "relative_bearing"    # 相对方位
]

# 边特征（交互）
edge_distance_threshold = 2000.0  # 2km，约1海里
min_vessel_count = 2              # 最小船舶数
max_vessel_count = 50             # 最大船舶数（显存友好）
```

### 4. 风险评估模块 (`tcpa_dcpa_calculator.py`)

#### 4.1 TCPA/DCPA计算
- **TCPA** (Time to Closest Point of Approach): 最近会遇时间
- **DCPA** (Distance at Closest Point of Approach): 最近会遇距离

#### 4.2 风险等级评估
```python
# 风险等级判定标准
RISK_LEVELS = {
    "HIGH": (tcpa < 600 and dcpa < 0.5),      # 10分钟内，0.5海里内
    "MEDIUM": (tcpa < 1200 and dcpa < 1.0),   # 20分钟内，1海里内
    "LOW": (tcpa < 1800 and dcpa < 2.0),      # 30分钟内，2海里内
}
```

## 使用指南

### 1. 基本使用

```python
from ais_dstgt.data.processor import AISDataProcessor
from ais_dstgt.data.preprocessing import SceneAggregator, DatasetConfig

# 创建处理器
processor = AISDataProcessor(
    enable_scene_aggregation=True,
    enable_trajectory_segmentation=True,
    enable_risk_assessment=True
)

# 处理AIS数据
ais_data = processor.process_file("raw_ais_data.csv")

# 创建场景数据集
scene_config = {
    "history_length": 20,      # 20分钟历史
    "future_length": 10,       # 10分钟预测
    "time_step_seconds": 60,   # 60秒步长
    "cache_dir": "cache/",     # 缓存目录
}

dataset, metadata = processor.create_scene_dataset(ais_data, scene_config)
```

### 2. 自定义参数配置

```python
# 自定义场景聚合参数
scene_aggregator = SceneAggregator(
    scene_radius_km=5.0,                    # 港内场景使用较小半径
    time_window_minutes=3.0,                # 高频数据使用更短时间窗
    congestion_density_threshold=6.0,       # 繁忙港口提高拥堵阈值
    interaction_distance_threshold=1.0,     # 更严格的交互距离
)

# 自定义数据集配置
dataset_config = DatasetConfig(
    history_length=30,                      # 更长历史用于复杂预测
    future_length=15,                       # 更长预测窗口
    time_step_seconds=30,                   # 更高时间分辨率
    max_vessel_count=100,                   # 大型锚地场景
)
```

### 3. 场景分析示例

```python
# 获取场景统计信息
stats = scene_aggregator.get_scene_statistics(scenes)
print(f"总场景数: {stats['total_scenes']}")
print(f"场景类型分布: {stats['scene_types']}")
print(f"平均船舶数: {stats['avg_vessel_count']:.1f}")
print(f"平均持续时间: {stats['avg_duration_hours']:.1f}小时")

# 分析特定类型场景
congestion_scenes = [s for s in scenes if s.scene_type == SceneType.CONGESTION]
interaction_scenes = [s for s in scenes if s.scene_type == SceneType.INTERACTION]

print(f"拥堵场景: {len(congestion_scenes)}")
print(f"交互场景: {len(interaction_scenes)}")
```

## 参数调优建议

### 1. 基于数据特征调优

**高频AIS数据（≤1分钟间隔）**：
```python
time_window_minutes = 3.0          # 更短时间窗
time_step_seconds = 30             # 更高分辨率
scene_radius_km = 5.0              # 更精细空间划分
```

**稀疏AIS数据（>5分钟间隔）**：
```python
time_window_minutes = 10.0         # 更长时间窗
time_step_seconds = 120            # 适应数据频率
scene_radius_km = 15.0             # 更大覆盖范围
```

### 2. 基于研究目标调优

**短期碰撞预警**：
```python
history_length = 10                # 10分钟历史
future_length = 5                  # 5分钟预测
interaction_distance_threshold = 1.0  # 更严格距离
```

**长期流量预测**：
```python
history_length = 60                # 60分钟历史
future_length = 30                 # 30分钟预测
time_step_seconds = 300            # 5分钟步长
```

### 3. 基于计算资源调优

**GPU显存限制**：
```python
max_vessel_count = 30              # 减少最大船舶数
scene_radius_km = 5.0              # 减少场景规模
edge_distance_threshold = 1500.0   # 减少边连接
```

**大规模数据处理**：
```python
precompute_features = True         # 预计算特征
cache_dir = "cache/"               # 启用缓存
chunk_size = 10000                 # 分块处理
```

## 性能优化

### 1. 并行处理
```python
# 多进程数据加载
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,          # 并行加载
    pin_memory=True,        # GPU加速
)
```

### 2. 缓存策略
```python
# 启用特征缓存
scene_config = {
    "cache_dir": "cache/features/",
    "precompute_features": True,
}
```

### 3. 内存管理
```python
# 分块处理大文件
processor.process_file(
    "large_ais_data.csv",
    chunk_size=50000        # 5万条记录一块
)
```

## 数据格式

### 输入数据格式
```csv
mmsi,timestamp,latitude,longitude,speed,course,heading,status
123456789,2023-01-01T00:00:00,51.5074,-0.1278,10.5,045,043,0
```

### 场景数据格式
```json
{
    "scene_id": "transit_20230101_1000_5v",
    "scene_type": "transit",
    "start_time": "2023-01-01T10:00:00",
    "end_time": "2023-01-01T10:15:00",
    "center_lat": 51.5074,
    "center_lon": -0.1278,
    "radius_km": 7.0,
    "vessels": [123456789, 987654321],
    "vessel_count": 2,
    "properties": {
        "avg_speed": 12.3,
        "density": 0.04,
        "interactions": 1
    }
}
```

## 验证与测试

### 1. 单元测试
```bash
python -m pytest tests/
```

### 2. 数据质量检查
```python
# 检查处理结果
report = processor.get_processing_report()
print(f"数据质量得分: {report['quality_score']:.2f}")
print(f"异常点数量: {report['anomalies_detected']}")
```

### 3. 场景有效性验证
```python
# 验证场景合理性
for scene in scenes:
    if scene.vessel_count < 2:
        print(f"警告: 场景{scene.scene_id}船舶数不足")
    if scene.duration.total_seconds() < 300:
        print(f"警告: 场景{scene.scene_id}持续时间过短")
```

## 扩展开发

### 1. 自定义场景类型
```python
class CustomSceneType(Enum):
    FISHING_AREA = "fishing_area"
    CONSTRUCTION_ZONE = "construction_zone"
    
# 扩展分类逻辑
def custom_scene_classifier(vessel_states):
    # 自定义分类逻辑
    pass
```

### 2. 新增特征提取
```python
def extract_custom_features(trajectories):
    # 自定义特征提取
    return custom_features
```

### 3. 集成外部数据
```python
# 天气数据集成
weather_data = load_weather_data()
enriched_scenes = enrich_with_weather(scenes, weather_data)
```

## 常见问题

### Q1: 场景数量过少怎么办？
**A**: 尝试以下调整：
- 增大`scene_radius_km`
- 减少`min_vessels_per_scene`
- 增加`time_window_minutes`
- 检查数据质量和覆盖范围

### Q2: 内存不足如何处理？
**A**: 
- 启用分块处理：`chunk_size=10000`
- 减少`max_vessel_count`
- 使用缓存：`cache_dir="cache/"`
- 减少`history_length`和`future_length`

### Q3: 场景类型分类不准确？
**A**:
- 检查港口和航道定义
- 调整速度和密度阈值
- 增加训练数据
- 使用领域专家知识标注

### Q4: 预测精度不满足要求？
**A**:
- 增加`history_length`
- 优化特征工程
- 调整模型架构
- 增加训练数据量

## 引用

如果您在研究中使用了本系统，请引用：

```bibtex
@software{ais_dstgt,
  title={AIS Data Processing and Maritime Scene Analysis System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/AIS_D-STGT}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交问题报告和改进建议！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

## 联系方式

- 项目主页: https://github.com/yourusername/AIS_D-STGT
- 问题报告: https://github.com/yourusername/AIS_D-STGT/issues
- 邮箱: your.email@example.com

---

*最后更新: 2024年*


