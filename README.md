# AIS Data Processing Framework (AIS-DSTGT)

**Comprehensive AIS Data Processing with Advanced Trajectory Analysis**

[![CI](https://github.com/EvanYzl/AIS_D-STGT/workflows/CI/badge.svg)](https://github.com/EvanYzl/AIS_D-STGT/actions)
[![codecov](https://codecov.io/gh/EvanYzl/AIS_D-STGT/branch/main/graph/badge.svg)](https://codecov.io/gh/EvanYzl/AIS_D-STGT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

AIS-DSTGT is a comprehensive framework for processing and analyzing Automatic Identification System (AIS) maritime data. It provides advanced preprocessing capabilities including trajectory smoothing, behavior-based segmentation, scene aggregation, and collision risk assessment for deep learning applications in maritime domain.

## Key Features

- **Data Ingestion**: Robust CSV file handling with validation and error recovery
- **Advanced Preprocessing**: Multi-layer data cleaning, coordinate transformation, and anomaly detection
- **Trajectory Smoothing**: Enhanced Kalman filtering with RTS smoothing and gap interpolation
- **Trajectory Segmentation**: Behavior-based segmentation with port visit detection and voyage partitioning
- **Scene Aggregation**: Multi-vessel interaction analysis and navigation scene classification
- **Risk Assessment**: TCPA/DCPA calculations for collision risk evaluation
- **PyTorch Integration**: Custom dataset classes for efficient deep learning model training
- **Production Ready**: Docker support with comprehensive configuration options

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/EvanYzl/AIS_D-STGT.git
cd AIS_D-STGT

# Install with Poetry (recommended)
poetry install

# Or install with pip
pip install -e .
```

### Basic Usage

```python
from ais_dstgt.data.processor import AISDataProcessor

# Initialize processor with configuration
processor = AISDataProcessor()

# Load and process AIS data
df = processor.load_data('path/to/ais_data.csv')
processed_data = processor.process_data(df)

# Access processed components
smoothed_trajectories = processed_data['smoothed_trajectories']
trajectory_segments = processed_data['trajectory_segments']
scene_analysis = processed_data['scene_analysis']
risk_assessment = processed_data['risk_assessment']
```

### Docker Usage

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t ais-dstgt .
docker run --gpus all -p 8000:8000 ais-dstgt
```

## Architecture

```
ais_dstgt/
├── data/
│   ├── ingestion/           # Data loading and parsing
│   ├── preprocessing/       # Core preprocessing modules
│   │   ├── trajectory_segmenter.py      # Trajectory segmentation
│   │   ├── trajectory_smoother.py       # Enhanced smoothing
│   │   ├── scene_aggregator.py          # Scene analysis
│   │   ├── tcpa_dcpa_calculator.py      # Risk assessment
│   │   ├── scene_dataset.py             # PyTorch dataset
│   │   └── ...
│   ├── validation/          # Data quality validation
│   └── processor.py         # Main processing pipeline
```

## Preprocessing Modules

### 1. Enhanced Trajectory Smoother (`trajectory_smoother.py`)

The trajectory smoother implements advanced Kalman filtering with Rauch-Tung-Striebel (RTS) smoothing for optimal trajectory estimation.

**Key Features:**
- Local coordinate projection using Transverse Mercator
- Multi-layer outlier detection (speed, acceleration, statistical)
- Gap interpolation using physics-based predictions
- Quality metrics and uncertainty estimation

**Code Reference:**
```python
# Main smoothing pipeline (lines 120-180)
def smooth_trajectory(self, df: pd.DataFrame) -> SmoothingResult:
    # Project to local coordinates
    projected_data = self._project_to_local_coordinates(df_sorted)

    # Detect and remove outliers
    cleaned_data, outliers_removed = self._detect_and_remove_outliers(projected_data)

    # Fill gaps with interpolation
    filled_data, gaps_filled = self._fill_gaps(cleaned_data)

    # Apply Kalman filtering
    filtered_states, filtered_covariances = self._apply_kalman_filter(filled_data)

    # Apply RTS smoothing
    smoothed_states, smoothed_covariances = self._apply_rts_smoothing(
        filtered_states, filtered_covariances, filled_data
    )
```

### 2. Trajectory Segmentation (`trajectory_segmenter.py`)

The trajectory segmenter provides comprehensive trajectory partitioning based on maritime behavior patterns.

#### 时间划分策略 (Time Division Strategies)

**1. 基于行为的分段 (Behavior-Based Segmentation)**
- **速度变化检测**: 当速度变化超过阈值时分段轨迹
- **航向变化检测**: 识别显著的航向变化 (>30°)
- **时间窗口**: 可配置的分析窗口 (默认: 300秒)

```python
# 速度变化分段 (lines 180-200)
def _detect_speed_changes(self, df: pd.DataFrame) -> List[int]:
    speed_changes = []
    for i in range(1, len(df)):
        speed_change = abs(df.iloc[i]['speed'] - df.iloc[i-1]['speed'])
        if speed_change > self.speed_change_threshold:
            speed_changes.append(i)
    return speed_changes

# 航向变化分段 (lines 202-220)
def _detect_course_changes(self, df: pd.DataFrame) -> List[int]:
    course_changes = []
    for i in range(1, len(df)):
        course_change = self._calculate_course_change(
            df.iloc[i-1]['course'], df.iloc[i]['course']
        )
        if course_change > self.course_change_threshold:
            course_changes.append(i)
    return course_changes
```

**2. 港口访问检测 (Port Visit Detection)**
- **基于邻近度**: 检测船舶何时在港口边界内
- **基于速度**: 识别表示港口活动的低速期
- **持续时间阈值**: 最小停留时间 (默认: 1800秒)

```python
# 港口访问检测 (lines 250-290)
def _detect_port_visits(self, df: pd.DataFrame) -> List[Dict]:
    port_visits = []

    for port_name, port_location in self.port_locations.items():
        # 检查与港口的邻近度
        distances = df.apply(lambda row: geodesic(
            (row['latitude'], row['longitude']), port_location
        ).meters, axis=1)

        # 找到港口半径内的连续时期
        in_port = distances <= self.port_radius
        port_periods = self._find_continuous_periods(in_port)

        for start_idx, end_idx in port_periods:
            duration = (df.iloc[end_idx]['timestamp'] -
                       df.iloc[start_idx]['timestamp']).total_seconds()

            if duration >= self.min_port_duration:
                port_visits.append({
                    'port_name': port_name,
                    'start_index': start_idx,
                    'end_index': end_idx,
                    'duration': duration
                })
```

**3. 航次分割 (Voyage Partitioning)**
- **港口间段**: 在港口访问之间划分轨迹
- **开阔海域段**: 识别连续的导航期
- **时间间隙处理**: 基于时间不连续性的分段

```python
# 航次分割 (lines 320-360)
def _partition_voyage(self, df: pd.DataFrame, port_visits: List[Dict]) -> List[Dict]:
    voyage_segments = []

    # 按开始时间排序港口访问
    port_visits.sort(key=lambda x: x['start_index'])

    # 在港口之间创建段
    for i in range(len(port_visits) - 1):
        current_port = port_visits[i]
        next_port = port_visits[i + 1]

        # 从当前港口离开到下一个港口到达的航次段
        voyage_start = current_port['end_index']
        voyage_end = next_port['start_index']

        if voyage_end > voyage_start:
            voyage_segments.append({
                'type': 'voyage',
                'start_index': voyage_start,
                'end_index': voyage_end,
                'origin_port': current_port['port_name'],
                'destination_port': next_port['port_name']
            })
```

**4. 导航状态检测 (Navigation State Detection)**
- **锚泊**: 速度 < 0.5节 持续 > 30分钟
- **航行**: 速度 > 3节 且航向一致
- **操纵**: 频繁航向变化且速度可变
- **靠泊**: 港口设施附近的极低速度

```python
# 导航状态检测 (lines 380-420)
def _detect_navigation_states(self, df: pd.DataFrame) -> List[Dict]:
    states = []

    for i in range(len(df)):
        speed = df.iloc[i]['speed']

        if speed < 0.5:  # 锚泊阈值
            # 检查低速持续时间
            low_speed_duration = self._calculate_low_speed_duration(df, i)
            if low_speed_duration > 1800:  # 30分钟
                states.append({
                    'index': i,
                    'state': 'anchoring',
                    'duration': low_speed_duration
                })
        elif speed > 3.0:  # 航行阈值
            # 检查航向一致性
            course_stability = self._calculate_course_stability(df, i)
            if course_stability > 0.8:  # 高稳定性
                states.append({
                    'index': i,
                    'state': 'sailing',
                    'course_stability': course_stability
                })
```

### 3. Scene Aggregation (`scene_aggregator.py`)

场景聚合器分析多船舶交互并分类导航场景。

**场景分析的时间划分:**
- **时间窗口**: 可配置的分析窗口 (默认: 600秒)
- **滑动窗口**: 重叠分析以进行连续监控
- **事件驱动分段**: 基于交互事件的分段

```python
# 时间场景创建 (lines 150-200)
def _create_temporal_scenes(self, df: pd.DataFrame) -> List[Dict]:
    scenes = []

    # 创建滑动时间窗口
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    current_time = start_time

    while current_time < end_time:
        window_end = current_time + pd.Timedelta(seconds=self.scene_window_size)

        # 获取当前时间窗口中的船舶
        window_data = df[
            (df['timestamp'] >= current_time) &
            (df['timestamp'] < window_end)
        ]

        if len(window_data) > 0:
            scene = self._analyze_scene(window_data, current_time)
            scenes.append(scene)

        # 移动到下一个窗口
        current_time += pd.Timedelta(seconds=self.scene_step_size)

    return scenes
```

### 4. TCPA/DCPA Calculator (`tcpa_dcpa_calculator.py`)

使用向量化操作计算碰撞风险指标以提高效率。

**时间风险评估:**
- **预测时间跨度**: 可配置的时间范围 (默认: 1800秒)
- **风险更新**: 在每个时间步连续计算风险
- **动态邻接**: 基于邻近度更新船舶关系

```python
# 向量化TCPA/DCPA计算 (lines 120-160)
def calculate_tcpa_dcpa_vectorized(self, positions: np.ndarray,
                                  velocities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    多船舶对的向量化TCPA/DCPA计算

    Args:
        positions: 形状 (n_vessels, 2) - [x, y] 位置
        velocities: 形状 (n_vessels, 2) - [vx, vy] 速度

    Returns:
        tcpa: 形状 (n_vessels, n_vessels) - CPA时间矩阵
        dcpa: 形状 (n_vessels, n_vessels) - CPA距离矩阵
    """
    n_vessels = positions.shape[0]

    # 相对位置和速度
    rel_pos = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    rel_vel = velocities[:, np.newaxis, :] - velocities[np.newaxis, :, :]

    # TCPA计算
    rel_speed_sq = np.sum(rel_vel**2, axis=2)
    rel_pos_dot_vel = np.sum(rel_pos * rel_vel, axis=2)

    tcpa = np.where(
        rel_speed_sq > 1e-6,
        -rel_pos_dot_vel / rel_speed_sq,
        np.inf
    )

    # DCPA计算
    dcpa = np.sqrt(np.sum(rel_pos**2, axis=2) +
                   2 * tcpa * rel_pos_dot_vel +
                   tcpa**2 * rel_speed_sq)

    return tcpa, dcpa
```

### 5. Scene Dataset (`scene_dataset.py`)

用于高效模型训练的PyTorch数据集实现。

**数据组织:**
- **时间序列**: 将数据组织成基于时间的序列
- **特征提取**: 为模型输入提取相关特征
- **动态批处理**: 处理可变长度序列

```python
# PyTorch数据集项获取 (lines 180-220)
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    scene = self.scenes[idx]

    # 提取时间特征
    history_features = self._extract_history_features(scene)
    future_features = self._extract_future_features(scene)
    static_features = self._extract_static_features(scene)

    # 构建邻接矩阵
    adjacency_matrix = self._build_adjacency_matrix(scene)

    return {
        'history': torch.tensor(history_features, dtype=torch.float32),
        'future': torch.tensor(future_features, dtype=torch.float32),
        'static': torch.tensor(static_features, dtype=torch.float32),
        'adjacency': torch.tensor(adjacency_matrix, dtype=torch.float32),
        'scene_id': scene['scene_id']
    }
```

## Usage Examples

### Advanced Trajectory Analysis

```python
from ais_dstgt.data.preprocessing.trajectory_segmenter import TrajectorySegmenter

# 配置分段参数
segmenter = TrajectorySegmenter(
    speed_change_threshold=2.0,  # 节
    course_change_threshold=30.0,  # 度
    min_segment_duration=300.0,  # 秒
    port_radius=5000.0,  # 米
    min_port_duration=1800.0  # 秒
)

# 分段轨迹
segments = segmenter.segment_trajectories(df)

# 分析航次模式
voyage_analysis = segmenter.analyze_voyage_patterns(segments)
```

### Scene Analysis and Risk Assessment

```python
from ais_dstgt.data.preprocessing.scene_aggregator import SceneAggregator
from ais_dstgt.data.preprocessing.tcpa_dcpa_calculator import TCPADCPACalculator

# 初始化组件
scene_aggregator = SceneAggregator(scene_window_size=600)
risk_calculator = TCPADCPACalculator(prediction_horizon=1800)

# 分析海事场景
scenes = scene_aggregator.aggregate_scenes(df)
risk_graphs = risk_calculator.calculate_risk_graphs(scenes)

# 识别高风险场景
high_risk_scenes = [
    scene for scene in scenes
    if scene['risk_level'] > 0.7
]
```

## Development

### Prerequisites

- Python 3.12+
- Poetry for dependency management
- CUDA-compatible GPU (recommended)

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/EvanYzl/AIS_D-STGT.git
cd AIS_D-STGT

# Install dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest

# Run linting
poetry run black .
poetry run isort .
poetry run ruff check .
poetry run mypy ais_dstgt
```

## Configuration

The framework supports extensive configuration through parameter settings:

```python
# Trajectory Smoother Configuration
smoother_config = {
    'process_noise_std': 1.0,
    'observation_noise_std': 10.0,
    'max_gap_duration': 3600.0,
    'outlier_threshold': 3.0
}

# Trajectory Segmenter Configuration
segmenter_config = {
    'speed_change_threshold': 2.0,
    'course_change_threshold': 30.0,
    'min_segment_duration': 300.0,
    'port_locations': {
        'Port_A': (40.7128, -74.0060),
        'Port_B': (51.5074, -0.1278)
    }
}

# Scene Aggregator Configuration
scene_config = {
    'scene_window_size': 600,
    'scene_step_size': 300,
    'interaction_distance': 10000.0,
    'congestion_threshold': 5
}
```

## Performance Metrics

The framework provides comprehensive quality metrics:

- **Trajectory Smoothing**: Position accuracy, velocity consistency, gap interpolation quality
- **Segmentation Quality**: Segment coherence, behavior classification accuracy
- **Scene Analysis**: Interaction detection rate, scene classification accuracy
- **Risk Assessment**: TCPA/DCPA calculation accuracy, collision prediction performance

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
torch>=1.9.0
geopy>=2.2.0
scikit-learn>=1.0.0
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{ais_dstgt,
  title={AIS Data Processing Framework: Advanced Maritime Trajectory Analysis},
  author={Evan Yzl},
  year={2024},
  url={https://github.com/EvanYzl/AIS_D-STGT}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/EvanYzl/AIS_D-STGT/issues)
- **Discussions**: [GitHub Discussions](https://github.com/EvanYzl/AIS_D-STGT/discussions)

## Acknowledgments

- Maritime domain expertise from research publications
- AIS data processing methodologies from maritime safety organizations
- Deep learning frameworks for trajectory prediction research
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
   ```
2. 克隆代码并安装依赖
   ```bash
   git clone https://github.com/EvanYzl/AIS_D-STGT.git
   cd AIS_D-STGT
   # 仅清洗功能
   poetry install --without dev,docs
   # 若需投影、卡尔曼、异常检测
   poetry add pyproj filterpy scikit-learn pyarrow
   ```

### 2. 目录约定
```
AIS_D-STGT/
├── data/
│   ├── raw/        # 原始数据 (CSV)
│   ├── processed/  # 清洗后数据
│   └── external/   # 参考文档
└── ais_dstgt/      # 代码
```

### 3. 单文件处理示例
```python
from ais_dstgt.data import AISDataProcessor

processor = AISDataProcessor(
    output_dir="data/processed",
    enable_coordinate_transform=False,   # 仅清洗
    enable_kalman_filter=False,
    enable_anomaly_detection=False
)
processor.process_file(
    "data/raw/AIS_2024_01_01.csv",
    output_file="data/processed/AIS_2024_01_01.parquet",
    chunk_size=500_000
)
```

### 4. 启用高级功能
```python
processor = AISDataProcessor(
    output_dir="data/processed",
    enable_coordinate_transform=True,    # pyproj
    enable_kalman_filter=True,           # filterpy
    enable_anomaly_detection=True        # scikit-learn
)
processor.process_file("data/raw/AIS_2024_01_01.csv")
```

### 5. 多文件批处理
```python
processor = AISDataProcessor(output_dir="data/processed")
processor.process_directory("data/raw", file_pattern="*.csv", combine_files=True)
```

### 6. 分区存储
```python
from ais_dstgt.data import AISDataProcessor, AISDataFrame
from pathlib import Path

ais_df = AISDataFrame.read_parquet("data/processed/AIS_2024_01_01.parquet")
processor = AISDataProcessor()
processor.create_data_partitions(
    ais_df, partition_by="date", output_dir=Path("data/processed/partitions")
)
```

### 7. 报告文件
流水线会生成 `processing_report.json`，包含各步骤记录数、异常统计、过滤质量等信息，便于审计。

### 8. 常见问题
| 问题 | 解决方案 |
| ---- | -------- |
| ImportError: pyproj/filterpy/scikit-learn | `poetry add` 安装或在构造 `AISDataProcessor` 时禁用相关功能 |
| MemoryError | 调小 `chunk_size`，或使用 Parquet 分区 |
| BaseDateTime 解析失败 | 确保时间列为 ISO-8601 格式，或在 `CSVIngestionHandler` 传入 `parse_dates` 参数 |

> 📌 完整示例脚本见 `examples/process_ais_data.py`，支持 CLI 一键运行。
