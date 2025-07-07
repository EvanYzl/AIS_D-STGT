# AIS D-STGT

**Deep Spatio-Temporal Graph Transformer for Maritime Vessel Trajectory Prediction**

[![CI](https://github.com/EvanYzl/AIS_D-STGT/workflows/CI/badge.svg)](https://github.com/EvanYzl/AIS_D-STGT/actions)
[![codecov](https://codecov.io/gh/EvanYzl/AIS_D-STGT/branch/main/graph/badge.svg)](https://codecov.io/gh/EvanYzl/AIS_D-STGT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

AIS D-STGT is a deep learning framework for predicting maritime vessel trajectories using AIS (Automatic Identification System) data. The core model is based on a Deep Spatio-Temporal Graph Transformer (D-STGT) architecture that captures both spatial and temporal dependencies in vessel movement patterns.

## Key Features

- **Advanced Architecture**: Deep Spatio-Temporal Graph Transformer with attention mechanisms
- **Scalable Processing**: Efficient handling of large-scale AIS datasets
- **Flexible Configuration**: Configurable model parameters and training settings
- **Production Ready**: Docker support with GPU acceleration
- **Comprehensive Testing**: High test coverage with automated CI/CD
- **Rich Documentation**: Detailed API documentation and user guides

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
from ais_dstgt import TrajectoryPredictor

# Load pretrained model
predictor = TrajectoryPredictor.from_pretrained("path/to/model")

# Make predictions
predictions = predictor.predict(ais_data)
```

### Docker Usage

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t ais-dstgt .
docker run --gpus all -p 8000:8000 ais-dstgt
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

### Project Structure

```
AIS_D-STGT/
├── ais_dstgt/              # Main package
│   ├── __init__.py
│   ├── models/             # Model implementations
│   ├── data/               # Data processing
│   ├── training/           # Training utilities
│   └── utils/              # Utility functions
├── tests/                  # Test suite
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── pyproject.toml          # Poetry configuration
├── Dockerfile              # Docker configuration
└── docker-compose.yml      # Docker Compose configuration
```

## Code Quality Standards

This project follows strict code quality standards:

- **Python 3.12** with full type annotations
- **Black** for code formatting (line length 88)
- **isort** for import sorting
- **Ruff** for linting
- **mypy** for type checking (strict mode)
- **pytest** for testing (≥80% coverage)
- **pre-commit** hooks for automated checks

## Git Workflow

- **main**: Production-ready code
- **dev**: Development integration branch
- **feature/**: Feature development branches
- **hotfix/**: Bug fix branches

### Commit Format

```
feat|fix|refactor|docs|test: <summary>

<optional body>

<optional footer>
```

### Example Workflow

```bash
# Create feature branch
git checkout -b feature/new-model-architecture

# Make changes and commit
git add .
git commit -m "feat: implement attention mechanism for D-STGT model"

# Push with tags
git push --follow-tags origin feature/new-model-architecture

# Create pull request to dev branch
```

## Documentation

- **[Full Documentation](https://evanyzl.github.io/AIS_D-STGT/)** - Complete user guide and API reference
- **[Installation Guide](docs/getting-started/installation.md)** - Detailed setup instructions
- **[API Reference](docs/reference/)** - Comprehensive API documentation
- **[Contributing Guide](docs/development/contributing.md)** - Development guidelines

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{ais_dstgt,
  title={AIS D-STGT: Deep Spatio-Temporal Graph Transformer for Maritime Vessel Trajectory Prediction},
  author={Evan Yzl},
  year={2023},
  url={https://github.com/EvanYzl/AIS_D-STGT}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/EvanYzl/AIS_D-STGT/issues)
- **Discussions**: [GitHub Discussions](https://github.com/EvanYzl/AIS_D-STGT/discussions)
- **Documentation**: [Project Documentation](https://evanyzl.github.io/AIS_D-STGT/)

## Acknowledgments

- NOAA for providing AIS data access
- The open-source community for the excellent tools and libraries
- Maritime research community for insights and feedback

## 数据预处理流水线使用指南

本节介绍如何在 Linux 平台（Ubuntu 22.04+/RHEL 9+/WSL2 等）使用 `ais_dstgt.data` 提供的流水线，对原始 AIS CSV 数据进行清洗、平滑、异常检测与缓存。

### 1. 环境准备
1. 安装 Python 3.12+ 与 Poetry
   ```bash
   sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv build-essential
   curl -sSL https://install.python-poetry.org | python3 -
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
