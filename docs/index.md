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

```bash
# Install with Poetry
poetry install

# Or install with pip
pip install ais-dstgt

# Basic usage
from ais_dstgt import TrajectoryPredictor

predictor = TrajectoryPredictor.from_pretrained("path/to/model")
predictions = predictor.predict(ais_data)
```

## Architecture

The D-STGT model combines:

- **Graph Neural Networks**: For modeling spatial relationships between vessels
- **Transformer Architecture**: For capturing long-term temporal dependencies
- **Attention Mechanisms**: For focusing on relevant spatial-temporal patterns
- **Multi-scale Processing**: For handling different temporal resolutions

## Use Cases

- **Maritime Traffic Management**: Predict vessel movements for traffic optimization
- **Collision Avoidance**: Early warning systems for potential collisions
- **Port Operations**: Optimize berth allocation and scheduling
- **Environmental Monitoring**: Track vessel emissions and environmental impact
- **Security Applications**: Detect anomalous vessel behavior

## Getting Started

1. [Installation](getting-started/installation.md) - Set up the development environment
2. [Quick Start](getting-started/quickstart.md) - Run your first prediction
3. [Configuration](getting-started/configuration.md) - Customize model parameters

## Documentation

- [User Guide](user-guide/data-processing.md) - Comprehensive usage instructions
- [API Reference](reference/) - Detailed API documentation
- [Development](development/contributing.md) - Contributing guidelines

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/EvanYzl/AIS_D-STGT/blob/main/LICENSE) file for details.

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
