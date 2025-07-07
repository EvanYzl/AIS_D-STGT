"""AIS D-STGT: Deep Spatio-Temporal Graph Transformer for Maritime Vessel Trajectory Prediction.

This package implements a deep learning framework for predicting maritime vessel
trajectories using AIS (Automatic Identification System) data. The core model
is based on a Deep Spatio-Temporal Graph Transformer (D-STGT) architecture.

Example:
    Basic usage of the package:

    >>> from ais_dstgt import TrajectoryPredictor
    >>> predictor = TrajectoryPredictor.from_pretrained("path/to/model")
    >>> predictions = predictor.predict(ais_data)
"""

__version__ = "0.1.0"
__author__ = "EvanYzl"
__email__ = "3258244847@qq.com"

__all__: list[str] = [
    "__version__",
    "__author__",
    "__email__",
]
