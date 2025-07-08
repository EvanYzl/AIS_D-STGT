"""AIS Data Preprocessing Module.

This module provides comprehensive preprocessing capabilities for AIS data including:
- Data cleaning and validation
- Coordinate transformation and projection
- Kalman filtering for trajectory smoothing
- Anomaly detection and outlier removal
- Data normalization and feature engineering

The preprocessing pipeline follows these stages:
1. Data cleaning and validation
2. Coordinate transformation (WGS84 to local projection)
3. Kalman filtering for trajectory smoothing
4. Anomaly detection and quality assessment
5. Data partitioning and output formatting
"""

from typing import List

__all__: list[str] = [
    "AnomalyDetector",
    "CoordinateTransformer",
    "DataCleaner",
    "KalmanFilter",
    "TrajectorySegmenter",
    "SegmentType",
    "TrajectorySegment",
    "SegmentationResult",
    "SceneAggregator",
    "SceneType",
    "Scene",
    "InteractionType",
    "InteractionEvent",
    "TCPADCPACalculator",
    "VesselPosition",
    "TCPAResult",
    "EnhancedTrajectorySmootherRTS",
    "SmoothedPoint",
    "SmoothingResult",
    "SceneDataset",
    "SceneFeatures",
    "DatasetConfig",
    "collate_scene_batch",
    "create_scene_dataloader",
]

from .anomaly_detector import AnomalyDetector
from .coordinate_transform import CoordinateTransformer
from .data_cleaner import DataCleaner
from .kalman_filter import KalmanFilter
from .scene_aggregator import (
    InteractionEvent,
    InteractionType,
    Scene,
    SceneAggregator,
    SceneType,
)
from .scene_dataset import (
    DatasetConfig,
    SceneDataset,
    SceneFeatures,
    collate_scene_batch,
    create_scene_dataloader,
)
from .tcpa_dcpa_calculator import TCPADCPACalculator, TCPAResult, VesselPosition
from .trajectory_segmenter import (
    SegmentationResult,
    SegmentType,
    TrajectorySegment,
    TrajectorySegmenter,
)
from .trajectory_smoother import (
    EnhancedTrajectorySmootherRTS,
    SmoothedPoint,
    SmoothingResult,
)
