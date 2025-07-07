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
    "AISPreprocessor",
    "CoordinateTransformer",
    "KalmanFilter",
    "AnomalyDetector",
    "DataCleaner",
]

# from .processor import AISPreprocessor
# from .coordinate_transform import CoordinateTransformer
# from .kalman_filter import KalmanFilter
# from .anomaly_detector import AnomalyDetector
# from .data_cleaner import DataCleaner
