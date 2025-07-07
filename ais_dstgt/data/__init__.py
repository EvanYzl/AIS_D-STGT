"""AIS D-STGT Data Processing Module.

This module provides comprehensive data processing capabilities for AIS
(Automatic Identification System) data, including:

- Data ingestion from various sources (NOAA, local files, etc.)
- Data preprocessing and cleaning
- Data validation and quality assessment
- Utility functions for data manipulation

The data processing pipeline follows these stages:
1. Raw data ingestion
2. Data cleaning and validation
3. Coordinate transformation and filtering
4. Anomaly detection and quality control
5. Output to structured formats (Parquet, HDF5)

Example:
    Basic usage of the data processing pipeline:

    >>> from ais_dstgt.data import AISDataProcessor
    >>> processor = AISDataProcessor()
    >>> clean_data = processor.process_file("data/raw/ais_data.csv")
"""

from typing import List

__all__: list[str] = [
    "AISDataProcessor",
    "CSVIngestionHandler",
    "DataCleaner",
    "CoordinateTransformer",
    "KalmanFilter",
    "AnomalyDetector",
    "AISDataFrame",
    "AISRecord",
]

from .ingestion.csv_handler import CSVIngestionHandler
from .preprocessing.anomaly_detector import AnomalyDetector
from .preprocessing.coordinate_transform import CoordinateTransformer
from .preprocessing.data_cleaner import DataCleaner
from .preprocessing.kalman_filter import KalmanFilter

# Import main classes
from .processor import AISDataProcessor
from .schemas import AISDataFrame, AISRecord
