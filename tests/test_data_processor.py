"""Tests for AIS Data Processing Pipeline."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Determine if scikit-learn is available
SKLEARN_AVAILABLE = True
try:
    import sklearn  # type: ignore
except ImportError:
    SKLEARN_AVAILABLE = False

# Check pyproj availability
PYPROJ_AVAILABLE = True
try:
    import pyproj  # type: ignore
except ImportError:
    PYPROJ_AVAILABLE = False

# Check filterpy availability
FILTERPY_AVAILABLE = True
try:
    import filterpy  # type: ignore
except ImportError:
    FILTERPY_AVAILABLE = False

from ais_dstgt.data import (
    AISDataFrame,
    AISDataProcessor,
    AnomalyDetector,
    CoordinateTransformer,
    CSVIngestionHandler,
    DataCleaner,
    KalmanFilter,
)


@pytest.fixture
def sample_ais_data():
    """Create sample AIS data for testing."""
    # Generate sample data
    np.random.seed(42)
    n_records = 1000
    n_vessels = 5

    data = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)

    for vessel_id in range(n_vessels):
        mmsi = 123456780 + vessel_id
        # Generate trajectory for each vessel
        n_points = n_records // n_vessels

        # Starting position (random location in Atlantic)
        start_lat = 40.0 + np.random.uniform(-5, 5)
        start_lon = -70.0 + np.random.uniform(-5, 5)

        for i in range(n_points):
            # Simple random walk trajectory
            lat = start_lat + np.cumsum(np.random.normal(0, 0.001, i + 1))[-1]
            lon = start_lon + np.cumsum(np.random.normal(0, 0.001, i + 1))[-1]

            # Ensure valid ranges
            lat = np.clip(lat, -90, 90)
            lon = np.clip(lon, -180, 180)

            timestamp = base_time + timedelta(minutes=i * 10)

            record = {
                "MMSI": mmsi,
                "BaseDateTime": timestamp,
                "LAT": lat,
                "LON": lon,
                "SOG": np.random.uniform(0, 20),  # Speed in knots
                "COG": np.random.uniform(0, 360),  # Course in degrees
                "Heading": np.random.choice(
                    [np.random.uniform(0, 360), 511]
                ),  # Heading or N/A
                "VesselName": f"TEST_VESSEL_{vessel_id}",
                "VesselType": 70,  # Cargo vessel
                "Status": 0,  # Under way using engine
                "Length": 100.0,
                "Width": 15.0,
                "Draft": 5.0,
                "TransceiverClass": "A",
            }
            data.append(record)

    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(tmp_path, sample_ais_data):
    """Create a temporary CSV file with sample data."""
    csv_file = tmp_path / "test_ais_data.csv"
    sample_ais_data.to_csv(csv_file, index=False)
    return csv_file


class TestAISDataFrame:
    """Test AISDataFrame wrapper class."""

    def test_initialization(self, sample_ais_data):
        """Test AISDataFrame initialization."""
        ais_df = AISDataFrame(sample_ais_data)

        assert len(ais_df.df) == len(sample_ais_data)
        assert "MMSI" in ais_df.df.columns
        assert "BaseDateTime" in ais_df.df.columns

    def test_validation(self, sample_ais_data):
        """Test data validation."""
        ais_df = AISDataFrame(sample_ais_data)

        # Should not raise any validation errors
        invalid_records = ais_df.validate_ranges()
        assert isinstance(invalid_records, list)

    def test_summary_stats(self, sample_ais_data):
        """Test summary statistics."""
        ais_df = AISDataFrame(sample_ais_data)

        stats = ais_df.get_summary_stats()

        assert "total_records" in stats
        assert "unique_vessels" in stats
        assert "time_range" in stats
        assert "geographic_bounds" in stats
        assert stats["total_records"] == len(sample_ais_data)
        assert stats["unique_vessels"] == sample_ais_data["MMSI"].nunique()


class TestCSVIngestionHandler:
    """Test CSV ingestion handler."""

    def test_load_file(self, sample_csv_file):
        """Test loading a CSV file."""
        handler = CSVIngestionHandler()

        ais_data = handler.load_file(sample_csv_file)

        assert isinstance(ais_data, AISDataFrame)
        assert len(ais_data.df) > 0
        assert "MMSI" in ais_data.df.columns

    def test_get_file_info(self, sample_csv_file):
        """Test getting file information."""
        handler = CSVIngestionHandler()

        info = handler.get_file_info(sample_csv_file)

        assert "file_path" in info
        assert "file_size_bytes" in info
        assert "columns" in info
        assert "encoding" in info

    def test_validate_csv_structure(self, sample_csv_file):
        """Test CSV structure validation."""
        handler = CSVIngestionHandler()

        validation = handler.validate_csv_structure(sample_csv_file)

        assert "is_valid" in validation
        assert "missing_required_columns" in validation
        assert validation["is_valid"] is True


class TestDataCleaner:
    """Test data cleaning functionality."""

    def test_clean_data(self, sample_ais_data):
        """Test data cleaning pipeline."""
        cleaner = DataCleaner()
        ais_df = AISDataFrame(sample_ais_data)

        cleaned_data, report = cleaner.clean_data(ais_df)

        assert isinstance(cleaned_data, AISDataFrame)
        assert isinstance(report, dict)
        assert "initial_records" in report
        assert "final_records" in report
        assert "records_removed" in report

    def test_remove_duplicates(self, sample_ais_data):
        """Test duplicate removal."""
        # Add some duplicates
        duplicated_data = pd.concat([sample_ais_data, sample_ais_data.head(10)])

        cleaner = DataCleaner()
        ais_df = AISDataFrame(duplicated_data)

        cleaned_data, report = cleaner.clean_data(ais_df)

        # Should have removed duplicates
        assert len(cleaned_data.df) <= len(duplicated_data)

    def test_data_quality_report(self, sample_ais_data):
        """Test data quality report generation."""
        cleaner = DataCleaner()
        ais_df = AISDataFrame(sample_ais_data)

        report = cleaner.get_data_quality_report(ais_df)

        assert "total_records" in report
        assert "unique_vessels" in report
        assert "time_range" in report
        assert "geographic_bounds" in report


@pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not installed")
class TestCoordinateTransformer:
    """Test coordinate transformation."""

    def test_setup_projection(self, sample_ais_data):
        """Test projection setup."""
        transformer = CoordinateTransformer()
        ais_df = AISDataFrame(sample_ais_data)

        projection_info = transformer.setup_projection(ais_df, "utm")

        assert "projection_type" in projection_info
        assert "source_crs" in projection_info
        assert "target_crs" in projection_info
        assert projection_info["projection_type"] == "utm"

    def test_transform_coordinates(self, sample_ais_data):
        """Test coordinate transformation."""
        transformer = CoordinateTransformer()
        ais_df = AISDataFrame(sample_ais_data)

        # Setup projection
        transformer.setup_projection(ais_df, "utm")

        # Transform coordinates
        transformed_data = transformer.transform_coordinates(ais_df)

        assert "X" in transformed_data.df.columns
        assert "Y" in transformed_data.df.columns
        assert "LON_ORIG" in transformed_data.df.columns
        assert "LAT_ORIG" in transformed_data.df.columns

    def test_calculate_distances(self, sample_ais_data):
        """Test distance calculations."""
        transformer = CoordinateTransformer()
        ais_df = AISDataFrame(sample_ais_data)

        # Setup projection and transform
        transformer.setup_projection(ais_df, "utm")
        transformed_data = transformer.transform_coordinates(ais_df)

        # Calculate distances
        distance_data = transformer.calculate_distances(transformed_data)

        assert "Distance" in distance_data.df.columns
        assert "CalculatedSpeed" in distance_data.df.columns
        assert "TimeDelta" in distance_data.df.columns


@pytest.mark.skipif(not FILTERPY_AVAILABLE, reason="filterpy not installed")
class TestKalmanFilter:
    """Test Kalman filtering."""

    def test_filter_trajectory(self, sample_ais_data):
        """Test trajectory filtering."""
        kalman_filter = KalmanFilter()
        ais_df = AISDataFrame(sample_ais_data)

        filtered_data = kalman_filter.filter_trajectory(ais_df)

        assert "LON_filtered" in filtered_data.df.columns
        assert "LAT_filtered" in filtered_data.df.columns
        assert "velocity_x_filtered" in filtered_data.df.columns
        assert "velocity_y_filtered" in filtered_data.df.columns
        assert "filter_quality" in filtered_data.df.columns

    def test_filtering_quality_report(self, sample_ais_data):
        """Test filtering quality report."""
        kalman_filter = KalmanFilter()
        ais_df = AISDataFrame(sample_ais_data)

        # Apply filtering
        filtered_data = kalman_filter.filter_trajectory(ais_df)

        # Get quality report
        report = kalman_filter.get_filtering_quality_report(filtered_data)

        assert "total_vessels_filtered" in report
        assert "quality_statistics" in report


class TestAnomalyDetector:
    """Test anomaly detection."""

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_detect_anomalies(self, sample_ais_data):
        """Test anomaly detection."""
        detector = AnomalyDetector()
        ais_df = AISDataFrame(sample_ais_data)

        anomaly_data, report = detector.detect_anomalies(ais_df)

        assert "anomaly_kinematic" in anomaly_data.df.columns
        assert "anomaly_signal_gap" in anomaly_data.df.columns
        assert "anomaly_composite" in anomaly_data.df.columns
        assert "anomaly_score" in anomaly_data.df.columns

        assert "total_records" in report
        assert "anomaly_counts" in report
        assert "total_anomalies" in report

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_anomaly_report(self, sample_ais_data):
        """Test anomaly report generation."""
        detector = AnomalyDetector()
        ais_df = AISDataFrame(sample_ais_data)

        # Detect anomalies
        anomaly_data, _ = detector.detect_anomalies(ais_df)

        # Get anomaly report
        report = detector.get_anomaly_report(anomaly_data)

        assert "total_records" in report
        assert "anomaly_statistics" in report
        assert "overall_anomaly_rate" in report


class TestAISDataProcessor:
    """Test main data processor."""

    def test_initialization(self, tmp_path):
        """Test processor initialization."""
        processor = AISDataProcessor(output_dir=tmp_path / "output")

        assert processor.output_dir.exists()
        assert isinstance(processor.csv_handler, CSVIngestionHandler)
        assert isinstance(processor.data_cleaner, DataCleaner)
        if PYPROJ_AVAILABLE:
            assert isinstance(processor.coordinate_transformer, CoordinateTransformer)
        else:
            assert processor.coordinate_transformer is None
        assert isinstance(processor.kalman_filter, KalmanFilter)
        assert isinstance(processor.anomaly_detector, AnomalyDetector)

    def test_process_file(self, sample_csv_file, tmp_path):
        """Test processing a single file."""
        processor = AISDataProcessor(
            output_dir=tmp_path / "output",
            enable_coordinate_transform=False,  # Disable for faster testing
            enable_kalman_filter=False,
            enable_anomaly_detection=False,
        )

        processed_data = processor.process_file(sample_csv_file)

        assert isinstance(processed_data, AISDataFrame)
        assert len(processed_data.df) > 0

        # Check that processing report was generated
        report = processor.get_processing_report()
        assert "pipeline_steps" in report
        assert "data_quality_metrics" in report

    def test_processing_report(self, sample_csv_file, tmp_path):
        """Test processing report generation."""
        processor = AISDataProcessor(
            output_dir=tmp_path / "output",
            enable_coordinate_transform=False,
            enable_kalman_filter=False,
            enable_anomaly_detection=False,
        )

        # Process file
        processor.process_file(sample_csv_file)

        # Get report
        report = processor.get_processing_report()

        assert isinstance(report, dict)
        assert "pipeline_steps" in report
        assert "data_quality_metrics" in report

        # Save report
        report_file = tmp_path / "processing_report.json"
        processor.save_processing_report(report_file)

        assert report_file.exists()

    def test_create_data_partitions(self, sample_ais_data, tmp_path):
        """Test data partitioning."""
        processor = AISDataProcessor(output_dir=tmp_path / "output")
        ais_df = AISDataFrame(sample_ais_data)

        # Create date partitions
        partition_files = processor.create_data_partitions(
            ais_df, partition_by="date", output_dir=tmp_path / "partitions"
        )

        assert len(partition_files) > 0
        assert all(f.exists() for f in partition_files)
        assert all(f.suffix == ".parquet" for f in partition_files)


@pytest.mark.skipif(
    not PYPROJ_AVAILABLE or not SKLEARN_AVAILABLE or not FILTERPY_AVAILABLE,
    reason="Required libraries not installed",
)
@pytest.mark.integration
class TestIntegrationPipeline:
    """Integration tests for the complete pipeline."""

    def test_complete_pipeline(self, sample_csv_file, tmp_path):
        """Test complete data processing pipeline."""
        processor = AISDataProcessor(
            output_dir=tmp_path / "output",
            enable_kalman_filter=True,
            enable_anomaly_detection=True,
            enable_coordinate_transform=True,
        )

        # Process file with all features enabled
        processed_data = processor.process_file(sample_csv_file)

        # Verify all processing steps were applied
        df = processed_data.df

        # Check coordinate transformation
        assert "X" in df.columns
        assert "Y" in df.columns

        # Check Kalman filtering
        assert "LON_filtered" in df.columns or "X_filtered" in df.columns
        assert "filter_quality" in df.columns

        # Check anomaly detection
        assert "anomaly_composite" in df.columns
        assert "anomaly_score" in df.columns

        # Check processing report
        report = processor.get_processing_report()
        step_names = [step["step"] for step in report["pipeline_steps"]]

        assert "data_loading" in step_names
        assert "data_cleaning" in step_names
        assert "coordinate_transformation" in step_names
        assert "kalman_filtering" in step_names
        assert "anomaly_detection" in step_names

    def test_pipeline_with_chunking(self, sample_csv_file, tmp_path):
        """Test pipeline with chunked processing."""
        processor = AISDataProcessor(
            output_dir=tmp_path / "output",
            enable_coordinate_transform=False,  # Disable for chunked processing
            enable_kalman_filter=True,
            enable_anomaly_detection=True,
        )

        # Process with small chunk size
        processed_data = processor.process_file(sample_csv_file, chunk_size=100)

        assert isinstance(processed_data, AISDataFrame)
        assert len(processed_data.df) > 0

        # Check that chunked processing was used
        report = processor.get_processing_report()
        assert "chunked_processing" in report
