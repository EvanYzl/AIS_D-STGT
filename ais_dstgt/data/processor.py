"""Main AIS Data Processor.

This module provides the main data processing pipeline that integrates all
data processing steps including ingestion, cleaning, transformation, filtering,
and anomaly detection.
"""

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .ingestion.csv_handler import CSVIngestionHandler
from .preprocessing.anomaly_detector import AnomalyDetector
from .preprocessing.coordinate_transform import CoordinateTransformer
from .preprocessing.data_cleaner import DataCleaner
from .preprocessing.kalman_filter import KalmanFilter
from .schemas import AISDataFrame

logger = logging.getLogger(__name__)


class AISDataProcessor:
    """Main AIS data processing pipeline."""

    def __init__(
        self,
        output_dir: str | Path = "data/processed",
        enable_kalman_filter: bool = True,
        enable_anomaly_detection: bool = True,
        enable_coordinate_transform: bool = True,
        projection_type: str = "utm",
    ):
        """Initialize AIS data processor.

        Args:
            output_dir: Directory for processed data output
            enable_kalman_filter: Whether to apply Kalman filtering
            enable_anomaly_detection: Whether to perform anomaly detection
            enable_coordinate_transform: Whether to apply coordinate transformation
            projection_type: Type of coordinate projection ('utm', 'mercator')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_kalman_filter = enable_kalman_filter
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_coordinate_transform = enable_coordinate_transform
        self.projection_type = projection_type

        # Initialize components
        self.csv_handler = CSVIngestionHandler()
        self.data_cleaner = DataCleaner()
        try:
            self.coordinate_transformer = CoordinateTransformer()
        except ImportError:
            logger.warning(
                "pyproj not installed, disabling coordinate transformation features."
            )
            self.coordinate_transformer = None
            self.enable_coordinate_transform = False
        self.kalman_filter = KalmanFilter()
        self.anomaly_detector = AnomalyDetector()

        # Processing report
        self.processing_report = {
            "pipeline_steps": [],
            "data_quality_metrics": {},
            "performance_metrics": {},
        }

    def process_file(
        self,
        input_file: str | Path,
        output_file: str | Path | None = None,
        chunk_size: int | None = None,
    ) -> AISDataFrame:
        """Process a single AIS data file.

        Args:
            input_file: Path to input CSV file
            output_file: Path to output file (optional)
            chunk_size: Process in chunks if specified

        Returns:
            AISDataFrame: Processed AIS data
        """
        input_file = Path(input_file)

        if output_file is None:
            output_file = self.output_dir / f"{input_file.stem}_processed.parquet"

        logger.info(f"Processing AIS file: {input_file}")

        # Process in chunks if specified
        if chunk_size:
            return self._process_file_chunked(input_file, output_file, chunk_size)
        else:
            return self._process_file_complete(input_file, output_file)

    def _process_file_complete(
        self, input_file: Path, output_file: Path
    ) -> AISDataFrame:
        """Process complete file in memory."""
        # Step 1: Load data
        logger.info("Step 1: Loading data")
        ais_data = self.csv_handler.load_file(input_file)
        self.processing_report["pipeline_steps"].append(
            {
                "step": "data_loading",
                "records_input": len(ais_data.df),
                "records_output": len(ais_data.df),
            }
        )

        # Step 2: Data cleaning
        logger.info("Step 2: Data cleaning")
        ais_data, cleaning_report = self.data_cleaner.clean_data(ais_data)
        self.processing_report["pipeline_steps"].append(
            {
                "step": "data_cleaning",
                "records_input": cleaning_report["initial_records"],
                "records_output": cleaning_report["final_records"],
                "records_removed": cleaning_report["records_removed"],
                "removal_rate": cleaning_report["removal_rate"],
            }
        )

        # Step 3: Coordinate transformation
        if self.enable_coordinate_transform and self.coordinate_transformer is not None:
            logger.info("Step 3: Coordinate transformation")
            projection_info = self.coordinate_transformer.setup_projection(
                ais_data, self.projection_type
            )
            ais_data = self.coordinate_transformer.transform_coordinates(ais_data)
            ais_data = self.coordinate_transformer.calculate_distances(ais_data)

            self.processing_report["pipeline_steps"].append(
                {
                    "step": "coordinate_transformation",
                    "projection_type": self.projection_type,
                    "projection_info": projection_info,
                }
            )

        # Step 4: Kalman filtering
        if self.enable_kalman_filter:
            logger.info("Step 4: Kalman filtering")
            ais_data = self.kalman_filter.filter_trajectory(ais_data)
            filtering_report = self.kalman_filter.get_filtering_quality_report(ais_data)

            self.processing_report["pipeline_steps"].append(
                {"step": "kalman_filtering", "filtering_quality": filtering_report}
            )

        # Step 5: Anomaly detection
        if self.enable_anomaly_detection:
            logger.info("Step 5: Anomaly detection")
            ais_data, anomaly_report = self.anomaly_detector.detect_anomalies(ais_data)

            self.processing_report["pipeline_steps"].append(
                {"step": "anomaly_detection", "anomaly_report": anomaly_report}
            )

        # Step 6: Save processed data
        logger.info("Step 6: Saving processed data")
        self._save_processed_data(ais_data, output_file)

        # Generate final report
        self._generate_processing_report(ais_data)

        logger.info(f"Processing completed. Output saved to: {output_file}")
        return ais_data

    def _process_file_chunked(
        self, input_file: Path, output_file: Path, chunk_size: int
    ) -> AISDataFrame:
        """Process file in chunks for memory efficiency."""
        logger.info(f"Processing file in chunks of {chunk_size} records")

        # Initialize chunk reader
        chunk_reader = self.csv_handler.load_chunked(input_file)

        processed_chunks = []
        chunk_reports = []

        # Process each chunk
        for i, chunk_df in enumerate(tqdm(chunk_reader, desc="Processing chunks")):
            logger.info(f"Processing chunk {i+1}")

            # Wrap in AISDataFrame
            chunk_ais = AISDataFrame(chunk_df)

            # Apply processing steps
            chunk_ais = self._process_chunk(chunk_ais)

            processed_chunks.append(chunk_ais.df)

            # Collect chunk statistics
            chunk_reports.append({"chunk_id": i + 1, "records": len(chunk_ais.df)})

        # Combine all chunks
        logger.info("Combining processed chunks")
        combined_df = pd.concat(processed_chunks, ignore_index=True)
        final_ais_data = AISDataFrame(combined_df)

        # Final processing steps that require full dataset
        if self.enable_coordinate_transform and self.coordinate_transformer is not None:
            self.coordinate_transformer.setup_projection(
                final_ais_data, self.projection_type
            )
            final_ais_data = self.coordinate_transformer.transform_coordinates(
                final_ais_data
            )

        # Save processed data
        self._save_processed_data(final_ais_data, output_file)

        # Update processing report
        self.processing_report["chunked_processing"] = {
            "total_chunks": len(processed_chunks),
            "chunk_reports": chunk_reports,
        }

        logger.info(f"Chunked processing completed. Output saved to: {output_file}")
        return final_ais_data

    def _process_chunk(self, chunk_ais: AISDataFrame) -> AISDataFrame:
        """Process a single chunk of data."""
        # Data cleaning
        chunk_ais, _ = self.data_cleaner.clean_data(chunk_ais)

        # Kalman filtering (per-vessel basis)
        if self.enable_kalman_filter:
            chunk_ais = self.kalman_filter.filter_trajectory(chunk_ais)

        # Anomaly detection
        if self.enable_anomaly_detection:
            chunk_ais, _ = self.anomaly_detector.detect_anomalies(chunk_ais)

        return chunk_ais

    def process_directory(
        self,
        input_dir: str | Path,
        file_pattern: str = "*.csv",
        combine_files: bool = False,
    ) -> AISDataFrame | list[AISDataFrame]:
        """Process all files in a directory.

        Args:
            input_dir: Directory containing AIS files
            file_pattern: File pattern to match
            combine_files: Whether to combine all files into one dataset

        Returns:
            AISDataFrame or List[AISDataFrame]: Processed data
        """
        input_dir = Path(input_dir)

        # Find all matching files
        files = list(input_dir.glob(file_pattern))

        if not files:
            raise ValueError(
                f"No files found matching pattern {file_pattern} in {input_dir}"
            )

        logger.info(f"Found {len(files)} files to process")

        processed_data = []

        # Process each file
        for file_path in tqdm(files, desc="Processing files"):
            try:
                processed_file = self.process_file(file_path)
                processed_data.append(processed_file)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue

        if not processed_data:
            raise ValueError("No files were successfully processed")

        # Combine files if requested
        if combine_files:
            logger.info("Combining all processed files")
            combined_df = pd.concat(
                [data.df for data in processed_data], ignore_index=True
            )

            # Save combined dataset
            combined_output = self.output_dir / "combined_processed.parquet"
            combined_ais = AISDataFrame(combined_df)
            self._save_processed_data(combined_ais, combined_output)

            return combined_ais

        return processed_data

    def _save_processed_data(self, ais_data: AISDataFrame, output_file: Path) -> None:
        """Save processed data to file."""
        output_file = Path(output_file)

        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Determine output format based on file extension
        if output_file.suffix.lower() == ".parquet":
            ais_data.to_parquet(output_file)
        elif output_file.suffix.lower() == ".csv":
            ais_data.to_csv(output_file)
        else:
            # Default to parquet
            output_file = output_file.with_suffix(".parquet")
            ais_data.to_parquet(output_file)

        logger.info(f"Saved processed data to: {output_file}")

    def _generate_processing_report(self, ais_data: AISDataFrame) -> None:
        """Generate comprehensive processing report."""
        # Data quality metrics
        self.processing_report["data_quality_metrics"] = {
            "final_record_count": len(ais_data.df),
            "unique_vessels": ais_data.df["MMSI"].nunique(),
            "time_range": {
                "start": ais_data.df["BaseDateTime"].min(),
                "end": ais_data.df["BaseDateTime"].max(),
            },
            "geographic_coverage": {
                "lat_range": [ais_data.df["LAT"].min(), ais_data.df["LAT"].max()],
                "lon_range": [ais_data.df["LON"].min(), ais_data.df["LON"].max()],
            },
        }

        # Add anomaly statistics if available
        if "anomaly_composite" in ais_data.df.columns:
            anomaly_stats = self.anomaly_detector.get_anomaly_report(ais_data)
            self.processing_report["data_quality_metrics"][
                "anomaly_statistics"
            ] = anomaly_stats

        # Add filtering statistics if available
        if "filter_quality" in ais_data.df.columns:
            filtering_stats = self.kalman_filter.get_filtering_quality_report(ais_data)
            self.processing_report["data_quality_metrics"][
                "filtering_statistics"
            ] = filtering_stats

    def get_processing_report(self) -> dict:
        """Get comprehensive processing report.

        Returns:
            Dict: Processing report with all steps and metrics
        """
        return self.processing_report

    def save_processing_report(self, output_file: str | Path) -> None:
        """Save processing report to JSON file.

        Args:
            output_file: Path to output JSON file
        """
        import json

        output_file = Path(output_file)

        # Convert datetime objects to strings for JSON serialization
        report_copy = self._serialize_report_for_json(self.processing_report)

        with open(output_file, "w") as f:
            json.dump(report_copy, f, indent=2, default=str)

        logger.info(f"Processing report saved to: {output_file}")

    def _serialize_report_for_json(self, obj):
        """Recursively serialize report for JSON output."""
        if isinstance(obj, dict):
            return {k: self._serialize_report_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_report_for_json(item) for item in obj]
        elif hasattr(obj, "isoformat"):  # datetime objects
            return obj.isoformat()
        elif isinstance(obj, pd.Timestamp | pd.Timedelta):
            return str(obj)
        else:
            return obj

    def create_data_partitions(
        self,
        ais_data: AISDataFrame,
        partition_by: str = "date",
        output_dir: str | Path | None = None,
    ) -> list[Path]:
        """Create partitioned datasets for efficient storage and access.

        Args:
            ais_data: Input AIS data
            partition_by: Partitioning strategy ('date', 'vessel', 'region')
            output_dir: Output directory for partitions

        Returns:
            List[Path]: List of created partition files
        """
        if output_dir is None:
            output_dir = self.output_dir / "partitions"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = ais_data.df
        partition_files = []

        if partition_by == "date":
            # Partition by date
            df["date"] = df["BaseDateTime"].dt.date

            for date in df["date"].unique():
                date_data = df[df["date"] == date]
                partition_file = output_dir / f"date={date}.parquet"

                date_ais = AISDataFrame(date_data.drop("date", axis=1))
                date_ais.to_parquet(partition_file)
                partition_files.append(partition_file)

        elif partition_by == "vessel":
            # Partition by vessel (MMSI)
            for mmsi in df["MMSI"].unique():
                vessel_data = df[df["MMSI"] == mmsi]
                partition_file = output_dir / f"vessel={mmsi}.parquet"

                vessel_ais = AISDataFrame(vessel_data)
                vessel_ais.to_parquet(partition_file)
                partition_files.append(partition_file)

        elif partition_by == "region":
            # Partition by geographic region (simple grid)
            lat_bins = pd.cut(df["LAT"], bins=10, labels=False)
            lon_bins = pd.cut(df["LON"], bins=10, labels=False)

            for lat_bin in range(10):
                for lon_bin in range(10):
                    mask = (lat_bins == lat_bin) & (lon_bins == lon_bin)
                    region_data = df[mask]

                    if len(region_data) > 0:
                        partition_file = (
                            output_dir / f"region={lat_bin}_{lon_bin}.parquet"
                        )

                        region_ais = AISDataFrame(region_data)
                        region_ais.to_parquet(partition_file)
                        partition_files.append(partition_file)

        logger.info(f"Created {len(partition_files)} partitions in {output_dir}")
        return partition_files
