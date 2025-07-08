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
from .preprocessing.scene_aggregator import SceneAggregator
from .preprocessing.scene_dataset import DatasetConfig, SceneDataset
from .preprocessing.tcpa_dcpa_calculator import TCPADCPACalculator
from .preprocessing.trajectory_segmenter import TrajectorySegmenter
from .preprocessing.trajectory_smoother import EnhancedTrajectorySmootherRTS
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
        enable_trajectory_segmentation: bool = True,
        enable_scene_aggregation: bool = True,
        enable_enhanced_smoothing: bool = True,
        enable_risk_assessment: bool = True,
        projection_type: str = "utm",
    ):
        """Initialize AIS data processor.

        Args:
            output_dir: Directory for processed data output
            enable_kalman_filter: Whether to apply Kalman filtering
            enable_anomaly_detection: Whether to perform anomaly detection
            enable_coordinate_transform: Whether to apply coordinate transformation
            enable_trajectory_segmentation: Whether to apply trajectory segmentation
            enable_scene_aggregation: Whether to apply scene aggregation
            enable_enhanced_smoothing: Whether to use enhanced trajectory smoothing
            enable_risk_assessment: Whether to perform TCPA/DCPA risk assessment
            projection_type: Type of coordinate projection ('utm', 'mercator')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_kalman_filter = enable_kalman_filter
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_coordinate_transform = enable_coordinate_transform
        self.enable_trajectory_segmentation = enable_trajectory_segmentation
        self.enable_scene_aggregation = enable_scene_aggregation
        self.enable_enhanced_smoothing = enable_enhanced_smoothing
        self.enable_risk_assessment = enable_risk_assessment
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
        self.trajectory_segmenter = TrajectorySegmenter()
        self.scene_aggregator = SceneAggregator()
        self.tcpa_calculator = TCPADCPACalculator()
        self.trajectory_smoother = EnhancedTrajectorySmootherRTS()

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

        # Step 4: Enhanced trajectory smoothing
        if self.enable_enhanced_smoothing:
            logger.info("Step 4: Enhanced trajectory smoothing")
            smoothing_results = self.trajectory_smoother.smooth_multiple_trajectories(
                ais_data.df
            )

            # Convert smoothed results back to DataFrame
            smoothed_df = self.trajectory_smoother.export_smoothed_trajectories(
                smoothing_results
            )
            if not smoothed_df.empty:
                ais_data.df = smoothed_df

            smoothing_stats = self.trajectory_smoother.get_smoothing_statistics(
                smoothing_results
            )
            self.processing_report["pipeline_steps"].append(
                {"step": "enhanced_smoothing", "smoothing_report": smoothing_stats}
            )

        # Step 5: Kalman filtering (fallback if enhanced smoothing disabled)
        elif self.enable_kalman_filter:
            logger.info("Step 5: Kalman filtering")
            ais_data = self.kalman_filter.filter_trajectory(ais_data)
            filtering_report = self.kalman_filter.get_filtering_quality_report(ais_data)

            self.processing_report["pipeline_steps"].append(
                {"step": "kalman_filtering", "filtering_quality": filtering_report}
            )

        # Step 6: Trajectory segmentation
        if self.enable_trajectory_segmentation:
            logger.info("Step 6: Trajectory segmentation")
            segmentation_results = (
                self.trajectory_segmenter.segment_multiple_trajectories(ais_data.df)
            )

            # Extract segments and update data
            all_segments = []
            for mmsi, result in segmentation_results.items():
                for segment in result.segments:
                    segment_data = {
                        "mmsi": mmsi,
                        "segment_id": segment.segment_id,
                        "segment_type": segment.segment_type.value,
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "duration_minutes": segment.duration.total_seconds() / 60,
                        "points_count": len(segment.points),
                        "avg_speed": segment.avg_speed,
                        "max_speed": segment.max_speed,
                        "distance_km": segment.distance_km,
                    }
                    all_segments.append(segment_data)

            segmentation_stats = self.trajectory_segmenter.get_segmentation_statistics(
                segmentation_results
            )
            self.processing_report["pipeline_steps"].append(
                {
                    "step": "trajectory_segmentation",
                    "segmentation_report": segmentation_stats,
                }
            )

        # Step 7: Risk assessment (TCPA/DCPA)
        if self.enable_risk_assessment:
            logger.info("Step 7: Risk assessment")
            tcpa_results = self.tcpa_calculator.calculate_batch_tcpa_dcpa(ais_data.df)

            # Export results to DataFrame and add to processing report
            if tcpa_results:
                self.tcpa_calculator.export_results_to_dataframe(tcpa_results)
                risk_stats = self.tcpa_calculator.get_statistics(tcpa_results)
            else:
                pd.DataFrame()
                risk_stats = {}

            self.processing_report["pipeline_steps"].append(
                {"step": "risk_assessment", "risk_report": risk_stats}
            )

        # Step 8: Scene aggregation
        if self.enable_scene_aggregation:
            logger.info("Step 8: Scene aggregation")
            scenes = self.scene_aggregator.aggregate_scenes(ais_data.df)

            # Export scenes and get statistics
            if scenes:
                self.scene_aggregator.export_scenes_to_dataframe(scenes)
                aggregation_stats = self.scene_aggregator.get_scene_statistics(scenes)
            else:
                pd.DataFrame()
                aggregation_stats = {}

            self.processing_report["pipeline_steps"].append(
                {"step": "scene_aggregation", "aggregation_report": aggregation_stats}
            )

        # Step 9: Anomaly detection
        if self.enable_anomaly_detection:
            logger.info("Step 9: Anomaly detection")
            ais_data, anomaly_report = self.anomaly_detector.detect_anomalies(ais_data)

            self.processing_report["pipeline_steps"].append(
                {"step": "anomaly_detection", "anomaly_report": anomaly_report}
            )

        # Step 10: Save processed data
        logger.info("Step 10: Saving processed data")
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

    def create_scene_dataset(
        self,
        ais_data: AISDataFrame,
        scene_config: dict,
        port_locations: list[dict] | None = None,
        waterway_definitions: list[dict] | None = None,
    ) -> tuple[SceneDataset, dict]:
        """Create a scene dataset for model training.

        Args:
            ais_data: Processed AIS data
            scene_config: Scene configuration parameters
            port_locations: Optional list of port locations
            waterway_definitions: Optional list of waterway definitions

        Returns:
            Tuple[AISSceneDataset, Dict]: Scene dataset and aggregation report
        """
        logger.info("Creating scene dataset")

        if not self.enable_scene_aggregation:
            raise ValueError("Scene aggregation is disabled")

        # Configure scene aggregator with additional parameters
        if port_locations:
            for port in port_locations:
                self.scene_aggregator.add_port_location(
                    name=port["name"],
                    latitude=port["latitude"],
                    longitude=port["longitude"],
                    radius_km=port.get("radius_km", 5.0),
                )

        if waterway_definitions:
            for waterway in waterway_definitions:
                # Convert polygon coordinates if provided
                if "polygon" in waterway:
                    from shapely.geometry import Polygon

                    polygon = Polygon(waterway["polygon"])
                    self.scene_aggregator.add_waterway_area(
                        name=waterway["name"],
                        polygon=polygon,
                        waterway_type=waterway.get("type", "channel"),
                    )

        # Generate scenes
        scenes = self.scene_aggregator.aggregate_scenes(ais_data.df)

        # Convert scenes to the format expected by SceneDataset
        scenes_data = []
        for scene in scenes:
            scene_dict = {
                "scene_id": scene.scene_id,
                "scene_type": scene.scene_type.value,
                "start_time": scene.start_time,
                "end_time": scene.end_time,
                "center_lat": scene.center_lat,
                "center_lon": scene.center_lon,
                "radius_km": scene.radius_km,
                "vessels": scene.vessels,
                "vessel_count": scene.vessel_count,
                "duration_minutes": scene.duration.total_seconds() / 60,
                "properties": scene.properties,
            }
            scenes_data.append(scene_dict)

        # Create dataset configuration
        dataset_config = DatasetConfig(
            history_length=scene_config.get("history_length", 10),
            future_length=scene_config.get("future_length", 5),
            time_step_seconds=scene_config.get("time_step_seconds", 60),
            min_vessel_count=scene_config.get("min_vessel_count", 2),
            max_vessel_count=scene_config.get("max_vessel_count", 50),
            edge_distance_threshold=scene_config.get("edge_distance_threshold", 2000.0),
            normalize_features=scene_config.get("normalize_features", True),
        )

        # Create dataset
        dataset = SceneDataset(
            scenes_data=scenes_data,
            vessel_trajectories=ais_data.df,
            config=dataset_config,
            cache_dir=scene_config.get("cache_dir"),
            precompute_features=scene_config.get("precompute_features", False),
        )

        # Metadata
        scene_stats = (
            self.scene_aggregator.get_scene_statistics(scenes) if scenes else {}
        )
        metadata = {
            "total_scenes": len(scenes),
            "scene_types": scene_stats.get("scene_types", {}),
            "time_range": {
                "start": str(ais_data.df["timestamp"].min()),
                "end": str(ais_data.df["timestamp"].max()),
            },
            "unique_vessels": ais_data.df["mmsi"].nunique(),
            "total_positions": len(ais_data.df),
            "config": scene_config,
            "dataset_size": len(dataset),
            "scene_statistics": scene_stats,
        }

        logger.info(f"Created scene dataset with {len(scenes)} scenes")
        return dataset, metadata

    def extract_voyage_segments(
        self,
        ais_data: AISDataFrame,
        mmsi: str | None = None,
    ) -> list[dict]:
        """Extract voyage segments from segmented trajectory data.

        Args:
            ais_data: Segmented AIS data
            mmsi: Specific vessel MMSI (optional)

        Returns:
            List of voyage segment dictionaries
        """
        if not self.enable_trajectory_segmentation:
            raise ValueError("Trajectory segmentation is disabled")

        # Get segmentation results for all vessels or specific MMSI
        if mmsi:
            vessel_df = ais_data.df[ais_data.df["mmsi"] == mmsi]
            if vessel_df.empty:
                return []
            segmentation_result = self.trajectory_segmenter.segment_trajectory(
                vessel_df
            )
            results = {mmsi: segmentation_result}
        else:
            results = self.trajectory_segmenter.segment_multiple_trajectories(
                ais_data.df
            )

        # Extract voyage segments
        voyage_segments = []
        for vessel_mmsi, result in results.items():
            for segment in result.segments:
                if segment.segment_type.value in ["voyage", "port_to_port", "transit"]:
                    segment_dict = {
                        "mmsi": vessel_mmsi,
                        "segment_id": segment.segment_id,
                        "segment_type": segment.segment_type.value,
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "duration_hours": segment.duration.total_seconds() / 3600,
                        "distance_km": segment.distance_km,
                        "avg_speed": segment.avg_speed,
                        "max_speed": segment.max_speed,
                        "points_count": len(segment.points),
                        "start_location": {
                            "latitude": segment.points[0].latitude
                            if segment.points
                            else None,
                            "longitude": segment.points[0].longitude
                            if segment.points
                            else None,
                        },
                        "end_location": {
                            "latitude": segment.points[-1].latitude
                            if segment.points
                            else None,
                            "longitude": segment.points[-1].longitude
                            if segment.points
                            else None,
                        },
                    }
                    voyage_segments.append(segment_dict)

        return voyage_segments

    def get_risk_assessment_summary(
        self,
        ais_data: AISDataFrame,
    ) -> dict:
        """Get risk assessment summary from processed data.

        Args:
            ais_data: Processed AIS data with risk information

        Returns:
            Risk assessment summary dictionary
        """
        if not self.enable_risk_assessment:
            raise ValueError("Risk assessment is disabled")

        # Calculate TCPA/DCPA for current data
        tcpa_results = self.tcpa_calculator.calculate_batch_tcpa_dcpa(ais_data.df)

        if not tcpa_results:
            return {"status": "no_risk_data", "message": "No risk encounters found"}

        # Get statistics from TCPA calculator
        risk_stats = self.tcpa_calculator.get_statistics(tcpa_results)

        # Get high-risk pairs
        high_risk_pairs = self.tcpa_calculator.get_high_risk_pairs(
            tcpa_results, risk_threshold=0.7
        )
        medium_risk_pairs = self.tcpa_calculator.get_high_risk_pairs(
            tcpa_results, risk_threshold=0.3
        )

        # Calculate additional summary metrics
        summary = {
            "status": "success",
            "total_calculations": len(tcpa_results),
            "high_risk_encounters": len(high_risk_pairs),
            "medium_risk_encounters": len(medium_risk_pairs) - len(high_risk_pairs),
            "low_risk_encounters": len(tcpa_results) - len(medium_risk_pairs),
            "unique_vessels_involved": len(
                set(
                    [r.vessel_1 for r in tcpa_results]
                    + [r.vessel_2 for r in tcpa_results]
                )
            ),
            "avg_dcpa_meters": risk_stats.get("dcpa_stats", {}).get("mean", 0),
            "min_dcpa_meters": risk_stats.get("dcpa_stats", {}).get("min", 0),
            "avg_tcpa_seconds": risk_stats.get("tcpa_stats", {}).get("mean", 0),
            "min_tcpa_seconds": risk_stats.get("tcpa_stats", {}).get("min", 0),
            "avg_risk_level": risk_stats.get("risk_stats", {}).get("mean", 0),
            "max_risk_level": risk_stats.get("risk_stats", {}).get("max", 0),
            "detailed_statistics": risk_stats,
        }

        return summary

    def interpolate_trajectory_gaps(
        self,
        ais_data: AISDataFrame,
        max_gap_seconds: int = 3600,
        interpolation_interval: int = 60,
    ) -> tuple[AISDataFrame, dict]:
        """Interpolate gaps in vessel trajectories.

        Args:
            ais_data: Input AIS data
            max_gap_seconds: Maximum gap to interpolate
            interpolation_interval: Interpolation interval in seconds

        Returns:
            Tuple[AISDataFrame, Dict]: Data with interpolated gaps and report
        """
        if not self.enable_enhanced_smoothing:
            raise ValueError("Enhanced smoothing is disabled")

        # Process trajectories for all vessels
        smoothing_results = self.trajectory_smoother.smooth_multiple_trajectories(
            ais_data.df
        )

        # Extract interpolation statistics
        total_gaps_filled = sum(
            result.gaps_filled for result in smoothing_results.values()
        )
        total_outliers_removed = sum(
            result.outliers_removed for result in smoothing_results.values()
        )

        # Convert smoothed results back to DataFrame
        smoothed_df = self.trajectory_smoother.export_smoothed_trajectories(
            smoothing_results
        )

        if not smoothed_df.empty:
            ais_data.df = smoothed_df

        # Generate report
        interpolation_report = {
            "vessels_processed": len(smoothing_results),
            "total_gaps_filled": total_gaps_filled,
            "total_outliers_removed": total_outliers_removed,
            "original_points": sum(
                result.original_length for result in smoothing_results.values()
            ),
            "final_points": sum(
                result.smoothed_length for result in smoothing_results.values()
            ),
            "processing_time": sum(
                result.processing_time for result in smoothing_results.values()
            ),
            "avg_quality_metrics": self.trajectory_smoother.get_smoothing_statistics(
                smoothing_results
            ),
        }

        return ais_data, interpolation_report

    def create_model_training_config(
        self,
        T_obs: int = 20,
        T_pred: int = 30,
        min_vessels: int = 2,
        max_vessels: int = 50,
        static_feature_dim: int = 10,
        t_safe: float = 1800.0,
        d_safe: float = 2.0,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> dict:
        """Create configuration for model training.

        Args:
            T_obs: Number of observation time steps
            T_pred: Number of prediction time steps
            min_vessels: Minimum vessels per scene
            max_vessels: Maximum vessels per scene
            static_feature_dim: Dimension of static features
            t_safe: Safe time threshold for TCPA
            d_safe: Safe distance threshold for DCPA
            alpha: DCPA weight in risk calculation
            beta: TCPA weight in risk calculation

        Returns:
            Configuration dictionary
        """
        return {
            "T_obs": T_obs,
            "T_pred": T_pred,
            "min_vessels": min_vessels,
            "max_vessels": max_vessels,
            "static_feature_dim": static_feature_dim,
            "t_safe": t_safe,
            "d_safe": d_safe,
            "alpha": alpha,
            "beta": beta,
            "dt": 60.0,  # Default time step in seconds
            "process_noise": 1.0,
            "measurement_noise": 10.0,
        }
