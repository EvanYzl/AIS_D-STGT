"""Data Cleaning Module for AIS Data.

This module provides comprehensive data cleaning capabilities including:
- Missing value handling
- Duplicate removal
- Data type validation and conversion
- Outlier detection and removal
- Data consistency checks
"""

import logging

import pandas as pd

from ..schemas import AISDataFrame

logger = logging.getLogger(__name__)


class DataCleaner:
    """Comprehensive data cleaning for AIS datasets."""

    def __init__(
        self,
        remove_duplicates: bool = True,
        handle_missing_values: bool = True,
        validate_ranges: bool = True,
        remove_outliers: bool = True,
    ):
        """Initialize data cleaner.

        Args:
            remove_duplicates: Whether to remove duplicate records
            handle_missing_values: Whether to handle missing values
            validate_ranges: Whether to validate data ranges
            remove_outliers: Whether to remove statistical outliers
        """
        self.remove_duplicates = remove_duplicates
        self.handle_missing_values = handle_missing_values
        self.validate_ranges = validate_ranges
        self.remove_outliers = remove_outliers

        # Define valid ranges for AIS data
        self.valid_ranges = {
            "LAT": (-90.0, 90.0),
            "LON": (-180.0, 180.0),
            "SOG": (0.0, 102.3),  # Max speed in knots
            "COG": (0.0, 360.0),
            "Heading": (0.0, 360.0),
            "Length": (0.0, 400.0),  # Max vessel length in meters
            "Width": (0.0, 63.0),  # Max vessel width in meters
            "Draft": (0.0, 25.5),  # Max draft in meters
        }

    def clean_data(self, ais_df: AISDataFrame) -> tuple[AISDataFrame, dict]:
        """Perform comprehensive data cleaning.

        Args:
            ais_df: Input AIS DataFrame

        Returns:
            Tuple[AISDataFrame, Dict]: Cleaned data and cleaning report
        """
        logger.info("Starting data cleaning process")

        df = ais_df.df.copy()
        initial_records = len(df)

        cleaning_report = {"initial_records": initial_records, "steps": []}

        # Step 1: Remove duplicates
        if self.remove_duplicates:
            df, step_report = self._remove_duplicates(df)
            cleaning_report["steps"].append(step_report)

        # Step 2: Handle missing values
        if self.handle_missing_values:
            df, step_report = self._handle_missing_values(df)
            cleaning_report["steps"].append(step_report)

        # Step 3: Validate ranges
        if self.validate_ranges:
            df, step_report = self._validate_ranges(df)
            cleaning_report["steps"].append(step_report)

        # Step 4: Remove outliers
        if self.remove_outliers:
            df, step_report = self._remove_outliers(df)
            cleaning_report["steps"].append(step_report)

        # Step 5: Sort data
        df = self._sort_data(df)

        cleaning_report["final_records"] = len(df)
        cleaning_report["records_removed"] = initial_records - len(df)
        cleaning_report["removal_rate"] = (
            cleaning_report["records_removed"] / initial_records
        )

        logger.info(
            f"Data cleaning completed. Removed {cleaning_report['records_removed']} "
            f"records ({cleaning_report['removal_rate']:.2%})"
        )

        return AISDataFrame(df), cleaning_report

    def _remove_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Remove duplicate records."""
        initial_count = len(df)

        # Define subset for duplicate detection (key fields)
        duplicate_subset = ["MMSI", "BaseDateTime", "LAT", "LON"]

        # Remove duplicates
        df_clean = df.drop_duplicates(subset=duplicate_subset, keep="first")

        removed_count = initial_count - len(df_clean)

        report = {
            "step": "remove_duplicates",
            "initial_count": initial_count,
            "final_count": len(df_clean),
            "removed_count": removed_count,
            "duplicate_subset": duplicate_subset,
        }

        logger.info(f"Removed {removed_count} duplicate records")
        return df_clean, report

    def _handle_missing_values(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Handle missing values in the dataset."""
        initial_count = len(df)

        # Check missing values before cleaning
        missing_before = df.isnull().sum().to_dict()

        # Remove records with missing critical fields
        critical_fields = ["MMSI", "BaseDateTime", "LAT", "LON"]
        df_clean = df.dropna(subset=critical_fields)

        # Handle missing values in non-critical fields
        # For SOG and COG, fill with 0 if missing
        if "SOG" in df_clean.columns:
            df_clean["SOG"] = df_clean["SOG"].fillna(0.0)

        if "COG" in df_clean.columns:
            df_clean["COG"] = df_clean["COG"].fillna(0.0)

        # For Heading, use special value 511 to indicate "not available"
        if "Heading" in df_clean.columns:
            df_clean["Heading"] = df_clean["Heading"].fillna(511.0)

        # Check missing values after cleaning
        missing_after = df_clean.isnull().sum().to_dict()

        removed_count = initial_count - len(df_clean)

        report = {
            "step": "handle_missing_values",
            "initial_count": initial_count,
            "final_count": len(df_clean),
            "removed_count": removed_count,
            "missing_before": missing_before,
            "missing_after": missing_after,
            "critical_fields": critical_fields,
        }

        logger.info(
            f"Handled missing values, removed {removed_count} records with missing critical fields"
        )
        return df_clean, report

    def _validate_ranges(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Validate data ranges and remove invalid records."""
        initial_count = len(df)

        invalid_records = {}
        mask = pd.Series(True, index=df.index)

        for column, (min_val, max_val) in self.valid_ranges.items():
            if column in df.columns:
                # Special handling for Heading (511 = not available)
                if column == "Heading":
                    invalid_mask = (df[column] < min_val) | (df[column] >= max_val)
                    # Allow 511 as valid "not available" value
                    invalid_mask = invalid_mask & (df[column] != 511.0)
                else:
                    invalid_mask = (df[column] < min_val) | (df[column] >= max_val)

                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    invalid_records[column] = invalid_count
                    mask = mask & ~invalid_mask

        df_clean = df[mask]
        removed_count = initial_count - len(df_clean)

        report = {
            "step": "validate_ranges",
            "initial_count": initial_count,
            "final_count": len(df_clean),
            "removed_count": removed_count,
            "invalid_records_by_field": invalid_records,
            "valid_ranges": self.valid_ranges,
        }

        logger.info(f"Range validation removed {removed_count} records")
        return df_clean, report

    def _remove_outliers(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Remove statistical outliers using IQR method."""
        initial_count = len(df)

        # Fields to check for outliers
        outlier_fields = ["SOG", "COG"]

        outlier_counts = {}
        mask = pd.Series(True, index=df.index)

        for field in outlier_fields:
            if field in df.columns:
                # Calculate IQR
                Q1 = df[field].quantile(0.25)
                Q3 = df[field].quantile(0.75)
                IQR = Q3 - Q1

                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Identify outliers
                outlier_mask = (df[field] < lower_bound) | (df[field] > upper_bound)
                outlier_count = outlier_mask.sum()

                if outlier_count > 0:
                    outlier_counts[field] = {
                        "count": outlier_count,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "Q1": Q1,
                        "Q3": Q3,
                        "IQR": IQR,
                    }
                    mask = mask & ~outlier_mask

        df_clean = df[mask]
        removed_count = initial_count - len(df_clean)

        report = {
            "step": "remove_outliers",
            "initial_count": initial_count,
            "final_count": len(df_clean),
            "removed_count": removed_count,
            "outlier_details": outlier_counts,
        }

        logger.info(f"Outlier removal removed {removed_count} records")
        return df_clean, report

    def _sort_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort data by MMSI and BaseDateTime."""
        return df.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)

    def get_data_quality_report(self, ais_df: AISDataFrame) -> dict:
        """Generate comprehensive data quality report.

        Args:
            ais_df: Input AIS DataFrame

        Returns:
            Dict: Data quality report
        """
        df = ais_df.df

        # Basic statistics
        total_records = len(df)
        unique_vessels = df["MMSI"].nunique()

        # Time range
        time_range = {
            "start": df["BaseDateTime"].min(),
            "end": df["BaseDateTime"].max(),
            "duration_hours": (
                df["BaseDateTime"].max() - df["BaseDateTime"].min()
            ).total_seconds()
            / 3600,
        }

        # Geographic coverage
        geo_bounds = {
            "lat_min": df["LAT"].min(),
            "lat_max": df["LAT"].max(),
            "lon_min": df["LON"].min(),
            "lon_max": df["LON"].max(),
        }

        # Missing values
        missing_values = df.isnull().sum().to_dict()

        # Data completeness
        completeness = {}
        for col in df.columns:
            completeness[col] = 1 - (df[col].isnull().sum() / total_records)

        # Speed and course statistics
        speed_stats = df["SOG"].describe().to_dict() if "SOG" in df.columns else {}
        course_stats = df["COG"].describe().to_dict() if "COG" in df.columns else {}

        return {
            "total_records": total_records,
            "unique_vessels": unique_vessels,
            "time_range": time_range,
            "geographic_bounds": geo_bounds,
            "missing_values": missing_values,
            "data_completeness": completeness,
            "speed_statistics": speed_stats,
            "course_statistics": course_stats,
        }
