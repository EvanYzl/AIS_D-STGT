"""Anomaly Detection Module for AIS Data.

This module provides comprehensive anomaly detection capabilities for AIS data,
including kinematic anomalies, signal gaps, and autoencoder-based reconstruction
error detection.
"""

import logging

import numpy as np
import pandas as pd

# Optional scikit-learn imports
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover
    IsolationForest = None  # type: ignore
    StandardScaler = None  # type: ignore
    MLPRegressor = None  # type: ignore

# Only used for type hints; no runtime dependency if sklearn missing
try:
    from sklearn.metrics import mean_squared_error  # noqa: F401
except ImportError:  # pragma: no cover
    mean_squared_error = None  # type: ignore

from ..schemas import AISDataFrame

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Comprehensive anomaly detection for AIS data."""

    def __init__(
        self,
        max_speed_knots: float = 50.0,
        max_acceleration_ms2: float = 5.0,
        max_turn_rate_deg_s: float = 10.0,
        max_signal_gap_hours: float = 6.0,
        isolation_forest_contamination: float = 0.1,
        autoencoder_threshold: float = 0.5,
    ):
        """Initialize anomaly detector.

        Args:
            max_speed_knots: Maximum reasonable speed in knots
            max_acceleration_ms2: Maximum acceleration in m/s²
            max_turn_rate_deg_s: Maximum turn rate in degrees/second
            max_signal_gap_hours: Maximum acceptable signal gap in hours
            isolation_forest_contamination: Contamination parameter for Isolation Forest
            autoencoder_threshold: Threshold for autoencoder reconstruction error
        """
        self.max_speed_knots = max_speed_knots
        self.max_acceleration_ms2 = max_acceleration_ms2
        self.max_turn_rate_deg_s = max_turn_rate_deg_s
        self.max_signal_gap_hours = max_signal_gap_hours
        self.isolation_forest_contamination = isolation_forest_contamination
        self.autoencoder_threshold = autoencoder_threshold

        # Initialize models
        self.isolation_forest = None
        self.autoencoder = None
        # Initialize scaler only if sklearn is available
        self.scaler = StandardScaler() if StandardScaler is not None else None

    def detect_anomalies(self, ais_df: AISDataFrame) -> tuple[AISDataFrame, dict]:
        """Detect all types of anomalies in AIS data.

        Args:
            ais_df: Input AIS DataFrame

        Returns:
            Tuple[AISDataFrame, Dict]: DataFrame with anomaly flags and detection report
        """
        logger.info("Starting comprehensive anomaly detection")

        df = ais_df.df.copy()

        # Initialize anomaly flags
        df["anomaly_kinematic"] = False
        df["anomaly_signal_gap"] = False
        df["anomaly_isolation_forest"] = False
        df["anomaly_autoencoder"] = False
        df["anomaly_composite"] = False
        df["anomaly_score"] = 0.0

        detection_report = {
            "total_records": len(df),
            "anomaly_counts": {},
            "detection_methods": [],
        }

        # If scikit-learn is not available, skip advanced anomaly detection
        sklearn_available = (
            IsolationForest is not None
            and StandardScaler is not None
            and MLPRegressor is not None
        )

        # 1. Kinematic anomaly detection
        df, kinematic_report = self._detect_kinematic_anomalies(df)
        detection_report["anomaly_counts"]["kinematic"] = kinematic_report[
            "anomaly_count"
        ]
        detection_report["detection_methods"].append("kinematic")

        # 2. Signal gap detection
        df, gap_report = self._detect_signal_gaps(df)
        detection_report["anomaly_counts"]["signal_gaps"] = gap_report["anomaly_count"]
        detection_report["detection_methods"].append("signal_gaps")

        if sklearn_available:
            # 3. Isolation Forest anomaly detection
            df, isolation_report = self._detect_isolation_forest_anomalies(df)
            detection_report["anomaly_counts"]["isolation_forest"] = isolation_report[
                "anomaly_count"
            ]
            detection_report["detection_methods"].append("isolation_forest")

            # 4. Autoencoder-based anomaly detection
            df, autoencoder_report = self._detect_autoencoder_anomalies(df)
            detection_report["anomaly_counts"]["autoencoder"] = autoencoder_report[
                "anomaly_count"
            ]
            detection_report["detection_methods"].append("autoencoder")
        else:
            detection_report["anomaly_counts"]["isolation_forest"] = 0
            detection_report["anomaly_counts"]["autoencoder"] = 0

        # 5. Composite anomaly score
        df = self._calculate_composite_anomaly_score(df)

        # Calculate overall statistics
        total_anomalies = df["anomaly_composite"].sum()
        detection_report["total_anomalies"] = total_anomalies
        detection_report["anomaly_rate"] = total_anomalies / len(df)

        logger.info(
            f"Anomaly detection completed. Found {total_anomalies} anomalies "
            f"({detection_report['anomaly_rate']:.2%} of records)"
        )

        return AISDataFrame(df), detection_report

    def _detect_kinematic_anomalies(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """Detect kinematic anomalies (speed, acceleration, turn rate)."""
        logger.info("Detecting kinematic anomalies")

        anomaly_count = 0

        # Use projected coordinates if available
        if "X" in df.columns and "Y" in df.columns:
            x_col, y_col = "X", "Y"
        else:
            x_col, y_col = "LON", "LAT"

        # Process each vessel separately
        for mmsi in df["MMSI"].unique():
            mask = df["MMSI"] == mmsi
            vessel_data = df[mask].copy().sort_values("BaseDateTime")

            if len(vessel_data) < 3:
                continue

            # Calculate kinematic parameters
            vessel_data = self._calculate_kinematic_parameters(
                vessel_data, x_col, y_col
            )

            # Detect anomalies
            speed_anomalies = (
                vessel_data["calculated_speed_knots"] > self.max_speed_knots
            )
            accel_anomalies = (
                vessel_data["acceleration_ms2"].abs() > self.max_acceleration_ms2
            )
            turn_anomalies = (
                vessel_data["turn_rate_deg_s"].abs() > self.max_turn_rate_deg_s
            )

            # Combine anomalies
            kinematic_anomalies = speed_anomalies | accel_anomalies | turn_anomalies

            # Update main DataFrame
            df.loc[mask, "anomaly_kinematic"] = kinematic_anomalies
            anomaly_count += kinematic_anomalies.sum()

        report = {
            "method": "kinematic",
            "anomaly_count": anomaly_count,
            "thresholds": {
                "max_speed_knots": self.max_speed_knots,
                "max_acceleration_ms2": self.max_acceleration_ms2,
                "max_turn_rate_deg_s": self.max_turn_rate_deg_s,
            },
        }

        logger.info(f"Found {anomaly_count} kinematic anomalies")
        return df, report

    def _detect_signal_gaps(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Detect signal gaps (missing data periods)."""
        logger.info("Detecting signal gaps")

        anomaly_count = 0
        max_gap_seconds = self.max_signal_gap_hours * 3600

        # Process each vessel separately
        for mmsi in df["MMSI"].unique():
            mask = df["MMSI"] == mmsi
            vessel_data = df[mask].copy().sort_values("BaseDateTime")

            if len(vessel_data) < 2:
                continue

            # Calculate time differences
            time_diffs = vessel_data["BaseDateTime"].diff().dt.total_seconds()

            # Identify records after long gaps
            gap_anomalies = time_diffs > max_gap_seconds

            # Update main DataFrame
            df.loc[mask, "anomaly_signal_gap"] = gap_anomalies
            anomaly_count += gap_anomalies.sum()

        report = {
            "method": "signal_gaps",
            "anomaly_count": anomaly_count,
            "threshold_hours": self.max_signal_gap_hours,
        }

        logger.info(f"Found {anomaly_count} signal gap anomalies")
        return df, report

    def _detect_isolation_forest_anomalies(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """Detect anomalies using Isolation Forest."""
        logger.info("Detecting anomalies using Isolation Forest")

        # If scikit-learn is not available, return early
        if IsolationForest is None:
            return df, {
                "method": "isolation_forest",
                "anomaly_count": 0,
                "status": "skipped_no_sklearn",
            }

        # Select features for anomaly detection
        feature_cols = ["SOG", "COG"]

        # Add calculated features if available
        if "calculated_speed_knots" in df.columns:
            feature_cols.extend(
                ["calculated_speed_knots", "acceleration_ms2", "turn_rate_deg_s"]
            )

        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]

        if len(available_cols) < 2:
            logger.warning("Insufficient features for Isolation Forest, skipping")
            return df, {
                "method": "isolation_forest",
                "anomaly_count": 0,
                "status": "skipped",
            }

        # Prepare features
        features = df[available_cols].fillna(0)

        # Fit Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.isolation_forest_contamination, random_state=42
        )

        # Predict anomalies
        anomaly_predictions = self.isolation_forest.fit_predict(features)
        anomaly_scores = self.isolation_forest.score_samples(features)

        # Convert to boolean flags (Isolation Forest returns -1 for anomalies, 1 for normal)
        df["anomaly_isolation_forest"] = anomaly_predictions == -1
        df["isolation_forest_score"] = anomaly_scores

        anomaly_count = (anomaly_predictions == -1).sum()

        report = {
            "method": "isolation_forest",
            "anomaly_count": anomaly_count,
            "features_used": available_cols,
            "contamination": self.isolation_forest_contamination,
        }

        logger.info(f"Found {anomaly_count} Isolation Forest anomalies")
        return df, report

    def _detect_autoencoder_anomalies(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """Detect anomalies using autoencoder reconstruction error."""
        logger.info("Detecting anomalies using autoencoder")

        # If scikit-learn is not available, return early
        if StandardScaler is None or MLPRegressor is None:
            return df, {
                "method": "autoencoder",
                "anomaly_count": 0,
                "status": "skipped_no_sklearn",
            }

        # Select features for autoencoder
        feature_cols = ["SOG", "COG", "LAT", "LON"]

        # Add calculated features if available
        if "calculated_speed_knots" in df.columns:
            feature_cols.extend(["calculated_speed_knots"])

        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]

        if len(available_cols) < 3:
            logger.warning("Insufficient features for autoencoder, skipping")
            return df, {
                "method": "autoencoder",
                "anomaly_count": 0,
                "status": "skipped",
            }

        # Prepare features
        features = df[available_cols].fillna(0)

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Simple autoencoder using MLPRegressor
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=(features_scaled.shape[1] * 2, features_scaled.shape[1]),
            max_iter=200,
            random_state=42,
        )

        # Train autoencoder
        self.autoencoder.fit(features_scaled, features_scaled)

        # Get reconstructions
        reconstructions = self.autoencoder.predict(features_scaled)

        # Calculate reconstruction errors
        reconstruction_errors = np.mean(
            (features_scaled - reconstructions) ** 2, axis=1
        )

        # Detect anomalies based on reconstruction error threshold
        anomaly_threshold = np.percentile(
            reconstruction_errors, 95
        )  # Top 5% as anomalies
        anomaly_flags = reconstruction_errors > anomaly_threshold

        df["anomaly_autoencoder"] = anomaly_flags
        df["autoencoder_error"] = reconstruction_errors

        anomaly_count = anomaly_flags.sum()

        report = {
            "method": "autoencoder",
            "anomaly_count": anomaly_count,
            "features_used": available_cols,
            "threshold": anomaly_threshold,
            "mean_reconstruction_error": np.mean(reconstruction_errors),
        }

        logger.info(f"Found {anomaly_count} autoencoder anomalies")
        return df, report

    def _calculate_kinematic_parameters(
        self, vessel_data: pd.DataFrame, x_col: str, y_col: str
    ) -> pd.DataFrame:
        """Calculate kinematic parameters for a vessel."""
        vessel_data = vessel_data.copy()

        # Calculate distances and time differences
        x_diff = vessel_data[x_col].diff()
        y_diff = vessel_data[y_col].diff()
        distances = np.sqrt(x_diff**2 + y_diff**2)
        time_diffs = vessel_data["BaseDateTime"].diff().dt.total_seconds()

        # Calculate speeds
        speeds_ms = distances / time_diffs
        speeds_knots = speeds_ms * 1.94384  # m/s to knots
        vessel_data["calculated_speed_knots"] = speeds_knots.fillna(0)

        # Calculate accelerations
        speed_diff = speeds_ms.diff()
        accelerations = speed_diff / time_diffs
        vessel_data["acceleration_ms2"] = accelerations.fillna(0)

        # Calculate turn rates
        cog_diff = vessel_data["COG"].diff()
        # Handle course wrap-around
        cog_diff = np.where(cog_diff > 180, cog_diff - 360, cog_diff)
        cog_diff = np.where(cog_diff < -180, cog_diff + 360, cog_diff)
        turn_rates = cog_diff / time_diffs
        vessel_data["turn_rate_deg_s"] = turn_rates.fillna(0)

        return vessel_data

    def _calculate_composite_anomaly_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite anomaly score and flag."""
        # Weight different anomaly types
        weights = {
            "kinematic": 0.3,
            "signal_gap": 0.2,
            "isolation_forest": 0.25,
            "autoencoder": 0.25,
        }

        # Calculate weighted score
        anomaly_score = (
            df["anomaly_kinematic"].astype(float) * weights["kinematic"]
            + df["anomaly_signal_gap"].astype(float) * weights["signal_gap"]
            + df["anomaly_isolation_forest"].astype(float) * weights["isolation_forest"]
            + df["anomaly_autoencoder"].astype(float) * weights["autoencoder"]
        )

        df["anomaly_score"] = anomaly_score

        # Flag as anomaly if score exceeds threshold
        df["anomaly_composite"] = anomaly_score > 0.3

        return df

    def get_anomaly_report(self, ais_df: AISDataFrame) -> dict:
        """Generate comprehensive anomaly detection report.

        Args:
            ais_df: AIS DataFrame with anomaly flags

        Returns:
            Dict: Anomaly detection report
        """
        df = ais_df.df

        # Check if anomaly detection has been performed
        anomaly_cols = [col for col in df.columns if col.startswith("anomaly_")]
        if not anomaly_cols:
            return {"status": "No anomaly detection performed"}

        # Calculate anomaly statistics
        anomaly_stats = {}
        for col in anomaly_cols:
            if col in df.columns:
                anomaly_stats[col] = {
                    "count": df[col].sum(),
                    "percentage": df[col].mean() * 100,
                }

        # Vessel-level anomaly statistics
        vessel_anomaly_stats = {}
        for mmsi in df["MMSI"].unique():
            vessel_data = df[df["MMSI"] == mmsi]
            vessel_anomaly_stats[mmsi] = {
                "total_records": len(vessel_data),
                "anomaly_count": vessel_data["anomaly_composite"].sum(),
                "anomaly_rate": vessel_data["anomaly_composite"].mean(),
            }

        # Top anomalous vessels
        top_anomalous = sorted(
            vessel_anomaly_stats.items(),
            key=lambda x: x[1]["anomaly_rate"],
            reverse=True,
        )[:10]

        return {
            "total_records": len(df),
            "anomaly_statistics": anomaly_stats,
            "vessel_anomaly_stats": vessel_anomaly_stats,
            "top_anomalous_vessels": top_anomalous,
            "overall_anomaly_rate": df["anomaly_composite"].mean(),
        }
