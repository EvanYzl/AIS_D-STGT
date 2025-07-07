"""Kalman Filter Module for AIS Trajectory Smoothing.

This module implements Kalman filtering for smoothing AIS vessel trajectories,
removing noise and interpolating missing positions.
"""

import logging

import numpy as np
import pandas as pd

# Optional filterpy imports
try:
    from filterpy.common import Q_discrete_white_noise
    from filterpy.kalman import KalmanFilter as FilterPyKalman
except ImportError:  # pragma: no cover
    FilterPyKalman = None  # type: ignore

    def Q_discrete_white_noise(*args, **kwargs):  # type: ignore
        raise NotImplementedError(
            "filterpy is required for Kalman filtering but not installed."
        )


from ..schemas import AISDataFrame

logger = logging.getLogger(__name__)


class KalmanFilter:
    """Kalman filter for AIS trajectory smoothing."""

    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
        initial_velocity_uncertainty: float = 10.0,
        initial_position_uncertainty: float = 100.0,
    ):
        """Initialize Kalman filter parameters.

        Args:
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
            initial_velocity_uncertainty: Initial velocity uncertainty
            initial_position_uncertainty: Initial position uncertainty
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initial_velocity_uncertainty = initial_velocity_uncertainty
        self.initial_position_uncertainty = initial_position_uncertainty

    def filter_trajectory(self, ais_df: AISDataFrame) -> AISDataFrame:
        """Apply Kalman filtering to vessel trajectories.

        Args:
            ais_df: Input AIS DataFrame

        Returns:
            AISDataFrame: DataFrame with smoothed trajectories
        """
        if FilterPyKalman is None:
            logger.warning("filterpy not installed; skipping Kalman filtering.")
            return ais_df

        df = ais_df.df.copy()

        # Ensure data is sorted by MMSI and time
        df = df.sort_values(["MMSI", "BaseDateTime"])

        # Use projected coordinates if available
        if "X" in df.columns and "Y" in df.columns:
            x_col, y_col = "X", "Y"
        else:
            x_col, y_col = "LON", "LAT"

        # Add filtered coordinate columns
        df[f"{x_col}_filtered"] = df[x_col]
        df[f"{y_col}_filtered"] = df[y_col]
        df["velocity_x_filtered"] = 0.0
        df["velocity_y_filtered"] = 0.0
        df["filter_quality"] = 0.0

        # Process each vessel separately
        for mmsi in df["MMSI"].unique():
            mask = df["MMSI"] == mmsi
            vessel_data = df[mask].copy()

            if len(vessel_data) < 3:
                logger.warning(
                    f"Insufficient data for vessel {mmsi}, skipping filtering"
                )
                continue

            try:
                # Apply Kalman filter to this vessel
                filtered_data = self._filter_vessel_trajectory(
                    vessel_data, x_col, y_col
                )

                # Update the main DataFrame
                df.loc[mask, f"{x_col}_filtered"] = filtered_data[f"{x_col}_filtered"]
                df.loc[mask, f"{y_col}_filtered"] = filtered_data[f"{y_col}_filtered"]
                df.loc[mask, "velocity_x_filtered"] = filtered_data[
                    "velocity_x_filtered"
                ]
                df.loc[mask, "velocity_y_filtered"] = filtered_data[
                    "velocity_y_filtered"
                ]
                df.loc[mask, "filter_quality"] = filtered_data["filter_quality"]

            except Exception as e:
                logger.warning(f"Failed to filter trajectory for vessel {mmsi}: {e}")
                continue

        logger.info("Kalman filtering completed for all vessels")
        return AISDataFrame(df)

    def _filter_vessel_trajectory(
        self, vessel_data: pd.DataFrame, x_col: str, y_col: str
    ) -> pd.DataFrame:
        """Apply Kalman filter to a single vessel's trajectory.

        Args:
            vessel_data: DataFrame for single vessel
            x_col: X coordinate column name
            y_col: Y coordinate column name

        Returns:
            pd.DataFrame: Filtered trajectory data
        """
        if FilterPyKalman is None:
            raise ImportError("filterpy not installed")

        # Create Kalman filter
        kf = FilterPyKalman(
            dim_x=4, dim_z=2
        )  # 4 states (x, y, vx, vy), 2 measurements (x, y)

        # State transition matrix (constant velocity model)
        dt = 1.0  # Will be updated with actual time differences
        kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Measurement function (we observe position)
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Measurement noise
        kf.R = np.eye(2) * self.measurement_noise

        # Process noise
        kf.Q = Q_discrete_white_noise(
            dim=2, dt=dt, var=self.process_noise, block_size=2
        )

        # Initial state covariance
        kf.P = np.eye(4) * 1000
        kf.P[0, 0] = self.initial_position_uncertainty
        kf.P[1, 1] = self.initial_position_uncertainty
        kf.P[2, 2] = self.initial_velocity_uncertainty
        kf.P[3, 3] = self.initial_velocity_uncertainty

        # Initialize state with first position and zero velocity
        first_pos = vessel_data.iloc[0]
        kf.x = np.array([first_pos[x_col], first_pos[y_col], 0.0, 0.0])

        # Prepare output arrays
        filtered_x = []
        filtered_y = []
        filtered_vx = []
        filtered_vy = []
        filter_quality = []

        # Process each measurement
        for i, row in vessel_data.iterrows():
            # Calculate time difference
            if i > 0:
                prev_time = vessel_data.iloc[vessel_data.index.get_loc(i) - 1][
                    "BaseDateTime"
                ]
                current_time = row["BaseDateTime"]
                dt = (current_time - prev_time).total_seconds()

                # Update state transition matrix with actual dt
                kf.F[0, 2] = dt
                kf.F[1, 3] = dt

                # Update process noise with actual dt
                kf.Q = Q_discrete_white_noise(
                    dim=2, dt=dt, var=self.process_noise, block_size=2
                )

            # Predict step
            kf.predict()

            # Update step with measurement
            measurement = np.array([row[x_col], row[y_col]])
            kf.update(measurement)

            # Store filtered results
            filtered_x.append(kf.x[0])
            filtered_y.append(kf.x[1])
            filtered_vx.append(kf.x[2])
            filtered_vy.append(kf.x[3])

            # Calculate filter quality (inverse of trace of covariance matrix)
            quality = 1.0 / (np.trace(kf.P) + 1e-6)
            filter_quality.append(quality)

        # Create result DataFrame
        result = vessel_data.copy()
        result[f"{x_col}_filtered"] = filtered_x
        result[f"{y_col}_filtered"] = filtered_y
        result["velocity_x_filtered"] = filtered_vx
        result["velocity_y_filtered"] = filtered_vy
        result["filter_quality"] = filter_quality

        return result

    def interpolate_missing_positions(
        self,
        ais_df: AISDataFrame,
        max_gap_seconds: int = 3600,
        interpolation_interval: int = 60,
    ) -> AISDataFrame:
        """Interpolate missing positions using Kalman filter predictions.

        Args:
            ais_df: Input AIS DataFrame
            max_gap_seconds: Maximum gap to interpolate (in seconds)
            interpolation_interval: Interpolation interval (in seconds)

        Returns:
            AISDataFrame: DataFrame with interpolated positions
        """
        if FilterPyKalman is None:
            logger.warning("filterpy not installed; skipping Kalman filtering.")
            return ais_df

        df = ais_df.df.copy()

        # Use projected coordinates if available
        if "X" in df.columns and "Y" in df.columns:
            x_col, y_col = "X", "Y"
        else:
            x_col, y_col = "LON", "LAT"

        interpolated_records = []

        # Process each vessel separately
        for mmsi in df["MMSI"].unique():
            mask = df["MMSI"] == mmsi
            vessel_data = df[mask].copy().sort_values("BaseDateTime")

            if len(vessel_data) < 2:
                continue

            # Find gaps in the trajectory
            time_diffs = vessel_data["BaseDateTime"].diff().dt.total_seconds()
            gap_indices = time_diffs > max_gap_seconds

            if not gap_indices.any():
                continue

            # Interpolate each gap
            for i in range(1, len(vessel_data)):
                if gap_indices.iloc[i]:
                    prev_record = vessel_data.iloc[i - 1]
                    curr_record = vessel_data.iloc[i]

                    # Calculate number of interpolation points
                    gap_duration = time_diffs.iloc[i]
                    num_points = int(gap_duration / interpolation_interval)

                    if num_points > 0:
                        interpolated = self._interpolate_gap(
                            prev_record,
                            curr_record,
                            num_points,
                            x_col,
                            y_col,
                            interpolation_interval,
                        )
                        interpolated_records.extend(interpolated)

        # Add interpolated records to the DataFrame
        if interpolated_records:
            interpolated_df = pd.DataFrame(interpolated_records)
            df = pd.concat([df, interpolated_df], ignore_index=True)
            df = df.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)

        logger.info(f"Interpolated {len(interpolated_records)} missing positions")
        return AISDataFrame(df)

    def _interpolate_gap(
        self,
        prev_record: pd.Series,
        curr_record: pd.Series,
        num_points: int,
        x_col: str,
        y_col: str,
        interval: int,
    ) -> list[dict]:
        """Interpolate positions between two records.

        Args:
            prev_record: Previous record
            curr_record: Current record
            num_points: Number of interpolation points
            x_col: X coordinate column name
            y_col: Y coordinate column name
            interval: Time interval between points

        Returns:
            List[Dict]: Interpolated records
        """
        interpolated = []

        # Calculate velocity
        time_diff = (
            curr_record["BaseDateTime"] - prev_record["BaseDateTime"]
        ).total_seconds()
        vx = (curr_record[x_col] - prev_record[x_col]) / time_diff
        vy = (curr_record[y_col] - prev_record[y_col]) / time_diff

        # Generate interpolated points
        for i in range(1, num_points + 1):
            dt = i * interval

            # Linear interpolation with constant velocity
            x_interp = prev_record[x_col] + vx * dt
            y_interp = prev_record[y_col] + vy * dt

            # Create interpolated record
            interp_record = prev_record.copy()
            interp_record["BaseDateTime"] = prev_record["BaseDateTime"] + pd.Timedelta(
                seconds=dt
            )
            interp_record[x_col] = x_interp
            interp_record[y_col] = y_interp

            # Mark as interpolated
            interp_record["interpolated"] = True

            interpolated.append(interp_record.to_dict())

        return interpolated

    def get_filtering_quality_report(self, ais_df: AISDataFrame) -> dict:
        """Generate quality report for Kalman filtering results.

        Args:
            ais_df: Filtered AIS DataFrame

        Returns:
            Dict: Quality report
        """
        if FilterPyKalman is None:
            logger.warning("filterpy not installed; skipping quality report.")
            return {"status": "No filtering applied"}

        df = ais_df.df

        # Check if filtering has been applied
        if "filter_quality" not in df.columns:
            return {"status": "No filtering applied"}

        # Calculate quality metrics
        quality_stats = df["filter_quality"].describe().to_dict()

        # Calculate smoothing effectiveness
        if "X" in df.columns and "X_filtered" in df.columns:
            x_col, y_col = "X", "Y"
        else:
            x_col, y_col = "LON", "LAT"

        smoothing_metrics = {}

        for mmsi in df["MMSI"].unique():
            vessel_data = df[df["MMSI"] == mmsi]

            if len(vessel_data) < 3:
                continue

            # Calculate position differences
            orig_x_diff = vessel_data[x_col].diff().abs()
            filt_x_diff = vessel_data[f"{x_col}_filtered"].diff().abs()

            orig_y_diff = vessel_data[y_col].diff().abs()
            filt_y_diff = vessel_data[f"{y_col}_filtered"].diff().abs()

            # Calculate smoothing factor
            smoothing_x = orig_x_diff.std() / (filt_x_diff.std() + 1e-6)
            smoothing_y = orig_y_diff.std() / (filt_y_diff.std() + 1e-6)

            smoothing_metrics[mmsi] = {
                "smoothing_factor_x": smoothing_x,
                "smoothing_factor_y": smoothing_y,
                "avg_quality": vessel_data["filter_quality"].mean(),
            }

        return {
            "total_vessels_filtered": len(smoothing_metrics),
            "quality_statistics": quality_stats,
            "smoothing_metrics": smoothing_metrics,
            "avg_smoothing_factor": np.mean(
                [
                    (m["smoothing_factor_x"] + m["smoothing_factor_y"]) / 2
                    for m in smoothing_metrics.values()
                ]
            ),
        }
