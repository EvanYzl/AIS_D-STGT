"""
Enhanced Trajectory Smoother Module

This module provides advanced trajectory smoothing functionality:
- Enhanced Kalman filtering with local coordinate projection
- Rauch-Tung-Striebel (RTS) smoothing
- Gap interpolation using Kalman filter predictions
- Quality metrics and uncertainty estimation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from geopy.distance import geodesic
from scipy.linalg import inv, cholesky
from scipy.interpolate import interp1d
import warnings

logger = logging.getLogger(__name__)


@dataclass
class SmoothedPoint:
    """Represents a smoothed trajectory point"""
    timestamp: pd.Timestamp
    latitude: float
    longitude: float
    speed: float
    course: float
    x: float  # local coordinate
    y: float  # local coordinate
    vx: float  # velocity x component
    vy: float  # velocity y component
    uncertainty_x: float  # position uncertainty in x
    uncertainty_y: float  # position uncertainty in y
    uncertainty_vx: float  # velocity uncertainty in x
    uncertainty_vy: float  # velocity uncertainty in y
    quality_score: float  # 0-1 quality score
    is_interpolated: bool = False
    original_index: Optional[int] = None


@dataclass
class SmoothingResult:
    """Result of trajectory smoothing"""
    mmsi: int
    smoothed_points: List[SmoothedPoint]
    original_length: int
    smoothed_length: int
    gaps_filled: int
    outliers_removed: int
    quality_metrics: Dict[str, float]
    processing_time: float


class EnhancedTrajectorySmootherRTS:
    """
    Enhanced trajectory smoother using Kalman filtering and RTS smoothing.
    Includes gap interpolation and quality assessment.
    """
    
    def __init__(self,
                 process_noise_std: float = 1.0,  # Process noise standard deviation (m/s²)
                 observation_noise_std: float = 10.0,  # Observation noise standard deviation (m)
                 max_gap_duration: float = 3600.0,  # Maximum gap to interpolate (seconds)
                 min_speed_threshold: float = 0.1,  # Minimum speed for motion detection (knots)
                 max_speed_threshold: float = 50.0,  # Maximum reasonable speed (knots)
                 max_acceleration: float = 2.0,  # Maximum acceleration (m/s²)
                 outlier_threshold: float = 3.0,  # Standard deviations for outlier detection
                 projection_center: Optional[Tuple[float, float]] = None):
        
        self.process_noise_std = process_noise_std
        self.observation_noise_std = observation_noise_std
        self.max_gap_duration = max_gap_duration
        self.min_speed_threshold = min_speed_threshold
        self.max_speed_threshold = max_speed_threshold
        self.max_acceleration = max_acceleration
        self.outlier_threshold = outlier_threshold
        self.projection_center = projection_center
        
        # Kalman filter matrices
        self.dt = 1.0  # Default time step (will be updated dynamically)
        self._initialize_kalman_matrices()
    
    def _initialize_kalman_matrices(self):
        """Initialize Kalman filter matrices"""
        # State vector: [x, y, vx, vy]
        # Transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Observation matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        q = self.process_noise_std ** 2
        self.Q = np.array([
            [q * self.dt**4 / 4, 0, q * self.dt**3 / 2, 0],
            [0, q * self.dt**4 / 4, 0, q * self.dt**3 / 2],
            [q * self.dt**3 / 2, 0, q * self.dt**2, 0],
            [0, q * self.dt**3 / 2, 0, q * self.dt**2]
        ])
        
        # Observation noise covariance
        r = self.observation_noise_std ** 2
        self.R = np.array([
            [r, 0],
            [0, r]
        ])
    
    def smooth_trajectory(self, df: pd.DataFrame) -> SmoothingResult:
        """
        Smooth a trajectory using enhanced Kalman filtering and RTS smoothing.
        
        Args:
            df: DataFrame with trajectory data for a single vessel
            
        Returns:
            SmoothingResult with smoothed trajectory
        """
        start_time = pd.Timestamp.now()
        
        if len(df) < 3:
            logger.warning(f"Insufficient data points for smoothing: {len(df)}")
            return self._create_empty_result(df.iloc[0]['mmsi'] if len(df) > 0 else 0)
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Project to local coordinates
        projected_data = self._project_to_local_coordinates(df_sorted)
        
        # Detect and remove outliers
        cleaned_data, outliers_removed = self._detect_and_remove_outliers(projected_data)
        
        # Fill gaps with interpolation
        filled_data, gaps_filled = self._fill_gaps(cleaned_data)
        
        # Apply Kalman filtering
        filtered_states, filtered_covariances = self._apply_kalman_filter(filled_data)
        
        # Apply RTS smoothing
        smoothed_states, smoothed_covariances = self._apply_rts_smoothing(
            filtered_states, filtered_covariances, filled_data
        )
        
        # Convert back to geographical coordinates
        smoothed_points = self._convert_to_smoothed_points(
            smoothed_states, smoothed_covariances, filled_data
        )
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            df_sorted, smoothed_points, outliers_removed, gaps_filled
        )
        
        processing_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        result = SmoothingResult(
            mmsi=df_sorted.iloc[0]['mmsi'],
            smoothed_points=smoothed_points,
            original_length=len(df_sorted),
            smoothed_length=len(smoothed_points),
            gaps_filled=gaps_filled,
            outliers_removed=outliers_removed,
            quality_metrics=quality_metrics,
            processing_time=processing_time
        )
        
        logger.info(f"Smoothed trajectory for MMSI {result.mmsi}: "
                   f"{result.original_length} -> {result.smoothed_length} points, "
                   f"{gaps_filled} gaps filled, {outliers_removed} outliers removed")
        
        return result
    
    def _project_to_local_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Project geographical coordinates to local Cartesian system"""
        df_proj = df.copy()
        
        # Determine projection center
        if self.projection_center is None:
            center_lat = df['latitude'].mean()
            center_lon = df['longitude'].mean()
        else:
            center_lat, center_lon = self.projection_center
        
        # Project coordinates
        x_coords = []
        y_coords = []
        
        for _, row in df.iterrows():
            x, y = self._project_point(row['latitude'], row['longitude'], 
                                     center_lat, center_lon)
            x_coords.append(x)
            y_coords.append(y)
        
        df_proj['x'] = x_coords
        df_proj['y'] = y_coords
        df_proj['center_lat'] = center_lat
        df_proj['center_lon'] = center_lon
        
        return df_proj
    
    def _project_point(self, lat: float, lon: float, 
                      center_lat: float, center_lon: float) -> Tuple[float, float]:
        """Project a single point to local coordinates"""
        # Transverse Mercator projection (more accurate than equirectangular)
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        center_lat_rad = np.radians(center_lat)
        center_lon_rad = np.radians(center_lon)
        
        # Earth radius
        R = 6371000.0
        
        # Simple approximation for small areas
        x = R * (lon_rad - center_lon_rad) * np.cos(center_lat_rad)
        y = R * (lat_rad - center_lat_rad)
        
        return x, y
    
    def _detect_and_remove_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Detect and remove outliers based on speed and acceleration"""
        df_clean = df.copy()
        outliers_removed = 0
        
        # Calculate speeds and accelerations
        df_clean['dt'] = df_clean['timestamp'].diff().dt.total_seconds()
        df_clean['dx'] = df_clean['x'].diff()
        df_clean['dy'] = df_clean['y'].diff()
        df_clean['distance'] = np.sqrt(df_clean['dx']**2 + df_clean['dy']**2)
        df_clean['calc_speed'] = df_clean['distance'] / df_clean['dt'] * 1.94384  # m/s to knots
        
        # Speed-based outlier detection
        speed_outliers = (
            (df_clean['calc_speed'] > self.max_speed_threshold) |
            (df_clean['calc_speed'] < 0)
        )
        
        # Acceleration-based outlier detection
        df_clean['acceleration'] = df_clean['calc_speed'].diff() / df_clean['dt']
        acceleration_outliers = np.abs(df_clean['acceleration']) > self.max_acceleration
        
        # Statistical outlier detection (position)
        x_mean, x_std = df_clean['x'].mean(), df_clean['x'].std()
        y_mean, y_std = df_clean['y'].mean(), df_clean['y'].std()
        
        position_outliers = (
            (np.abs(df_clean['x'] - x_mean) > self.outlier_threshold * x_std) |
            (np.abs(df_clean['y'] - y_mean) > self.outlier_threshold * y_std)
        )
        
        # Combine outlier conditions
        all_outliers = speed_outliers | acceleration_outliers | position_outliers
        
        # Remove outliers
        df_clean = df_clean[~all_outliers].reset_index(drop=True)
        outliers_removed = all_outliers.sum()
        
        return df_clean, outliers_removed
    
    def _fill_gaps(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Fill gaps in trajectory using interpolation"""
        if len(df) < 2:
            return df, 0
        
        df_filled = df.copy()
        gaps_filled = 0
        
        # Find large time gaps
        time_diffs = df['timestamp'].diff().dt.total_seconds()
        large_gaps = time_diffs > 300  # 5 minutes
        
        gap_indices = df[large_gaps].index.tolist()
        
        for gap_idx in gap_indices:
            if gap_idx == 0:
                continue
            
            gap_duration = time_diffs.iloc[gap_idx]
            
            # Only fill gaps smaller than threshold
            if gap_duration <= self.max_gap_duration:
                # Interpolate between points
                prev_point = df.iloc[gap_idx - 1]
                curr_point = df.iloc[gap_idx]
                
                # Number of points to interpolate
                n_points = max(1, int(gap_duration / 60))  # One point per minute
                
                interpolated_points = self._interpolate_gap(
                    prev_point, curr_point, n_points
                )
                
                # Insert interpolated points
                for i, interp_point in enumerate(interpolated_points):
                    insert_idx = gap_idx + i
                    df_filled = pd.concat([
                        df_filled.iloc[:insert_idx],
                        pd.DataFrame([interp_point]),
                        df_filled.iloc[insert_idx:]
                    ]).reset_index(drop=True)
                
                gaps_filled += len(interpolated_points)
        
        return df_filled, gaps_filled
    
    def _interpolate_gap(self, point1: pd.Series, point2: pd.Series, 
                        n_points: int) -> List[Dict]:
        """Interpolate points between two trajectory points"""
        interpolated = []
        
        # Time interpolation
        time_diff = (point2['timestamp'] - point1['timestamp']).total_seconds()
        time_step = time_diff / (n_points + 1)
        
        for i in range(1, n_points + 1):
            # Linear interpolation
            alpha = i / (n_points + 1)
            
            interp_time = point1['timestamp'] + pd.Timedelta(seconds=i * time_step)
            interp_lat = point1['latitude'] + alpha * (point2['latitude'] - point1['latitude'])
            interp_lon = point1['longitude'] + alpha * (point2['longitude'] - point1['longitude'])
            interp_x = point1['x'] + alpha * (point2['x'] - point1['x'])
            interp_y = point1['y'] + alpha * (point2['y'] - point1['y'])
            
            # Interpolate other fields
            interp_speed = point1.get('speed', 0) + alpha * (point2.get('speed', 0) - point1.get('speed', 0))
            interp_course = self._interpolate_course(point1.get('course', 0), point2.get('course', 0), alpha)
            
            interpolated_point = {
                'mmsi': point1['mmsi'],
                'timestamp': interp_time,
                'latitude': interp_lat,
                'longitude': interp_lon,
                'speed': interp_speed,
                'course': interp_course,
                'x': interp_x,
                'y': interp_y,
                'center_lat': point1['center_lat'],
                'center_lon': point1['center_lon'],
                'is_interpolated': True
            }
            
            interpolated.append(interpolated_point)
        
        return interpolated
    
    def _interpolate_course(self, course1: float, course2: float, alpha: float) -> float:
        """Interpolate course angles handling wraparound"""
        # Convert to radians
        c1_rad = np.radians(course1)
        c2_rad = np.radians(course2)
        
        # Convert to unit vectors
        x1, y1 = np.cos(c1_rad), np.sin(c1_rad)
        x2, y2 = np.cos(c2_rad), np.sin(c2_rad)
        
        # Interpolate vectors
        x_interp = x1 + alpha * (x2 - x1)
        y_interp = y1 + alpha * (y2 - y1)
        
        # Convert back to angle
        interp_course = np.degrees(np.arctan2(y_interp, x_interp))
        if interp_course < 0:
            interp_course += 360
        
        return interp_course
    
    def _apply_kalman_filter(self, df: pd.DataFrame) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Apply Kalman filter to trajectory data"""
        n_points = len(df)
        
        # Initialize state and covariance
        x_init = np.array([df.iloc[0]['x'], df.iloc[0]['y'], 0.0, 0.0])
        P_init = np.eye(4) * 100.0  # Initial uncertainty
        
        # Storage for results
        filtered_states = []
        filtered_covariances = []
        
        x = x_init.copy()
        P = P_init.copy()
        
        for i in range(n_points):
            # Update time step
            if i > 0:
                self.dt = (df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']).total_seconds()
                self.dt = max(1.0, min(3600.0, self.dt))  # Clamp between 1s and 1h
                self._update_kalman_matrices()
            
            # Prediction step
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q
            
            # Observation
            z = np.array([df.iloc[i]['x'], df.iloc[i]['y']])
            
            # Update step
            y = z - self.H @ x_pred  # Innovation
            S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance
            K = P_pred @ self.H.T @ inv(S)  # Kalman gain
            
            x = x_pred + K @ y
            P = (np.eye(4) - K @ self.H) @ P_pred
            
            filtered_states.append(x.copy())
            filtered_covariances.append(P.copy())
        
        return filtered_states, filtered_covariances
    
    def _update_kalman_matrices(self):
        """Update Kalman filter matrices with new time step"""
        # Update transition matrix
        self.F[0, 2] = self.dt
        self.F[1, 3] = self.dt
        
        # Update process noise covariance
        q = self.process_noise_std ** 2
        self.Q = np.array([
            [q * self.dt**4 / 4, 0, q * self.dt**3 / 2, 0],
            [0, q * self.dt**4 / 4, 0, q * self.dt**3 / 2],
            [q * self.dt**3 / 2, 0, q * self.dt**2, 0],
            [0, q * self.dt**3 / 2, 0, q * self.dt**2]
        ])
    
    def _apply_rts_smoothing(self, filtered_states: List[np.ndarray], 
                           filtered_covariances: List[np.ndarray],
                           df: pd.DataFrame) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Apply Rauch-Tung-Striebel (RTS) smoothing"""
        n_points = len(filtered_states)
        
        # Initialize with filtered estimates
        smoothed_states = filtered_states.copy()
        smoothed_covariances = filtered_covariances.copy()
        
        # Backward pass
        for i in range(n_points - 2, -1, -1):
            # Update time step
            if i < n_points - 1:
                self.dt = (df.iloc[i+1]['timestamp'] - df.iloc[i]['timestamp']).total_seconds()
                self.dt = max(1.0, min(3600.0, self.dt))
                self._update_kalman_matrices()
            
            # Prediction for next step
            x_pred = self.F @ smoothed_states[i]
            P_pred = self.F @ smoothed_covariances[i] @ self.F.T + self.Q
            
            # Smoother gain
            try:
                A = smoothed_covariances[i] @ self.F.T @ inv(P_pred)
            except np.linalg.LinAlgError:
                # Handle singular matrix
                A = np.zeros((4, 4))
            
            # Smoothed estimates
            smoothed_states[i] = smoothed_states[i] + A @ (smoothed_states[i+1] - x_pred)
            smoothed_covariances[i] = (smoothed_covariances[i] + 
                                     A @ (smoothed_covariances[i+1] - P_pred) @ A.T)
        
        return smoothed_states, smoothed_covariances
    
    def _convert_to_smoothed_points(self, smoothed_states: List[np.ndarray],
                                   smoothed_covariances: List[np.ndarray],
                                   df: pd.DataFrame) -> List[SmoothedPoint]:
        """Convert smoothed states back to geographical coordinates"""
        smoothed_points = []
        
        for i, (state, cov) in enumerate(zip(smoothed_states, smoothed_covariances)):
            row = df.iloc[i]
            
            # Extract state components
            x, y, vx, vy = state
            
            # Extract uncertainties
            uncertainty_x = np.sqrt(cov[0, 0])
            uncertainty_y = np.sqrt(cov[1, 1])
            uncertainty_vx = np.sqrt(cov[2, 2])
            uncertainty_vy = np.sqrt(cov[3, 3])
            
            # Convert back to geographical coordinates
            lat, lon = self._unproject_point(x, y, row['center_lat'], row['center_lon'])
            
            # Calculate speed and course from velocity components
            speed_ms = np.sqrt(vx**2 + vy**2)
            speed_knots = speed_ms * 1.94384  # m/s to knots
            course = np.degrees(np.arctan2(vx, vy))
            if course < 0:
                course += 360
            
            # Calculate quality score
            quality_score = self._calculate_point_quality(
                uncertainty_x, uncertainty_y, uncertainty_vx, uncertainty_vy
            )
            
            smoothed_point = SmoothedPoint(
                timestamp=row['timestamp'],
                latitude=lat,
                longitude=lon,
                speed=speed_knots,
                course=course,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                uncertainty_x=uncertainty_x,
                uncertainty_y=uncertainty_y,
                uncertainty_vx=uncertainty_vx,
                uncertainty_vy=uncertainty_vy,
                quality_score=quality_score,
                is_interpolated=row.get('is_interpolated', False),
                original_index=i
            )
            
            smoothed_points.append(smoothed_point)
        
        return smoothed_points
    
    def _unproject_point(self, x: float, y: float, 
                        center_lat: float, center_lon: float) -> Tuple[float, float]:
        """Convert local coordinates back to geographical coordinates"""
        R = 6371000.0  # Earth radius
        
        center_lat_rad = np.radians(center_lat)
        center_lon_rad = np.radians(center_lon)
        
        lat_rad = center_lat_rad + y / R
        lon_rad = center_lon_rad + x / (R * np.cos(center_lat_rad))
        
        lat = np.degrees(lat_rad)
        lon = np.degrees(lon_rad)
        
        return lat, lon
    
    def _calculate_point_quality(self, unc_x: float, unc_y: float, 
                                unc_vx: float, unc_vy: float) -> float:
        """Calculate quality score for a smoothed point"""
        # Normalize uncertainties
        max_pos_uncertainty = 100.0  # meters
        max_vel_uncertainty = 5.0   # m/s
        
        pos_quality = 1.0 - min(1.0, np.sqrt(unc_x**2 + unc_y**2) / max_pos_uncertainty)
        vel_quality = 1.0 - min(1.0, np.sqrt(unc_vx**2 + unc_vy**2) / max_vel_uncertainty)
        
        # Combined quality (weighted average)
        quality = 0.7 * pos_quality + 0.3 * vel_quality
        
        return max(0.0, min(1.0, quality))
    
    def _calculate_quality_metrics(self, original_df: pd.DataFrame,
                                  smoothed_points: List[SmoothedPoint],
                                  outliers_removed: int,
                                  gaps_filled: int) -> Dict[str, float]:
        """Calculate quality metrics for the smoothing result"""
        if not smoothed_points:
            return {}
        
        # Position accuracy (RMS error for non-interpolated points)
        position_errors = []
        speed_errors = []
        
        for i, point in enumerate(smoothed_points):
            if not point.is_interpolated and i < len(original_df):
                orig_row = original_df.iloc[i]
                
                # Position error
                pos_error = geodesic(
                    (orig_row['latitude'], orig_row['longitude']),
                    (point.latitude, point.longitude)
                ).meters
                position_errors.append(pos_error)
                
                # Speed error
                if 'speed' in orig_row:
                    speed_error = abs(orig_row['speed'] - point.speed)
                    speed_errors.append(speed_error)
        
        # Smoothness metrics
        accelerations = []
        course_changes = []
        
        for i in range(1, len(smoothed_points)):
            prev_point = smoothed_points[i-1]
            curr_point = smoothed_points[i]
            
            # Time difference
            dt = (curr_point.timestamp - prev_point.timestamp).total_seconds()
            if dt > 0:
                # Acceleration
                dv = np.sqrt((curr_point.vx - prev_point.vx)**2 + 
                           (curr_point.vy - prev_point.vy)**2)
                acceleration = dv / dt
                accelerations.append(acceleration)
                
                # Course change
                course_change = abs(curr_point.course - prev_point.course)
                if course_change > 180:
                    course_change = 360 - course_change
                course_changes.append(course_change)
        
        # Overall quality
        avg_quality = np.mean([point.quality_score for point in smoothed_points])
        
        # Data completeness
        interpolated_ratio = sum(1 for p in smoothed_points if p.is_interpolated) / len(smoothed_points)
        
        metrics = {
            'avg_position_error_m': np.mean(position_errors) if position_errors else 0.0,
            'max_position_error_m': np.max(position_errors) if position_errors else 0.0,
            'avg_speed_error_knots': np.mean(speed_errors) if speed_errors else 0.0,
            'avg_acceleration_ms2': np.mean(accelerations) if accelerations else 0.0,
            'max_acceleration_ms2': np.max(accelerations) if accelerations else 0.0,
            'avg_course_change_deg': np.mean(course_changes) if course_changes else 0.0,
            'avg_quality_score': avg_quality,
            'interpolated_ratio': interpolated_ratio,
            'outlier_ratio': outliers_removed / len(original_df) if len(original_df) > 0 else 0.0,
            'data_reduction_ratio': (len(original_df) - len(smoothed_points)) / len(original_df) if len(original_df) > 0 else 0.0
        }
        
        return metrics
    
    def _create_empty_result(self, mmsi: int) -> SmoothingResult:
        """Create empty result for insufficient data"""
        return SmoothingResult(
            mmsi=mmsi,
            smoothed_points=[],
            original_length=0,
            smoothed_length=0,
            gaps_filled=0,
            outliers_removed=0,
            quality_metrics={},
            processing_time=0.0
        )
    
    def smooth_multiple_trajectories(self, df: pd.DataFrame) -> Dict[int, SmoothingResult]:
        """
        Smooth trajectories for multiple vessels.
        
        Args:
            df: DataFrame with trajectory data for multiple vessels
            
        Returns:
            Dictionary mapping MMSI to smoothing results
        """
        results = {}
        
        for mmsi in df['mmsi'].unique():
            vessel_df = df[df['mmsi'] == mmsi].copy()
            result = self.smooth_trajectory(vessel_df)
            results[mmsi] = result
        
        return results
    
    def export_smoothed_trajectories(self, results: Dict[int, SmoothingResult]) -> pd.DataFrame:
        """Export smoothed trajectories to DataFrame"""
        all_points = []
        
        for mmsi, result in results.items():
            for point in result.smoothed_points:
                record = {
                    'mmsi': mmsi,
                    'timestamp': point.timestamp,
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'speed': point.speed,
                    'course': point.course,
                    'uncertainty_x': point.uncertainty_x,
                    'uncertainty_y': point.uncertainty_y,
                    'quality_score': point.quality_score,
                    'is_interpolated': point.is_interpolated
                }
                all_points.append(record)
        
        return pd.DataFrame(all_points)