"""
TCPA/DCPA Calculator Module

This module provides functionality for calculating:
- Time to Closest Point of Approach (TCPA)
- Distance at Closest Point of Approach (DCPA)
- Collision risk assessment with vectorized calculations
- Dynamic adjacency matrix construction for temporal scenes
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from geopy.distance import geodesic
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)


@dataclass
class VesselPosition:
    """Represents a vessel position with motion vectors"""
    mmsi: int
    timestamp: pd.Timestamp
    latitude: float
    longitude: float
    speed: float  # knots
    course: float  # degrees
    x: float  # local projected x coordinate (meters)
    y: float  # local projected y coordinate (meters)
    vx: float  # velocity x component (m/s)
    vy: float  # velocity y component (m/s)


@dataclass
class TCPAResult:
    """Result of TCPA/DCPA calculation"""
    vessel_1: int
    vessel_2: int
    tcpa: float  # seconds, negative if CPA is in the past
    dcpa: float  # meters
    risk_level: float  # 0-1 scale
    relative_speed: float  # m/s
    relative_bearing: float  # degrees
    cpa_time: Optional[pd.Timestamp] = None
    cpa_position_1: Optional[Tuple[float, float]] = None  # (x, y)
    cpa_position_2: Optional[Tuple[float, float]] = None  # (x, y)


class TCPADCPACalculator:
    """
    High-performance TCPA/DCPA calculator with vectorized operations.
    Uses local coordinate projection for accurate calculations.
    """
    
    def __init__(self,
                 risk_distance_threshold: float = 500.0,  # meters
                 risk_time_threshold: float = 1800.0,  # seconds (30 minutes)
                 min_speed_threshold: float = 0.5,  # knots
                 max_calculation_distance: float = 50000.0,  # meters (50 km)
                 projection_center: Optional[Tuple[float, float]] = None):
        
        self.risk_distance_threshold = risk_distance_threshold
        self.risk_time_threshold = risk_time_threshold
        self.min_speed_threshold = min_speed_threshold
        self.max_calculation_distance = max_calculation_distance
        self.projection_center = projection_center
        
        # Performance optimization: pre-allocate arrays
        self._position_cache = {}
        self._tcpa_cache = {}
    
    def calculate_tcpa_dcpa_matrix(self, df: pd.DataFrame, 
                                  timestamp: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Calculate TCPA/DCPA matrix for all vessel pairs at a given timestamp.
        
        Args:
            df: DataFrame with vessel positions
            timestamp: Timestamp for calculation
            
        Returns:
            Tuple of (tcpa_matrix, dcpa_matrix, vessel_list)
        """
        # Filter data for the specific timestamp (within tolerance)
        time_tolerance = pd.Timedelta(minutes=2)
        time_mask = (df['timestamp'] >= timestamp - time_tolerance) & \
                   (df['timestamp'] <= timestamp + time_tolerance)
        
        vessel_data = df[time_mask].copy()
        
        if len(vessel_data) < 2:
            return np.array([]), np.array([]), []
        
        # Get unique vessels
        vessels = vessel_data['mmsi'].unique()
        n_vessels = len(vessels)
        
        # Initialize matrices
        tcpa_matrix = np.full((n_vessels, n_vessels), np.inf)
        dcpa_matrix = np.full((n_vessels, n_vessels), np.inf)
        
        # Convert to vessel positions with local projection
        vessel_positions = self._prepare_vessel_positions(vessel_data, timestamp)
        
        # Vectorized TCPA/DCPA calculation
        for i, vessel_1 in enumerate(vessels):
            for j, vessel_2 in enumerate(vessels):
                if i != j:
                    pos1 = vessel_positions.get(vessel_1)
                    pos2 = vessel_positions.get(vessel_2)
                    
                    if pos1 and pos2:
                        tcpa, dcpa = self._calculate_tcpa_dcpa_vectorized(pos1, pos2)
                        tcpa_matrix[i, j] = tcpa
                        dcpa_matrix[i, j] = dcpa
        
        return tcpa_matrix, dcpa_matrix, vessels.tolist()
    
    def calculate_risk_graph(self, df: pd.DataFrame, 
                           timestamp: pd.Timestamp,
                           decay_factor: float = 0.1) -> np.ndarray:
        """
        Calculate risk graph with exponential decay weights.
        
        Args:
            df: DataFrame with vessel positions
            timestamp: Timestamp for calculation
            decay_factor: Exponential decay factor for time weighting
            
        Returns:
            Risk adjacency matrix
        """
        tcpa_matrix, dcpa_matrix, vessels = self.calculate_tcpa_dcpa_matrix(df, timestamp)
        
        if len(vessels) == 0:
            return np.array([])
        
        n_vessels = len(vessels)
        risk_matrix = np.zeros((n_vessels, n_vessels))
        
        # Calculate risk for each vessel pair
        for i in range(n_vessels):
            for j in range(n_vessels):
                if i != j:
                    tcpa = tcpa_matrix[i, j]
                    dcpa = dcpa_matrix[i, j]
                    
                    # Calculate base risk
                    risk = self._calculate_risk_score(tcpa, dcpa)
                    
                    # Apply exponential decay based on time
                    if tcpa > 0:
                        time_weight = np.exp(-decay_factor * tcpa / 3600.0)  # Convert to hours
                        risk *= time_weight
                    
                    risk_matrix[i, j] = risk
        
        return risk_matrix
    
    def calculate_batch_tcpa_dcpa(self, df: pd.DataFrame, 
                                 time_window_minutes: float = 5.0) -> List[TCPAResult]:
        """
        Calculate TCPA/DCPA for all vessel pairs in batch mode.
        
        Args:
            df: DataFrame with vessel positions
            time_window_minutes: Time window for grouping calculations
            
        Returns:
            List of TCPA results
        """
        results = []
        
        # Group by time windows
        df['time_window'] = df['timestamp'].dt.floor(f'{time_window_minutes}min')
        
        for window_time, window_df in df.groupby('time_window'):
            # Get all unique vessel pairs
            vessels = window_df['mmsi'].unique()
            
            for i, vessel_1 in enumerate(vessels):
                for vessel_2 in vessels[i+1:]:
                    # Get vessel data
                    v1_data = window_df[window_df['mmsi'] == vessel_1]
                    v2_data = window_df[window_df['mmsi'] == vessel_2]
                    
                    if len(v1_data) > 0 and len(v2_data) > 0:
                        # Take most recent position for each vessel
                        v1_pos = v1_data.iloc[-1]
                        v2_pos = v2_data.iloc[-1]
                        
                        # Calculate TCPA/DCPA
                        result = self._calculate_tcpa_dcpa_pair(v1_pos, v2_pos, window_time)
                        
                        if result and result.dcpa < self.max_calculation_distance:
                            results.append(result)
        
        return results
    
    def _prepare_vessel_positions(self, df: pd.DataFrame, 
                                 reference_time: pd.Timestamp) -> Dict[int, VesselPosition]:
        """Prepare vessel positions with local coordinate projection"""
        vessel_positions = {}
        
        # Determine projection center if not set
        if self.projection_center is None:
            center_lat = df['latitude'].mean()
            center_lon = df['longitude'].mean()
        else:
            center_lat, center_lon = self.projection_center
        
        # Convert to local coordinates
        for _, row in df.iterrows():
            # Project to local coordinates (approximate)
            x, y = self._project_to_local(row['latitude'], row['longitude'], 
                                        center_lat, center_lon)
            
            # Convert speed and course to velocity components
            speed_ms = row.get('speed', 0.0) * 0.514444  # knots to m/s
            course_rad = np.radians(row.get('course', 0.0))
            
            vx = speed_ms * np.sin(course_rad)  # East component
            vy = speed_ms * np.cos(course_rad)  # North component
            
            vessel_pos = VesselPosition(
                mmsi=row['mmsi'],
                timestamp=row['timestamp'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                speed=row.get('speed', 0.0),
                course=row.get('course', 0.0),
                x=x,
                y=y,
                vx=vx,
                vy=vy
            )
            
            vessel_positions[row['mmsi']] = vessel_pos
        
        return vessel_positions
    
    def _project_to_local(self, lat: float, lon: float, 
                         center_lat: float, center_lon: float) -> Tuple[float, float]:
        """Project lat/lon to local Cartesian coordinates"""
        # Simple equirectangular projection (good for small areas)
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        center_lat_rad = np.radians(center_lat)
        center_lon_rad = np.radians(center_lon)
        
        # Earth radius in meters
        R = 6371000.0
        
        x = R * (lon_rad - center_lon_rad) * np.cos(center_lat_rad)
        y = R * (lat_rad - center_lat_rad)
        
        return x, y
    
    def _calculate_tcpa_dcpa_vectorized(self, pos1: VesselPosition, 
                                      pos2: VesselPosition) -> Tuple[float, float]:
        """
        Calculate TCPA and DCPA using vectorized operations.
        
        Uses the standard maritime collision avoidance formulas:
        TCPA = -((x1-x2)*(vx1-vx2) + (y1-y2)*(vy1-vy2)) / ((vx1-vx2)^2 + (vy1-vy2)^2)
        DCPA = sqrt((x1-x2)^2 + (y1-y2)^2 + 2*TCPA*((x1-x2)*(vx1-vx2) + (y1-y2)*(vy1-vy2)) + TCPA^2*((vx1-vx2)^2 + (vy1-vy2)^2))
        """
        # Position differences
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        
        # Velocity differences
        dvx = pos1.vx - pos2.vx
        dvy = pos1.vy - pos2.vy
        
        # Relative velocity magnitude squared
        dv_squared = dvx**2 + dvy**2
        
        # If vessels have same velocity, no collision risk
        if dv_squared < 1e-6:  # Very small relative velocity
            current_distance = np.sqrt(dx**2 + dy**2)
            return np.inf, current_distance
        
        # Calculate TCPA
        tcpa = -(dx * dvx + dy * dvy) / dv_squared
        
        # Calculate DCPA
        if tcpa < 0:
            # CPA is in the past, use current distance
            dcpa = np.sqrt(dx**2 + dy**2)
        else:
            # CPA is in the future
            x1_cpa = pos1.x + pos1.vx * tcpa
            y1_cpa = pos1.y + pos1.vy * tcpa
            x2_cpa = pos2.x + pos2.vx * tcpa
            y2_cpa = pos2.y + pos2.vy * tcpa
            
            dcpa = np.sqrt((x1_cpa - x2_cpa)**2 + (y1_cpa - y2_cpa)**2)
        
        return tcpa, dcpa
    
    def _calculate_tcpa_dcpa_pair(self, v1_row: pd.Series, v2_row: pd.Series, 
                                 reference_time: pd.Timestamp) -> Optional[TCPAResult]:
        """Calculate TCPA/DCPA for a specific vessel pair"""
        # Skip if vessels are too far apart
        initial_distance = geodesic(
            (v1_row['latitude'], v1_row['longitude']),
            (v2_row['latitude'], v2_row['longitude'])
        ).meters
        
        if initial_distance > self.max_calculation_distance:
            return None
        
        # Skip if vessels are moving too slowly
        if (v1_row.get('speed', 0) < self.min_speed_threshold and 
            v2_row.get('speed', 0) < self.min_speed_threshold):
            return None
        
        # Create vessel positions
        center_lat = (v1_row['latitude'] + v2_row['latitude']) / 2
        center_lon = (v1_row['longitude'] + v2_row['longitude']) / 2
        
        # Project to local coordinates
        x1, y1 = self._project_to_local(v1_row['latitude'], v1_row['longitude'], 
                                       center_lat, center_lon)
        x2, y2 = self._project_to_local(v2_row['latitude'], v2_row['longitude'], 
                                       center_lat, center_lon)
        
        # Convert speeds to m/s and get velocity components
        speed1_ms = v1_row.get('speed', 0.0) * 0.514444
        speed2_ms = v2_row.get('speed', 0.0) * 0.514444
        
        course1_rad = np.radians(v1_row.get('course', 0.0))
        course2_rad = np.radians(v2_row.get('course', 0.0))
        
        vx1 = speed1_ms * np.sin(course1_rad)
        vy1 = speed1_ms * np.cos(course1_rad)
        vx2 = speed2_ms * np.sin(course2_rad)
        vy2 = speed2_ms * np.cos(course2_rad)
        
        # Create position objects
        pos1 = VesselPosition(
            mmsi=v1_row['mmsi'], timestamp=v1_row['timestamp'],
            latitude=v1_row['latitude'], longitude=v1_row['longitude'],
            speed=v1_row.get('speed', 0.0), course=v1_row.get('course', 0.0),
            x=x1, y=y1, vx=vx1, vy=vy1
        )
        
        pos2 = VesselPosition(
            mmsi=v2_row['mmsi'], timestamp=v2_row['timestamp'],
            latitude=v2_row['latitude'], longitude=v2_row['longitude'],
            speed=v2_row.get('speed', 0.0), course=v2_row.get('course', 0.0),
            x=x2, y=y2, vx=vx2, vy=vy2
        )
        
        # Calculate TCPA/DCPA
        tcpa, dcpa = self._calculate_tcpa_dcpa_vectorized(pos1, pos2)
        
        # Calculate additional metrics
        relative_speed = np.sqrt((vx1 - vx2)**2 + (vy1 - vy2)**2)
        relative_bearing = np.degrees(np.arctan2(x2 - x1, y2 - y1))
        if relative_bearing < 0:
            relative_bearing += 360
        
        # Calculate risk level
        risk_level = self._calculate_risk_score(tcpa, dcpa)
        
        # Calculate CPA positions and time
        cpa_time = None
        cpa_pos1 = None
        cpa_pos2 = None
        
        if tcpa > 0:
            cpa_time = reference_time + pd.Timedelta(seconds=tcpa)
            cpa_pos1 = (x1 + vx1 * tcpa, y1 + vy1 * tcpa)
            cpa_pos2 = (x2 + vx2 * tcpa, y2 + vy2 * tcpa)
        
        result = TCPAResult(
            vessel_1=v1_row['mmsi'],
            vessel_2=v2_row['mmsi'],
            tcpa=tcpa,
            dcpa=dcpa,
            risk_level=risk_level,
            relative_speed=relative_speed,
            relative_bearing=relative_bearing,
            cpa_time=cpa_time,
            cpa_position_1=cpa_pos1,
            cpa_position_2=cpa_pos2
        )
        
        return result
    
    def _calculate_risk_score(self, tcpa: float, dcpa: float) -> float:
        """Calculate risk score based on TCPA and DCPA"""
        if tcpa < 0:  # CPA in the past
            return 0.0
        
        if tcpa > self.risk_time_threshold:  # Too far in the future
            return 0.0
        
        # Distance risk (0-1, higher for closer approach)
        distance_risk = max(0, (self.risk_distance_threshold - dcpa) / self.risk_distance_threshold)
        
        # Time risk (0-1, higher for sooner approach)
        time_risk = max(0, (self.risk_time_threshold - tcpa) / self.risk_time_threshold)
        
        # Combined risk (geometric mean)
        risk_score = np.sqrt(distance_risk * time_risk)
        
        return min(1.0, risk_score)
    
    def get_high_risk_pairs(self, results: List[TCPAResult], 
                           risk_threshold: float = 0.5) -> List[TCPAResult]:
        """Filter results to get high-risk vessel pairs"""
        return [result for result in results if result.risk_level >= risk_threshold]
    
    def create_dynamic_adjacency_matrix(self, df: pd.DataFrame, 
                                      timestamps: List[pd.Timestamp],
                                      risk_threshold: float = 0.3) -> Dict[pd.Timestamp, np.ndarray]:
        """
        Create dynamic adjacency matrices for multiple timestamps.
        
        Args:
            df: DataFrame with vessel positions
            timestamps: List of timestamps to calculate matrices for
            risk_threshold: Minimum risk level for adjacency
            
        Returns:
            Dictionary mapping timestamps to adjacency matrices
        """
        adjacency_matrices = {}
        
        for timestamp in timestamps:
            risk_matrix = self.calculate_risk_graph(df, timestamp)
            
            if risk_matrix.size > 0:
                # Create binary adjacency matrix based on risk threshold
                adjacency_matrix = (risk_matrix >= risk_threshold).astype(int)
                adjacency_matrices[timestamp] = adjacency_matrix
            else:
                adjacency_matrices[timestamp] = np.array([])
        
        return adjacency_matrices
    
    def export_results_to_dataframe(self, results: List[TCPAResult]) -> pd.DataFrame:
        """Export TCPA results to DataFrame"""
        if not results:
            return pd.DataFrame()
        
        data = []
        for result in results:
            record = {
                'vessel_1': result.vessel_1,
                'vessel_2': result.vessel_2,
                'tcpa_seconds': result.tcpa,
                'dcpa_meters': result.dcpa,
                'risk_level': result.risk_level,
                'relative_speed_ms': result.relative_speed,
                'relative_bearing_deg': result.relative_bearing,
                'cpa_time': result.cpa_time
            }
            data.append(record)
        
        return pd.DataFrame(data)
    
    def clear_cache(self):
        """Clear internal caches to free memory"""
        self._position_cache.clear()
        self._tcpa_cache.clear()
    
    def get_statistics(self, results: List[TCPAResult]) -> Dict[str, Any]:
        """Get statistics about TCPA/DCPA calculations"""
        if not results:
            return {}
        
        tcpa_values = [r.tcpa for r in results if r.tcpa > 0]
        dcpa_values = [r.dcpa for r in results]
        risk_values = [r.risk_level for r in results]
        
        stats = {
            'total_calculations': len(results),
            'valid_tcpa_count': len(tcpa_values),
            'high_risk_count': len([r for r in results if r.risk_level > 0.7]),
            'medium_risk_count': len([r for r in results if 0.3 <= r.risk_level <= 0.7]),
            'low_risk_count': len([r for r in results if r.risk_level < 0.3]),
            'tcpa_stats': {
                'mean': np.mean(tcpa_values) if tcpa_values else 0,
                'std': np.std(tcpa_values) if tcpa_values else 0,
                'min': np.min(tcpa_values) if tcpa_values else 0,
                'max': np.max(tcpa_values) if tcpa_values else 0
            },
            'dcpa_stats': {
                'mean': np.mean(dcpa_values),
                'std': np.std(dcpa_values),
                'min': np.min(dcpa_values),
                'max': np.max(dcpa_values)
            },
            'risk_stats': {
                'mean': np.mean(risk_values),
                'std': np.std(risk_values),
                'min': np.min(risk_values),
                'max': np.max(risk_values)
            }
        }
        
        return stats