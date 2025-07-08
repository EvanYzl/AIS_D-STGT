"""
Trajectory Segmentation Module

This module provides functionality for segmenting AIS trajectories based on:
- Behavior changes (speed, course, navigation state)
- Port visits and voyage partitioning
- Time gaps and spatial discontinuities
- Navigation state transitions
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


class SegmentType(Enum):
    """Types of trajectory segments"""

    VOYAGE = "voyage"
    PORT_VISIT = "port_visit"
    ANCHORING = "anchoring"
    MANEUVERING = "maneuvering"
    TRANSIT = "transit"
    UNKNOWN = "unknown"


class NavigationState(Enum):
    """Navigation states based on AIS data"""

    UNDER_WAY_USING_ENGINE = 0
    AT_ANCHOR = 1
    NOT_UNDER_COMMAND = 2
    RESTRICTED_MANOEUVRABILITY = 3
    CONSTRAINED_BY_DRAUGHT = 4
    MOORED = 5
    AGROUND = 6
    ENGAGED_IN_FISHING = 7
    UNDER_WAY_SAILING = 8
    RESERVED_HSC = 9
    RESERVED_WIG = 10
    RESERVED_11 = 11
    RESERVED_12 = 12
    RESERVED_13 = 13
    AIS_SART = 14
    UNDEFINED = 15


@dataclass
class TrajectorySegment:
    """Represents a trajectory segment"""

    mmsi: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    start_idx: int
    end_idx: int
    segment_type: SegmentType
    properties: dict[str, Any]

    @property
    def duration(self) -> pd.Timedelta:
        return self.end_time - self.start_time

    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx + 1


@dataclass
class PortArea:
    """Represents a port area"""

    name: str
    polygon: Polygon
    center: tuple[float, float]  # (lat, lon)
    radius_km: float


class TrajectorySegmenter:
    """
    Segments AIS trajectories based on behavior changes, port visits, and navigation states.
    """

    def __init__(
        self,
        speed_threshold: float = 2.0,  # knots
        course_threshold: float = 30.0,  # degrees
        time_gap_threshold: float = 3600.0,  # seconds
        distance_gap_threshold: float = 10.0,  # km
        port_distance_threshold: float = 5.0,  # km
        anchoring_speed_threshold: float = 1.0,  # knots
        anchoring_time_threshold: float = 1800.0,  # seconds
        maneuvering_speed_threshold: float = 5.0,  # knots
        maneuvering_course_change_threshold: float = 45.0,
    ):  # degrees
        self.speed_threshold = speed_threshold
        self.course_threshold = course_threshold
        self.time_gap_threshold = time_gap_threshold
        self.distance_gap_threshold = distance_gap_threshold
        self.port_distance_threshold = port_distance_threshold
        self.anchoring_speed_threshold = anchoring_speed_threshold
        self.anchoring_time_threshold = anchoring_time_threshold
        self.maneuvering_speed_threshold = maneuvering_speed_threshold
        self.maneuvering_course_change_threshold = maneuvering_course_change_threshold

        # Default port areas (can be extended)
        self.port_areas = self._initialize_default_ports()

    def _initialize_default_ports(self) -> list[PortArea]:
        """Initialize default port areas"""
        ports = []

        # Example ports - in practice, these would be loaded from a database
        port_configs = [
            {"name": "Hamburg", "lat": 53.5511, "lon": 9.9937, "radius": 15.0},
            {"name": "Rotterdam", "lat": 51.9244, "lon": 4.4777, "radius": 20.0},
            {"name": "Antwerp", "lat": 51.2194, "lon": 4.4025, "radius": 12.0},
            {"name": "Bremen", "lat": 53.0793, "lon": 8.8017, "radius": 10.0},
        ]

        for config in port_configs:
            # Create circular polygon around port center
            center_point = Point(config["lon"], config["lat"])
            # Approximate degrees for radius (rough conversion)
            radius_deg = config["radius"] / 111.0  # ~111 km per degree
            polygon = center_point.buffer(radius_deg)

            ports.append(
                PortArea(
                    name=config["name"],
                    polygon=polygon,
                    center=(config["lat"], config["lon"]),
                    radius_km=config["radius"],
                )
            )

        return ports

    def segment_trajectory(self, df: pd.DataFrame) -> list[TrajectorySegment]:
        """
        Segment a trajectory based on behavior changes and port visits.

        Args:
            df: DataFrame with AIS data for a single vessel

        Returns:
            List of trajectory segments
        """
        if len(df) < 2:
            return []

        # Ensure data is sorted by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Calculate derived features
        df = self._calculate_derived_features(df)

        # Detect different types of segments

        # 1. Detect time/space gaps
        gap_segments = self._detect_gap_segments(df)

        # 2. Detect port visits
        port_segments = self._detect_port_visits(df)

        # 3. Detect anchoring periods
        anchoring_segments = self._detect_anchoring_periods(df)

        # 4. Detect maneuvering periods
        maneuvering_segments = self._detect_maneuvering_periods(df)

        # 5. Detect behavior changes
        behavior_segments = self._detect_behavior_changes(df)

        # Combine and merge overlapping segments
        all_segments = (
            gap_segments
            + port_segments
            + anchoring_segments
            + maneuvering_segments
            + behavior_segments
        )

        # Sort segments by start time
        all_segments.sort(key=lambda x: x.start_time)

        # Merge overlapping segments and fill gaps
        merged_segments = self._merge_and_fill_segments(df, all_segments)

        return merged_segments

    def _calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features for segmentation"""
        df = df.copy()

        # Calculate time differences
        df["time_diff"] = df["timestamp"].diff().dt.total_seconds()

        # Calculate distances
        distances = []
        for i in range(len(df)):
            if i == 0:
                distances.append(0.0)
            else:
                dist = geodesic(
                    (df.iloc[i - 1]["latitude"], df.iloc[i - 1]["longitude"]),
                    (df.iloc[i]["latitude"], df.iloc[i]["longitude"]),
                ).kilometers
                distances.append(dist)

        df["distance"] = distances

        # Calculate speed if not present
        if "speed" not in df.columns:
            df["speed"] = np.where(
                df["time_diff"] > 0,
                (df["distance"] * 1000) / df["time_diff"] * 1.94384,  # m/s to knots
                0,
            )

        # Calculate course changes
        if "course" in df.columns:
            df["course_change"] = np.abs(df["course"].diff())
            # Handle wraparound (e.g., 359° to 1°)
            df["course_change"] = np.where(
                df["course_change"] > 180,
                360 - df["course_change"],
                df["course_change"],
            )
        else:
            df["course_change"] = 0

        # Calculate acceleration
        df["acceleration"] = df["speed"].diff() / df["time_diff"]

        # Identify port proximity
        df["in_port"] = df.apply(self._is_in_port, axis=1)
        df["port_name"] = df.apply(self._get_port_name, axis=1)

        return df

    def _is_in_port(self, row) -> bool:
        """Check if position is within any port area"""
        point = Point(row["longitude"], row["latitude"])
        return any(port.polygon.contains(point) for port in self.port_areas)

    def _get_port_name(self, row) -> str | None:
        """Get name of port if position is within port area"""
        point = Point(row["longitude"], row["latitude"])
        for port in self.port_areas:
            if port.polygon.contains(point):
                return port.name
        return None

    def _detect_gap_segments(self, df: pd.DataFrame) -> list[TrajectorySegment]:
        """Detect segments separated by large time or distance gaps"""
        segments = []

        # Find large gaps
        time_gaps = df["time_diff"] > self.time_gap_threshold
        distance_gaps = df["distance"] > self.distance_gap_threshold
        gaps = time_gaps | distance_gaps

        gap_indices = df[gaps].index.tolist()

        # Create segments between gaps
        start_idx = 0
        for gap_idx in gap_indices:
            if gap_idx > start_idx:
                segment = TrajectorySegment(
                    mmsi=df.iloc[0]["mmsi"],
                    start_time=df.iloc[start_idx]["timestamp"],
                    end_time=df.iloc[gap_idx - 1]["timestamp"],
                    start_idx=start_idx,
                    end_idx=gap_idx - 1,
                    segment_type=SegmentType.VOYAGE,
                    properties={
                        "gap_before": gap_idx in gap_indices,
                        "time_gap": df.iloc[gap_idx]["time_diff"]
                        if gap_idx < len(df)
                        else 0,
                        "distance_gap": df.iloc[gap_idx]["distance"]
                        if gap_idx < len(df)
                        else 0,
                    },
                )
                segments.append(segment)
            start_idx = gap_idx

        # Add final segment
        if start_idx < len(df):
            segment = TrajectorySegment(
                mmsi=df.iloc[0]["mmsi"],
                start_time=df.iloc[start_idx]["timestamp"],
                end_time=df.iloc[-1]["timestamp"],
                start_idx=start_idx,
                end_idx=len(df) - 1,
                segment_type=SegmentType.VOYAGE,
                properties={"gap_before": False},
            )
            segments.append(segment)

        return segments

    def _detect_port_visits(self, df: pd.DataFrame) -> list[TrajectorySegment]:
        """Detect port visit segments"""
        segments = []

        if "in_port" not in df.columns:
            return segments

        # Find continuous port visits
        port_changes = df["in_port"].diff().fillna(0) != 0
        port_change_indices = df[port_changes].index.tolist()

        # Add start and end indices
        if df.iloc[0]["in_port"]:
            port_change_indices.insert(0, 0)
        if df.iloc[-1]["in_port"]:
            port_change_indices.append(len(df))

        # Create port visit segments
        for i in range(0, len(port_change_indices) - 1, 2):
            start_idx = port_change_indices[i]
            end_idx = port_change_indices[i + 1] - 1

            if end_idx > start_idx:
                port_name = df.iloc[start_idx]["port_name"]

                segment = TrajectorySegment(
                    mmsi=df.iloc[0]["mmsi"],
                    start_time=df.iloc[start_idx]["timestamp"],
                    end_time=df.iloc[end_idx]["timestamp"],
                    start_idx=start_idx,
                    end_idx=end_idx,
                    segment_type=SegmentType.PORT_VISIT,
                    properties={
                        "port_name": port_name,
                        "avg_speed": df.iloc[start_idx : end_idx + 1]["speed"].mean(),
                    },
                )
                segments.append(segment)

        return segments

    def _detect_anchoring_periods(self, df: pd.DataFrame) -> list[TrajectorySegment]:
        """Detect anchoring periods based on low speed and navigation state"""
        segments = []

        # Identify anchoring conditions
        low_speed = df["speed"] < self.anchoring_speed_threshold

        # Check navigation state if available
        if "navigation_state" in df.columns:
            anchored_state = df["navigation_state"].isin(
                [NavigationState.AT_ANCHOR.value, NavigationState.MOORED.value]
            )
            anchoring_condition = low_speed | anchored_state
        else:
            anchoring_condition = low_speed

        # Find continuous anchoring periods
        anchoring_changes = anchoring_condition.diff().fillna(0) != 0
        change_indices = df[anchoring_changes].index.tolist()

        # Add start and end indices
        if df.iloc[0]["speed"] < self.anchoring_speed_threshold:
            change_indices.insert(0, 0)
        if df.iloc[-1]["speed"] < self.anchoring_speed_threshold:
            change_indices.append(len(df))

        # Create anchoring segments (only if long enough)
        for i in range(0, len(change_indices) - 1, 2):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1] - 1

            if end_idx > start_idx:
                duration = (
                    df.iloc[end_idx]["timestamp"] - df.iloc[start_idx]["timestamp"]
                ).total_seconds()

                if duration >= self.anchoring_time_threshold:
                    segment = TrajectorySegment(
                        mmsi=df.iloc[0]["mmsi"],
                        start_time=df.iloc[start_idx]["timestamp"],
                        end_time=df.iloc[end_idx]["timestamp"],
                        start_idx=start_idx,
                        end_idx=end_idx,
                        segment_type=SegmentType.ANCHORING,
                        properties={
                            "avg_speed": df.iloc[start_idx : end_idx + 1][
                                "speed"
                            ].mean(),
                            "duration_hours": duration / 3600.0,
                        },
                    )
                    segments.append(segment)

        return segments

    def _detect_maneuvering_periods(self, df: pd.DataFrame) -> list[TrajectorySegment]:
        """Detect maneuvering periods based on course changes and speed"""
        segments = []

        if "course_change" not in df.columns:
            return segments

        # Identify maneuvering conditions
        high_course_change = (
            df["course_change"] > self.maneuvering_course_change_threshold
        )
        low_speed = df["speed"] < self.maneuvering_speed_threshold

        maneuvering_condition = high_course_change & low_speed

        # Find continuous maneuvering periods
        maneuvering_changes = maneuvering_condition.diff().fillna(0) != 0
        change_indices = df[maneuvering_changes].index.tolist()

        # Create maneuvering segments
        for i in range(0, len(change_indices) - 1, 2):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1] - 1

            if end_idx > start_idx:
                segment = TrajectorySegment(
                    mmsi=df.iloc[0]["mmsi"],
                    start_time=df.iloc[start_idx]["timestamp"],
                    end_time=df.iloc[end_idx]["timestamp"],
                    start_idx=start_idx,
                    end_idx=end_idx,
                    segment_type=SegmentType.MANEUVERING,
                    properties={
                        "avg_course_change": df.iloc[start_idx : end_idx + 1][
                            "course_change"
                        ].mean(),
                        "avg_speed": df.iloc[start_idx : end_idx + 1]["speed"].mean(),
                    },
                )
                segments.append(segment)

        return segments

    def _detect_behavior_changes(self, df: pd.DataFrame) -> list[TrajectorySegment]:
        """Detect segments based on significant behavior changes"""
        segments = []

        # Detect speed changes
        speed_changes = np.abs(df["speed"].diff()) > self.speed_threshold

        # Detect course changes
        course_changes = df["course_change"] > self.course_threshold

        # Combine behavior changes
        behavior_changes = speed_changes | course_changes
        change_indices = df[behavior_changes].index.tolist()

        # Create behavior-based segments
        start_idx = 0
        for change_idx in change_indices:
            if change_idx > start_idx:
                segment = TrajectorySegment(
                    mmsi=df.iloc[0]["mmsi"],
                    start_time=df.iloc[start_idx]["timestamp"],
                    end_time=df.iloc[change_idx - 1]["timestamp"],
                    start_idx=start_idx,
                    end_idx=change_idx - 1,
                    segment_type=SegmentType.TRANSIT,
                    properties={
                        "avg_speed": df.iloc[start_idx:change_idx]["speed"].mean(),
                        "speed_std": df.iloc[start_idx:change_idx]["speed"].std(),
                    },
                )
                segments.append(segment)
            start_idx = change_idx

        # Add final segment
        if start_idx < len(df):
            segment = TrajectorySegment(
                mmsi=df.iloc[0]["mmsi"],
                start_time=df.iloc[start_idx]["timestamp"],
                end_time=df.iloc[-1]["timestamp"],
                start_idx=start_idx,
                end_idx=len(df) - 1,
                segment_type=SegmentType.TRANSIT,
                properties={
                    "avg_speed": df.iloc[start_idx:]["speed"].mean(),
                    "speed_std": df.iloc[start_idx:]["speed"].std(),
                },
            )
            segments.append(segment)

        return segments

    def _merge_and_fill_segments(
        self, df: pd.DataFrame, segments: list[TrajectorySegment]
    ) -> list[TrajectorySegment]:
        """Merge overlapping segments and fill gaps"""
        if not segments:
            return []

        # Sort segments by start time
        segments.sort(key=lambda x: x.start_time)

        merged_segments = []
        current_segment = segments[0]

        for next_segment in segments[1:]:
            # Check for overlap or adjacency
            if (
                next_segment.start_time <= current_segment.end_time
                or next_segment.start_idx <= current_segment.end_idx + 1
            ):
                # Merge segments - keep the more specific type
                segment_priority = {
                    SegmentType.PORT_VISIT: 4,
                    SegmentType.ANCHORING: 3,
                    SegmentType.MANEUVERING: 2,
                    SegmentType.TRANSIT: 1,
                    SegmentType.VOYAGE: 0,
                }

                if segment_priority.get(
                    next_segment.segment_type, 0
                ) > segment_priority.get(current_segment.segment_type, 0):
                    segment_type = next_segment.segment_type
                    properties = next_segment.properties
                else:
                    segment_type = current_segment.segment_type
                    properties = current_segment.properties

                # Extend current segment
                current_segment = TrajectorySegment(
                    mmsi=current_segment.mmsi,
                    start_time=current_segment.start_time,
                    end_time=max(current_segment.end_time, next_segment.end_time),
                    start_idx=current_segment.start_idx,
                    end_idx=max(current_segment.end_idx, next_segment.end_idx),
                    segment_type=segment_type,
                    properties=properties,
                )
            else:
                # No overlap, add current segment and move to next
                merged_segments.append(current_segment)
                current_segment = next_segment

        # Add final segment
        merged_segments.append(current_segment)

        # Fill gaps between segments
        filled_segments = []
        for i, segment in enumerate(merged_segments):
            filled_segments.append(segment)

            # Check for gap to next segment
            if i < len(merged_segments) - 1:
                next_segment = merged_segments[i + 1]
                if segment.end_idx + 1 < next_segment.start_idx:
                    # Create gap-filling segment
                    gap_segment = TrajectorySegment(
                        mmsi=segment.mmsi,
                        start_time=df.iloc[segment.end_idx + 1]["timestamp"],
                        end_time=df.iloc[next_segment.start_idx - 1]["timestamp"],
                        start_idx=segment.end_idx + 1,
                        end_idx=next_segment.start_idx - 1,
                        segment_type=SegmentType.TRANSIT,
                        properties={"gap_filler": True},
                    )
                    filled_segments.append(gap_segment)

        return filled_segments

    def add_port_area(self, name: str, lat: float, lon: float, radius_km: float):
        """Add a port area for segmentation"""
        center_point = Point(lon, lat)
        radius_deg = radius_km / 111.0
        polygon = center_point.buffer(radius_deg)

        port_area = PortArea(
            name=name, polygon=polygon, center=(lat, lon), radius_km=radius_km
        )

        self.port_areas.append(port_area)

    def segment_trajectories(
        self, df: pd.DataFrame
    ) -> dict[int, list[TrajectorySegment]]:
        """
        Segment trajectories for multiple vessels.

        Args:
            df: DataFrame with AIS data for multiple vessels

        Returns:
            Dictionary mapping MMSI to list of segments
        """
        segments_by_mmsi = {}

        for mmsi in df["mmsi"].unique():
            vessel_df = df[df["mmsi"] == mmsi].copy()
            segments = self.segment_trajectory(vessel_df)
            segments_by_mmsi[mmsi] = segments

            logger.info(
                f"Segmented trajectory for MMSI {mmsi}: {len(segments)} segments"
            )

        return segments_by_mmsi

    def get_segment_statistics(
        self, segments: list[TrajectorySegment]
    ) -> dict[str, Any]:
        """Get statistics about trajectory segments"""
        if not segments:
            return {}

        segment_types = [seg.segment_type.value for seg in segments]
        durations = [seg.duration.total_seconds() / 3600.0 for seg in segments]  # hours
        lengths = [seg.length for seg in segments]

        stats = {
            "total_segments": len(segments),
            "segment_types": {
                seg_type: segment_types.count(seg_type)
                for seg_type in set(segment_types)
            },
            "avg_duration_hours": np.mean(durations),
            "avg_length_points": np.mean(lengths),
            "total_duration_hours": sum(durations),
            "duration_distribution": {
                "min": np.min(durations),
                "max": np.max(durations),
                "std": np.std(durations),
            },
        }

        return stats
