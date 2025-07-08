"""
Scene Aggregation Module

This module provides functionality for aggregating AIS data into maritime scenes:
- Navigation scene classification (port approach/departure, transit, anchoring)
- Multi-vessel interaction scene identification
- Traffic density and congestion analysis
- Waterway and maritime area scene aggregation
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


class SceneType(Enum):
    """Types of maritime scenes"""

    PORT_APPROACH = "port_approach"
    PORT_DEPARTURE = "port_departure"
    TRANSIT = "transit"
    ANCHORING = "anchoring"
    CONGESTION = "congestion"
    INTERACTION = "interaction"
    WATERWAY = "waterway"
    OPEN_WATER = "open_water"


class InteractionType(Enum):
    """Types of vessel interactions"""

    OVERTAKING = "overtaking"
    CROSSING = "crossing"
    HEAD_ON = "head_on"
    PARALLEL = "parallel"
    CONVERGING = "converging"
    DIVERGING = "diverging"
    FOLLOWING = "following"


@dataclass
class VesselState:
    """Represents a vessel state at a specific time"""

    mmsi: int
    timestamp: pd.Timestamp
    latitude: float
    longitude: float
    speed: float
    course: float
    heading: float | None = None
    navigation_state: int | None = None
    vessel_type: int | None = None
    length: float | None = None
    width: float | None = None


@dataclass
class Scene:
    """Represents a maritime scene"""

    scene_id: str
    scene_type: SceneType
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    center_lat: float
    center_lon: float
    radius_km: float
    vessels: list[int]  # MMSIs
    vessel_states: list[VesselState]
    properties: dict[str, Any]

    @property
    def duration(self) -> pd.Timedelta:
        return self.end_time - self.start_time

    @property
    def vessel_count(self) -> int:
        return len(self.vessels)


@dataclass
class WaterwayArea:
    """Represents a waterway area"""

    name: str
    polygon: Polygon
    waterway_type: str  # "channel", "strait", "river", "canal"
    traffic_separation: bool = False
    speed_limit: float | None = None


@dataclass
class InteractionEvent:
    """Represents a vessel interaction event"""

    event_id: str
    interaction_type: InteractionType
    vessel_1: int
    vessel_2: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    min_distance: float  # km
    closest_point_time: pd.Timestamp
    risk_level: float  # 0-1 scale
    properties: dict[str, Any]


class SceneAggregator:
    """
    Aggregates AIS data into maritime scenes for analysis and model training.
    """

    def __init__(
        self,
        scene_radius_km: float = 10.0,
        min_scene_duration_minutes: float = 10.0,
        max_scene_duration_minutes: float = 120.0,
        min_vessels_per_scene: int = 2,
        congestion_density_threshold: float = 5.0,  # vessels per km²
        interaction_distance_threshold: float = 2.0,  # km
        port_approach_distance_km: float = 15.0,
        speed_threshold_knots: float = 1.0,
    ):
        self.scene_radius_km = scene_radius_km
        self.min_scene_duration = timedelta(minutes=min_scene_duration_minutes)
        self.max_scene_duration = timedelta(minutes=max_scene_duration_minutes)
        self.min_vessels_per_scene = min_vessels_per_scene
        self.congestion_density_threshold = congestion_density_threshold
        self.interaction_distance_threshold = interaction_distance_threshold
        self.port_approach_distance_km = port_approach_distance_km
        self.speed_threshold_knots = speed_threshold_knots

        # Initialize waterway areas
        self.waterway_areas = self._initialize_default_waterways()

        # Initialize port areas (from trajectory segmenter)
        self.port_areas = self._initialize_default_ports()

    def _initialize_default_waterways(self) -> list[WaterwayArea]:
        """Initialize default waterway areas"""
        waterways = []

        # Example waterways - in practice, these would be loaded from a database
        waterway_configs = [
            {
                "name": "English Channel",
                "type": "strait",
                "bounds": [(49.0, -5.0), (51.5, 2.0)],
                "traffic_separation": True,
            },
            {
                "name": "Dover Strait",
                "type": "strait",
                "bounds": [(50.8, 1.2), (51.2, 1.8)],
                "traffic_separation": True,
                "speed_limit": 12.0,
            },
            {
                "name": "Elbe River",
                "type": "river",
                "bounds": [(53.3, 8.5), (53.8, 9.2)],
                "traffic_separation": False,
            },
        ]

        for config in waterway_configs:
            # Create rectangular polygon from bounds
            bounds = config["bounds"]
            polygon = Polygon(
                [
                    (bounds[0][1], bounds[0][0]),  # lon, lat
                    (bounds[1][1], bounds[0][0]),
                    (bounds[1][1], bounds[1][0]),
                    (bounds[0][1], bounds[1][0]),
                ]
            )

            waterway = WaterwayArea(
                name=config["name"],
                polygon=polygon,
                waterway_type=config["type"],
                traffic_separation=config.get("traffic_separation", False),
                speed_limit=config.get("speed_limit"),
            )
            waterways.append(waterway)

        return waterways

    def _initialize_default_ports(self) -> list[dict[str, Any]]:
        """Initialize default port areas"""
        return [
            {"name": "Hamburg", "lat": 53.5511, "lon": 9.9937, "radius": 15.0},
            {"name": "Rotterdam", "lat": 51.9244, "lon": 4.4777, "radius": 20.0},
            {"name": "Antwerp", "lat": 51.2194, "lon": 4.4025, "radius": 12.0},
            {"name": "Bremen", "lat": 53.0793, "lon": 8.8017, "radius": 10.0},
        ]

    def aggregate_scenes(self, df: pd.DataFrame) -> list[Scene]:
        """
        Aggregate AIS data into maritime scenes.

        Args:
            df: DataFrame with AIS data

        Returns:
            List of maritime scenes
        """
        logger.info(f"Aggregating scenes from {len(df)} AIS records")

        # Ensure data is sorted by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Convert to vessel states
        vessel_states = self._create_vessel_states(df)

        # Group vessel states by time windows
        time_windows = self._create_time_windows(vessel_states)

        # Identify different types of scenes
        scenes = []

        for window_start, window_states in time_windows.items():
            # Skip windows with insufficient vessels
            if (
                len({state.mmsi for state in window_states})
                < self.min_vessels_per_scene
            ):
                continue

            # Identify spatial clusters
            spatial_clusters = self._identify_spatial_clusters(window_states)

            for cluster in spatial_clusters:
                # Classify scene type
                scene_type = self._classify_scene_type(cluster)

                # Create scene
                scene = self._create_scene(
                    scene_type=scene_type,
                    vessel_states=cluster,
                    window_start=window_start,
                )

                if scene:
                    scenes.append(scene)

        # Merge temporal scenes
        merged_scenes = self._merge_temporal_scenes(scenes)

        # Detect interaction events
        interaction_events = self._detect_interaction_events(df)

        # Add interaction information to scenes
        enriched_scenes = self._enrich_scenes_with_interactions(
            merged_scenes, interaction_events
        )

        logger.info(f"Generated {len(enriched_scenes)} maritime scenes")
        return enriched_scenes

    def _create_vessel_states(self, df: pd.DataFrame) -> list[VesselState]:
        """Convert DataFrame records to VesselState objects"""
        vessel_states = []

        for _, row in df.iterrows():
            state = VesselState(
                mmsi=row["mmsi"],
                timestamp=row["timestamp"],
                latitude=row["latitude"],
                longitude=row["longitude"],
                speed=row.get("speed", 0.0),
                course=row.get("course", 0.0),
                heading=row.get("heading"),
                navigation_state=row.get("navigation_state"),
                vessel_type=row.get("vessel_type"),
                length=row.get("length"),
                width=row.get("width"),
            )
            vessel_states.append(state)

        return vessel_states

    def _create_time_windows(
        self, vessel_states: list[VesselState]
    ) -> dict[pd.Timestamp, list[VesselState]]:
        """Group vessel states into time windows"""
        time_windows = defaultdict(list)

        # Use 5-minute time windows
        window_size = timedelta(minutes=5)

        for state in vessel_states:
            # Round timestamp to nearest window
            window_start = state.timestamp.floor(window_size)
            time_windows[window_start].append(state)

        return dict(time_windows)

    def _identify_spatial_clusters(
        self, vessel_states: list[VesselState]
    ) -> list[list[VesselState]]:
        """Identify spatial clusters of vessels"""
        clusters = []
        remaining_states = vessel_states.copy()

        while remaining_states:
            # Start new cluster with first remaining state
            seed_state = remaining_states.pop(0)
            cluster = [seed_state]

            # Find nearby vessels
            nearby_states = []
            for state in remaining_states:
                distance = geodesic(
                    (seed_state.latitude, seed_state.longitude),
                    (state.latitude, state.longitude),
                ).kilometers

                if distance <= self.scene_radius_km:
                    nearby_states.append(state)

            # Add nearby vessels to cluster
            for state in nearby_states:
                cluster.append(state)
                remaining_states.remove(state)

            # Only keep clusters with minimum vessel count
            if len({state.mmsi for state in cluster}) >= self.min_vessels_per_scene:
                clusters.append(cluster)

        return clusters

    def _classify_scene_type(self, vessel_states: list[VesselState]) -> SceneType:
        """Classify the type of maritime scene"""
        # Calculate cluster center
        center_lat = np.mean([state.latitude for state in vessel_states])
        center_lon = np.mean([state.longitude for state in vessel_states])

        # Check if near port
        near_port = self._is_near_port(center_lat, center_lon)

        # Check if in waterway
        in_waterway = self._is_in_waterway(center_lat, center_lon)

        # Calculate vessel density
        area_km2 = np.pi * (self.scene_radius_km**2)
        vessel_count = len({state.mmsi for state in vessel_states})
        density = vessel_count / area_km2

        # Calculate average speed
        avg_speed = np.mean([state.speed for state in vessel_states])

        # Classification logic
        if density > self.congestion_density_threshold:
            return SceneType.CONGESTION
        elif near_port:
            if avg_speed < self.speed_threshold_knots:
                return SceneType.ANCHORING
            else:
                # Determine if approaching or departing based on movement patterns
                return self._classify_port_scene(vessel_states, center_lat, center_lon)
        elif in_waterway:
            return SceneType.WATERWAY
        elif avg_speed < self.speed_threshold_knots:
            return SceneType.ANCHORING
        else:
            return SceneType.TRANSIT

    def _is_near_port(self, lat: float, lon: float) -> bool:
        """Check if location is near a port"""
        for port in self.port_areas:
            distance = geodesic((lat, lon), (port["lat"], port["lon"])).kilometers
            if distance <= port["radius"]:
                return True
        return False

    def _is_in_waterway(self, lat: float, lon: float) -> bool:
        """Check if location is in a waterway"""
        point = Point(lon, lat)
        return any(waterway.polygon.contains(point) for waterway in self.waterway_areas)

    def _classify_port_scene(
        self, vessel_states: list[VesselState], center_lat: float, center_lon: float
    ) -> SceneType:
        """Classify port-related scene as approach or departure"""
        # Find nearest port
        nearest_port = None
        min_distance = float("inf")

        for port in self.port_areas:
            distance = geodesic(
                (center_lat, center_lon), (port["lat"], port["lon"])
            ).kilometers
            if distance < min_distance:
                min_distance = distance
                nearest_port = port

        if not nearest_port:
            return SceneType.TRANSIT

        # Analyze vessel movements relative to port
        approaching_count = 0
        departing_count = 0

        for state in vessel_states:
            # Calculate bearing to port
            port_bearing = self._calculate_bearing(
                state.latitude,
                state.longitude,
                nearest_port["lat"],
                nearest_port["lon"],
            )

            # Compare with vessel course
            course_diff = abs(state.course - port_bearing)
            if course_diff > 180:
                course_diff = 360 - course_diff

            if course_diff < 45:  # Heading towards port
                approaching_count += 1
            elif course_diff > 135:  # Heading away from port
                departing_count += 1

        if approaching_count > departing_count:
            return SceneType.PORT_APPROACH
        elif departing_count > approaching_count:
            return SceneType.PORT_DEPARTURE
        else:
            return SceneType.TRANSIT

    def _calculate_bearing(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate bearing between two points"""
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlon_rad = np.radians(lon2 - lon1)

        y = np.sin(dlon_rad) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(
            lat2_rad
        ) * np.cos(dlon_rad)

        bearing_rad = np.arctan2(y, x)
        bearing_deg = np.degrees(bearing_rad)

        return (bearing_deg + 360) % 360

    def _create_scene(
        self,
        scene_type: SceneType,
        vessel_states: list[VesselState],
        window_start: pd.Timestamp,
    ) -> Scene | None:
        """Create a scene from vessel states"""
        if not vessel_states:
            return None

        # Calculate scene properties
        center_lat = np.mean([state.latitude for state in vessel_states])
        center_lon = np.mean([state.longitude for state in vessel_states])
        vessels = list({state.mmsi for state in vessel_states})

        # Calculate scene duration
        timestamps = [state.timestamp for state in vessel_states]
        start_time = min(timestamps)
        end_time = max(timestamps)

        # Generate scene ID
        scene_id = (
            f"{scene_type.value}_{window_start.strftime('%Y%m%d_%H%M')}_{len(vessels)}v"
        )

        # Calculate additional properties
        properties = {
            "avg_speed": np.mean([state.speed for state in vessel_states]),
            "max_speed": np.max([state.speed for state in vessel_states]),
            "speed_std": np.std([state.speed for state in vessel_states]),
            "vessel_types": list(
                {state.vessel_type for state in vessel_states if state.vessel_type}
            ),
            "navigation_states": list(
                {
                    state.navigation_state
                    for state in vessel_states
                    if state.navigation_state
                }
            ),
            "density": len(vessels) / (np.pi * (self.scene_radius_km**2)),
        }

        scene = Scene(
            scene_id=scene_id,
            scene_type=scene_type,
            start_time=start_time,
            end_time=end_time,
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=self.scene_radius_km,
            vessels=vessels,
            vessel_states=vessel_states,
            properties=properties,
        )

        return scene

    def _merge_temporal_scenes(self, scenes: list[Scene]) -> list[Scene]:
        """Merge scenes that are temporally adjacent"""
        if not scenes:
            return []

        # Sort scenes by start time
        scenes.sort(key=lambda x: x.start_time)

        merged_scenes = []
        current_scene = scenes[0]

        for next_scene in scenes[1:]:
            # Check if scenes can be merged
            if (
                next_scene.scene_type == current_scene.scene_type
                and next_scene.start_time - current_scene.end_time
                <= self.min_scene_duration
                and self._scenes_spatially_overlap(current_scene, next_scene)
            ):
                # Merge scenes
                current_scene = self._merge_two_scenes(current_scene, next_scene)
            else:
                # Cannot merge, add current scene and move to next
                merged_scenes.append(current_scene)
                current_scene = next_scene

        # Add final scene
        merged_scenes.append(current_scene)

        return merged_scenes

    def _scenes_spatially_overlap(self, scene1: Scene, scene2: Scene) -> bool:
        """Check if two scenes spatially overlap"""
        distance = geodesic(
            (scene1.center_lat, scene1.center_lon),
            (scene2.center_lat, scene2.center_lon),
        ).kilometers

        return distance <= (scene1.radius_km + scene2.radius_km)

    def _merge_two_scenes(self, scene1: Scene, scene2: Scene) -> Scene:
        """Merge two scenes into one"""
        # Combine vessel states
        combined_states = scene1.vessel_states + scene2.vessel_states
        combined_vessels = list(set(scene1.vessels + scene2.vessels))

        # Calculate new center (weighted by vessel count)
        w1 = len(scene1.vessels)
        w2 = len(scene2.vessels)
        total_weight = w1 + w2

        new_center_lat = (
            scene1.center_lat * w1 + scene2.center_lat * w2
        ) / total_weight
        new_center_lon = (
            scene1.center_lon * w1 + scene2.center_lon * w2
        ) / total_weight

        # Calculate new time range
        new_start_time = min(scene1.start_time, scene2.start_time)
        new_end_time = max(scene1.end_time, scene2.end_time)

        # Generate new scene ID
        new_scene_id = f"{scene1.scene_type.value}_{new_start_time.strftime('%Y%m%d_%H%M')}_{len(combined_vessels)}v_merged"

        # Combine properties
        combined_properties = scene1.properties.copy()
        combined_properties.update(
            {
                "merged_from": [scene1.scene_id, scene2.scene_id],
                "avg_speed": np.mean([state.speed for state in combined_states]),
                "max_speed": np.max([state.speed for state in combined_states]),
                "speed_std": np.std([state.speed for state in combined_states]),
                "density": len(combined_vessels)
                / (np.pi * (self.scene_radius_km**2)),
            }
        )

        merged_scene = Scene(
            scene_id=new_scene_id,
            scene_type=scene1.scene_type,
            start_time=new_start_time,
            end_time=new_end_time,
            center_lat=new_center_lat,
            center_lon=new_center_lon,
            radius_km=self.scene_radius_km,
            vessels=combined_vessels,
            vessel_states=combined_states,
            properties=combined_properties,
        )

        return merged_scene

    def _detect_interaction_events(self, df: pd.DataFrame) -> list[InteractionEvent]:
        """Detect vessel interaction events"""
        interaction_events = []

        # Group by time windows for efficient processing
        time_windows = df.groupby(df["timestamp"].dt.floor("5min"))

        for window_time, window_df in time_windows:
            # Get all vessel pairs in this window
            vessels = window_df["mmsi"].unique()

            for i, vessel1 in enumerate(vessels):
                for vessel2 in vessels[i + 1 :]:
                    # Get vessel data
                    v1_data = window_df[window_df["mmsi"] == vessel1]
                    v2_data = window_df[window_df["mmsi"] == vessel2]

                    if len(v1_data) == 0 or len(v2_data) == 0:
                        continue

                    # Calculate distances between vessels
                    interaction_event = self._analyze_vessel_interaction(
                        v1_data, v2_data, window_time
                    )

                    if interaction_event:
                        interaction_events.append(interaction_event)

        return interaction_events

    def _analyze_vessel_interaction(
        self, v1_data: pd.DataFrame, v2_data: pd.DataFrame, window_time: pd.Timestamp
    ) -> InteractionEvent | None:
        """Analyze interaction between two vessels"""
        # Find closest approach
        min_distance = float("inf")
        closest_time = None

        for _, row1 in v1_data.iterrows():
            for _, row2 in v2_data.iterrows():
                distance = geodesic(
                    (row1["latitude"], row1["longitude"]),
                    (row2["latitude"], row2["longitude"]),
                ).kilometers

                if distance < min_distance:
                    min_distance = distance
                    closest_time = row1["timestamp"]

        # Only consider close interactions
        if min_distance > self.interaction_distance_threshold:
            return None

        # Classify interaction type
        interaction_type = self._classify_interaction_type(v1_data, v2_data)

        # Calculate risk level
        risk_level = self._calculate_risk_level(min_distance, v1_data, v2_data)

        # Generate event ID
        event_id = f"interaction_{window_time.strftime('%Y%m%d_%H%M')}_{v1_data.iloc[0]['mmsi']}_{v2_data.iloc[0]['mmsi']}"

        interaction_event = InteractionEvent(
            event_id=event_id,
            interaction_type=interaction_type,
            vessel_1=v1_data.iloc[0]["mmsi"],
            vessel_2=v2_data.iloc[0]["mmsi"],
            start_time=window_time,
            end_time=window_time + timedelta(minutes=5),
            min_distance=min_distance,
            closest_point_time=closest_time,
            risk_level=risk_level,
            properties={
                "v1_avg_speed": v1_data["speed"].mean(),
                "v2_avg_speed": v2_data["speed"].mean(),
                "v1_course": v1_data["course"].mean(),
                "v2_course": v2_data["course"].mean(),
            },
        )

        return interaction_event

    def _classify_interaction_type(
        self, v1_data: pd.DataFrame, v2_data: pd.DataFrame
    ) -> InteractionType:
        """Classify the type of vessel interaction"""
        # Calculate average courses
        v1_course = v1_data["course"].mean()
        v2_course = v2_data["course"].mean()

        # Calculate course difference
        course_diff = abs(v1_course - v2_course)
        if course_diff > 180:
            course_diff = 360 - course_diff

        # Calculate average speeds
        v1_speed = v1_data["speed"].mean()
        v2_speed = v2_data["speed"].mean()

        # Classification logic
        if course_diff < 15:  # Similar courses
            if abs(v1_speed - v2_speed) > 2:  # Speed difference
                return InteractionType.OVERTAKING
            else:
                return InteractionType.PARALLEL
        elif course_diff > 165:  # Opposite courses
            return InteractionType.HEAD_ON
        elif 45 <= course_diff <= 135:  # Perpendicular courses
            return InteractionType.CROSSING
        elif course_diff < 45:
            return InteractionType.CONVERGING
        else:
            return InteractionType.DIVERGING

    def _calculate_risk_level(
        self, min_distance: float, v1_data: pd.DataFrame, v2_data: pd.DataFrame
    ) -> float:
        """Calculate risk level for vessel interaction"""
        # Base risk from distance (closer = higher risk)
        distance_risk = max(0, (2.0 - min_distance) / 2.0)

        # Speed risk (faster vessels = higher risk)
        max_speed = max(v1_data["speed"].max(), v2_data["speed"].max())
        speed_risk = min(1.0, max_speed / 20.0)  # Normalize to 0-1

        # Combined risk
        risk_level = min(1.0, (distance_risk + speed_risk) / 2.0)

        return risk_level

    def _enrich_scenes_with_interactions(
        self, scenes: list[Scene], interaction_events: list[InteractionEvent]
    ) -> list[Scene]:
        """Add interaction information to scenes"""
        enriched_scenes = []

        for scene in scenes:
            # Find interactions that overlap with this scene
            scene_interactions = []

            for interaction in interaction_events:
                # Check temporal overlap
                if (
                    interaction.start_time <= scene.end_time
                    and interaction.end_time >= scene.start_time
                ):
                    # Check if vessels are in the scene
                    if (
                        interaction.vessel_1 in scene.vessels
                        or interaction.vessel_2 in scene.vessels
                    ):
                        scene_interactions.append(interaction)

            # Add interaction information to scene properties
            scene.properties["interactions"] = len(scene_interactions)
            scene.properties["interaction_events"] = [
                {
                    "event_id": event.event_id,
                    "type": event.interaction_type.value,
                    "risk_level": event.risk_level,
                    "min_distance": event.min_distance,
                }
                for event in scene_interactions
            ]

            # Update scene type if high interaction
            if len(scene_interactions) > 0:
                avg_risk = np.mean([event.risk_level for event in scene_interactions])
                if avg_risk > 0.7:
                    scene.scene_type = SceneType.INTERACTION

            enriched_scenes.append(scene)

        return enriched_scenes

    def get_scene_statistics(self, scenes: list[Scene]) -> dict[str, Any]:
        """Get statistics about generated scenes"""
        if not scenes:
            return {}

        scene_types = [scene.scene_type.value for scene in scenes]
        durations = [
            scene.duration.total_seconds() / 3600.0 for scene in scenes
        ]  # hours
        vessel_counts = [scene.vessel_count for scene in scenes]

        stats = {
            "total_scenes": len(scenes),
            "scene_types": {
                scene_type: scene_types.count(scene_type)
                for scene_type in set(scene_types)
            },
            "avg_duration_hours": np.mean(durations),
            "avg_vessel_count": np.mean(vessel_counts),
            "total_duration_hours": sum(durations),
            "duration_distribution": {
                "min": np.min(durations),
                "max": np.max(durations),
                "std": np.std(durations),
            },
            "vessel_count_distribution": {
                "min": np.min(vessel_counts),
                "max": np.max(vessel_counts),
                "std": np.std(vessel_counts),
            },
        }

        return stats
