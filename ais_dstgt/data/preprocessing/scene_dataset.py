"""
Scene Dataset Module

This module provides PyTorch Dataset implementation for maritime scenes:
- Just-in-time processing and feature extraction
- Dynamic adjacency matrix construction
- Support for efficient data loading with multiprocessing
- Custom collate function for batching variable-sized scenes
"""

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class SceneFeatures:
    """Features extracted from a maritime scene"""

    # Node features (vessels)
    node_features: torch.Tensor  # [n_vessels, n_node_features]
    node_positions: torch.Tensor  # [n_vessels, 2] (x, y coordinates)
    node_velocities: torch.Tensor  # [n_vessels, 2] (vx, vy)
    node_types: torch.Tensor  # [n_vessels] vessel type indices

    # Edge features (interactions)
    edge_index: torch.Tensor  # [2, n_edges] adjacency matrix in COO format
    edge_features: torch.Tensor  # [n_edges, n_edge_features]

    # Graph-level features
    graph_features: torch.Tensor  # [n_graph_features]

    # Temporal features
    history_features: torch.Tensor  # [n_vessels, history_length, n_features]
    future_features: torch.Tensor  # [n_vessels, future_length, n_features]

    # Static scene information
    scene_type: int
    timestamp: datetime
    vessel_count: int
    scene_id: str


@dataclass
class DatasetConfig:
    """Configuration for scene dataset"""

    # Feature extraction - 基于科研需求优化的时间窗口参数
    history_length: int = 20  # 历史时间步数，20分钟历史轨迹，覆盖两轮避碰决策
    future_length: int = 10  # 未来时间步数，10分钟预测窗口，符合预测文献标准
    time_step_seconds: int = 60  # 时间步长，60秒间隔与AIS采样频率对齐

    # Node features
    node_feature_names: list[str] = None
    normalize_features: bool = True

    # Edge features - 基于COLREGS规则的交互距离
    edge_distance_threshold: float = 2000.0  # 边连接距离阈值（米），约1海里
    include_edge_features: bool = True

    # Graph features
    include_graph_features: bool = True

    # Data processing - 场景规模控制
    min_vessel_count: int = 2  # 最小船舶数，满足交互场景定义
    max_vessel_count: int = 50  # 最大船舶数，平衡显存与场景完整性
    filter_scene_types: list[str] | None = None

    def __post_init__(self):
        if self.node_feature_names is None:
            self.node_feature_names = [
                "speed",
                "course",
                "acceleration",
                "course_change",
                "distance_to_center",
                "relative_bearing",
            ]


class SceneDataset(Dataset):
    """
    PyTorch Dataset for maritime scenes with just-in-time processing.
    """

    def __init__(
        self,
        scenes_data: list[dict] | str | Path,
        vessel_trajectories: pd.DataFrame | str | Path,
        config: DatasetConfig = None,
        cache_dir: str | None = None,
        precompute_features: bool = False,
    ):
        self.config = config or DatasetConfig()
        self.cache_dir = cache_dir
        self.precompute_features = precompute_features

        # Load data
        self.scenes_data = self._load_scenes_data(scenes_data)
        self.vessel_trajectories = self._load_trajectories(vessel_trajectories)

        # Filter scenes based on config
        self.valid_scenes = self._filter_scenes()

        # Feature normalization parameters
        self.feature_stats = None
        if self.config.normalize_features:
            self._compute_feature_statistics()

        # Precompute features if requested
        if self.precompute_features:
            self._precompute_all_features()

        logger.info(f"Initialized SceneDataset with {len(self.valid_scenes)} scenes")

    def __len__(self) -> int:
        return len(self.valid_scenes)

    def __getitem__(self, idx: int) -> SceneFeatures:
        """Get scene features for a given index"""
        scene_data = self.valid_scenes[idx]

        # Check cache first
        if self.cache_dir:
            cached_features = self._load_cached_features(scene_data["scene_id"])
            if cached_features is not None:
                return cached_features

        # Extract features
        features = self._extract_scene_features(scene_data)

        # Cache features if enabled
        if self.cache_dir:
            self._save_cached_features(scene_data["scene_id"], features)

        return features

    def _load_scenes_data(
        self, scenes_data: list[dict] | str | Path
    ) -> list[dict]:
        """Load scenes data from various sources"""
        if isinstance(scenes_data, str | Path):
            path = Path(scenes_data)
            if path.suffix == ".pkl":
                with open(path, "rb") as f:
                    return pickle.load(f)
            elif path.suffix == ".csv":
                df = pd.read_csv(path)
                return df.to_dict("records")
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            return scenes_data

    def _load_trajectories(
        self, trajectories: pd.DataFrame | str | Path
    ) -> pd.DataFrame:
        """Load vessel trajectories"""
        if isinstance(trajectories, pd.DataFrame):
            return trajectories
        elif isinstance(trajectories, str | Path):
            return pd.read_csv(trajectories, parse_dates=["timestamp"])
        else:
            raise ValueError("Invalid trajectories format")

    def _filter_scenes(self) -> list[dict]:
        """Filter scenes based on configuration"""
        filtered_scenes = []

        for scene in self.scenes_data:
            # Filter by vessel count
            vessel_count = scene.get("vessel_count", 0)
            if not (
                self.config.min_vessel_count
                <= vessel_count
                <= self.config.max_vessel_count
            ):
                continue

            # Filter by scene type
            if self.config.filter_scene_types:
                scene_type = scene.get("scene_type", "")
                if scene_type not in self.config.filter_scene_types:
                    continue

            # Check if we have trajectory data for vessels in this scene
            vessel_mmsis = scene.get("vessels", [])
            if not vessel_mmsis:
                continue

            # Check if trajectory data exists
            scene_trajectories = self.vessel_trajectories[
                self.vessel_trajectories["mmsi"].isin(vessel_mmsis)
            ]
            if len(scene_trajectories) == 0:
                continue

            filtered_scenes.append(scene)

        return filtered_scenes

    def _extract_scene_features(self, scene_data: dict) -> SceneFeatures:
        """Extract features for a single scene"""
        scene_id = scene_data["scene_id"]
        vessels = scene_data["vessels"]
        scene_timestamp = pd.to_datetime(scene_data["start_time"])

        # Get trajectory data for vessels in this scene
        scene_trajectories = self._get_scene_trajectories(vessels, scene_timestamp)

        if len(scene_trajectories) == 0:
            raise ValueError(f"No trajectory data found for scene {scene_id}")

        # Extract node features
        (
            node_features,
            node_positions,
            node_velocities,
            node_types,
        ) = self._extract_node_features(scene_trajectories, scene_timestamp)

        # Extract edge features
        edge_index, edge_features = self._extract_edge_features(
            node_positions, node_velocities, scene_trajectories
        )

        # Extract graph features
        graph_features = self._extract_graph_features(scene_data, scene_trajectories)

        # Extract temporal features
        history_features, future_features = self._extract_temporal_features(
            vessels, scene_timestamp
        )

        # Create scene features object
        features = SceneFeatures(
            node_features=node_features,
            node_positions=node_positions,
            node_velocities=node_velocities,
            node_types=node_types,
            edge_index=edge_index,
            edge_features=edge_features,
            graph_features=graph_features,
            history_features=history_features,
            future_features=future_features,
            scene_type=self._encode_scene_type(scene_data.get("scene_type", "unknown")),
            timestamp=scene_timestamp,
            vessel_count=len(vessels),
            scene_id=scene_id,
        )

        return features

    def _get_scene_trajectories(
        self, vessels: list[int], timestamp: pd.Timestamp
    ) -> pd.DataFrame:
        """Get trajectory data for vessels around scene timestamp"""
        # Time window around scene
        time_window = timedelta(minutes=30)  # ±30 minutes
        start_time = timestamp - time_window
        end_time = timestamp + time_window

        # Filter trajectories
        scene_trajectories = self.vessel_trajectories[
            (self.vessel_trajectories["mmsi"].isin(vessels))
            & (self.vessel_trajectories["timestamp"] >= start_time)
            & (self.vessel_trajectories["timestamp"] <= end_time)
        ].copy()

        return scene_trajectories.sort_values(["mmsi", "timestamp"])

    def _extract_node_features(
        self, trajectories: pd.DataFrame, timestamp: pd.Timestamp
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract node (vessel) features"""
        vessels = trajectories["mmsi"].unique()
        n_vessels = len(vessels)
        n_features = len(self.config.node_feature_names)

        node_features = torch.zeros(n_vessels, n_features)
        node_positions = torch.zeros(n_vessels, 2)
        node_velocities = torch.zeros(n_vessels, 2)
        node_types = torch.zeros(n_vessels, dtype=torch.long)

        # Scene center (for relative positioning)
        scene_center_lat = trajectories["latitude"].mean()
        scene_center_lon = trajectories["longitude"].mean()

        for i, vessel_mmsi in enumerate(vessels):
            vessel_data = trajectories[trajectories["mmsi"] == vessel_mmsi]

            # Find closest timestamp
            time_diffs = (vessel_data["timestamp"] - timestamp).abs()
            closest_idx = time_diffs.idxmin()
            vessel_row = vessel_data.loc[closest_idx]

            # Extract basic features
            features = []
            for feature_name in self.config.node_feature_names:
                if feature_name == "speed":
                    features.append(vessel_row.get("speed", 0.0))
                elif feature_name == "course":
                    features.append(vessel_row.get("course", 0.0))
                elif feature_name == "acceleration":
                    features.append(
                        self._calculate_acceleration(vessel_data, closest_idx)
                    )
                elif feature_name == "course_change":
                    features.append(
                        self._calculate_course_change(vessel_data, closest_idx)
                    )
                elif feature_name == "distance_to_center":
                    features.append(
                        self._calculate_distance_to_center(
                            vessel_row, scene_center_lat, scene_center_lon
                        )
                    )
                elif feature_name == "relative_bearing":
                    features.append(
                        self._calculate_relative_bearing(
                            vessel_row, scene_center_lat, scene_center_lon
                        )
                    )
                else:
                    features.append(0.0)  # Default value

            node_features[i] = torch.tensor(features, dtype=torch.float32)

            # Position (local coordinates)
            x, y = self._project_to_local(
                vessel_row["latitude"],
                vessel_row["longitude"],
                scene_center_lat,
                scene_center_lon,
            )
            node_positions[i] = torch.tensor([x, y], dtype=torch.float32)

            # Velocity
            speed_ms = vessel_row.get("speed", 0.0) * 0.514444  # knots to m/s
            course_rad = np.radians(vessel_row.get("course", 0.0))
            vx = speed_ms * np.sin(course_rad)
            vy = speed_ms * np.cos(course_rad)
            node_velocities[i] = torch.tensor([vx, vy], dtype=torch.float32)

            # Vessel type
            vessel_type = vessel_row.get("vessel_type", 0)
            node_types[i] = torch.tensor(vessel_type, dtype=torch.long)

        # Normalize features if enabled
        if self.config.normalize_features and self.feature_stats is not None:
            node_features = self._normalize_features(node_features)

        return node_features, node_positions, node_velocities, node_types

    def _extract_edge_features(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        trajectories: pd.DataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract edge (interaction) features"""
        n_vessels = positions.shape[0]
        edge_list = []
        edge_features_list = []

        # Calculate pairwise distances and create edges
        for i in range(n_vessels):
            for j in range(i + 1, n_vessels):
                # Calculate distance
                pos_i = positions[i]
                pos_j = positions[j]
                distance = torch.norm(pos_i - pos_j).item()

                # Only create edge if within threshold
                if distance <= self.config.edge_distance_threshold:
                    # Add both directions
                    edge_list.extend([[i, j], [j, i]])

                    if self.config.include_edge_features:
                        # Calculate edge features
                        vel_i = velocities[i]
                        vel_j = velocities[j]

                        # Relative velocity
                        rel_velocity = torch.norm(vel_i - vel_j).item()

                        # Approach rate (negative if diverging)
                        rel_pos = pos_j - pos_i
                        rel_vel = vel_j - vel_i
                        approach_rate = -torch.dot(rel_pos, rel_vel).item() / (
                            distance + 1e-6
                        )

                        # Bearing
                        bearing = torch.atan2(rel_pos[0], rel_pos[1]).item()

                        edge_feat = [distance, rel_velocity, approach_rate, bearing]
                        edge_features_list.extend(
                            [edge_feat, edge_feat]
                        )  # Same for both directions
                    else:
                        edge_features_list.extend([[distance], [distance]])

        if not edge_list:
            # No edges - create empty tensors
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_features = torch.zeros(
                0, 1 if not self.config.include_edge_features else 4
            )
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_features = torch.tensor(edge_features_list, dtype=torch.float32)

        return edge_index, edge_features

    def _extract_graph_features(
        self, scene_data: dict, trajectories: pd.DataFrame
    ) -> torch.Tensor:
        """Extract graph-level features"""
        if not self.config.include_graph_features:
            return torch.zeros(1)  # Minimal graph features

        features = []

        # Scene properties
        features.append(scene_data.get("vessel_count", 0))
        features.append(scene_data.get("duration_minutes", 0))
        features.append(scene_data.get("radius_km", 0))

        # Trajectory statistics
        features.append(trajectories["speed"].mean())
        features.append(trajectories["speed"].std())
        features.append(trajectories["course"].std())

        # Density
        area_km2 = np.pi * (scene_data.get("radius_km", 1.0) ** 2)
        density = scene_data.get("vessel_count", 0) / area_km2
        features.append(density)

        return torch.tensor(features, dtype=torch.float32)

    def _extract_temporal_features(
        self, vessels: list[int], timestamp: pd.Timestamp
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract historical and future features"""
        n_vessels = len(vessels)
        n_features = 4  # x, y, vx, vy

        history_features = torch.zeros(
            n_vessels, self.config.history_length, n_features
        )
        future_features = torch.zeros(n_vessels, self.config.future_length, n_features)

        for i, vessel_mmsi in enumerate(vessels):
            vessel_data = self.vessel_trajectories[
                self.vessel_trajectories["mmsi"] == vessel_mmsi
            ].copy()

            if len(vessel_data) == 0:
                continue

            # Sort by time
            vessel_data = vessel_data.sort_values("timestamp")

            # Find reference point
            time_diffs = (vessel_data["timestamp"] - timestamp).abs()
            ref_idx = time_diffs.idxmin()
            ref_position = vessel_data.index.get_loc(ref_idx)

            # Extract history
            history_start = max(0, ref_position - self.config.history_length)
            history_data = vessel_data.iloc[history_start:ref_position]

            for j, (_, row) in enumerate(history_data.iterrows()):
                if j < self.config.history_length:
                    # Convert to local coordinates
                    x, y = self._project_to_local(
                        row["latitude"],
                        row["longitude"],
                        vessel_data.iloc[ref_position]["latitude"],
                        vessel_data.iloc[ref_position]["longitude"],
                    )

                    # Calculate velocity
                    speed_ms = row.get("speed", 0.0) * 0.514444
                    course_rad = np.radians(row.get("course", 0.0))
                    vx = speed_ms * np.sin(course_rad)
                    vy = speed_ms * np.cos(course_rad)

                    history_features[i, j] = torch.tensor([x, y, vx, vy])

            # Extract future
            future_end = min(
                len(vessel_data), ref_position + self.config.future_length + 1
            )
            future_data = vessel_data.iloc[ref_position + 1 : future_end]

            for j, (_, row) in enumerate(future_data.iterrows()):
                if j < self.config.future_length:
                    # Convert to local coordinates
                    x, y = self._project_to_local(
                        row["latitude"],
                        row["longitude"],
                        vessel_data.iloc[ref_position]["latitude"],
                        vessel_data.iloc[ref_position]["longitude"],
                    )

                    # Calculate velocity
                    speed_ms = row.get("speed", 0.0) * 0.514444
                    course_rad = np.radians(row.get("course", 0.0))
                    vx = speed_ms * np.sin(course_rad)
                    vy = speed_ms * np.cos(course_rad)

                    future_features[i, j] = torch.tensor([x, y, vx, vy])

        return history_features, future_features

    def _project_to_local(
        self, lat: float, lon: float, center_lat: float, center_lon: float
    ) -> tuple[float, float]:
        """Project coordinates to local system"""
        R = 6371000.0  # Earth radius

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        center_lat_rad = np.radians(center_lat)
        center_lon_rad = np.radians(center_lon)

        x = R * (lon_rad - center_lon_rad) * np.cos(center_lat_rad)
        y = R * (lat_rad - center_lat_rad)

        return x, y

    def _calculate_acceleration(
        self, vessel_data: pd.DataFrame, current_idx: int
    ) -> float:
        """Calculate acceleration at current index"""
        if len(vessel_data) < 2:
            return 0.0

        vessel_data = vessel_data.sort_values("timestamp")
        positions = list(vessel_data.index)

        if current_idx not in positions:
            return 0.0

        pos = positions.index(current_idx)

        if pos == 0:
            return 0.0

        current_speed = (
            vessel_data.loc[current_idx].get("speed", 0.0) * 0.514444
        )  # knots to m/s
        prev_speed = vessel_data.iloc[pos - 1].get("speed", 0.0) * 0.514444

        time_diff = (
            vessel_data.loc[current_idx]["timestamp"]
            - vessel_data.iloc[pos - 1]["timestamp"]
        ).total_seconds()

        if time_diff > 0:
            return (current_speed - prev_speed) / time_diff

        return 0.0

    def _calculate_course_change(
        self, vessel_data: pd.DataFrame, current_idx: int
    ) -> float:
        """Calculate course change at current index"""
        if len(vessel_data) < 2:
            return 0.0

        vessel_data = vessel_data.sort_values("timestamp")
        positions = list(vessel_data.index)

        if current_idx not in positions:
            return 0.0

        pos = positions.index(current_idx)

        if pos == 0:
            return 0.0

        current_course = vessel_data.loc[current_idx].get("course", 0.0)
        prev_course = vessel_data.iloc[pos - 1].get("course", 0.0)

        course_diff = abs(current_course - prev_course)
        if course_diff > 180:
            course_diff = 360 - course_diff

        return course_diff

    def _calculate_distance_to_center(
        self, vessel_row: pd.Series, center_lat: float, center_lon: float
    ) -> float:
        """Calculate distance to scene center"""
        from geopy.distance import geodesic

        return geodesic(
            (vessel_row["latitude"], vessel_row["longitude"]), (center_lat, center_lon)
        ).meters

    def _calculate_relative_bearing(
        self, vessel_row: pd.Series, center_lat: float, center_lon: float
    ) -> float:
        """Calculate bearing from vessel to scene center"""
        lat1 = np.radians(vessel_row["latitude"])
        lon1 = np.radians(vessel_row["longitude"])
        lat2 = np.radians(center_lat)
        lon2 = np.radians(center_lon)

        dlon = lon2 - lon1

        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

        bearing = np.degrees(np.arctan2(y, x))
        return (bearing + 360) % 360

    def _encode_scene_type(self, scene_type: str) -> int:
        """Encode scene type as integer"""
        scene_type_mapping = {
            "port_approach": 0,
            "port_departure": 1,
            "transit": 2,
            "anchoring": 3,
            "congestion": 4,
            "interaction": 5,
            "waterway": 6,
            "open_water": 7,
            "unknown": 8,
        }
        return scene_type_mapping.get(scene_type.lower(), 8)

    def _compute_feature_statistics(self):
        """Compute statistics for feature normalization"""
        logger.info("Computing feature statistics for normalization...")

        # Sample a subset of scenes for statistics
        sample_size = min(100, len(self.valid_scenes))
        sample_indices = np.random.choice(
            len(self.valid_scenes), sample_size, replace=False
        )

        all_features = []

        for idx in sample_indices:
            try:
                scene_data = self.valid_scenes[idx]
                features = self._extract_scene_features(scene_data)
                all_features.append(features.node_features)
            except Exception as e:
                logger.warning(f"Error extracting features for statistics: {e}")
                continue

        if all_features:
            all_features_tensor = torch.cat(all_features, dim=0)
            self.feature_stats = {
                "mean": all_features_tensor.mean(dim=0),
                "std": all_features_tensor.std(dim=0) + 1e-6,  # Avoid division by zero
            }
        else:
            logger.warning("No features extracted for statistics computation")
            self.feature_stats = None

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features using computed statistics"""
        if self.feature_stats is None:
            return features

        mean = self.feature_stats["mean"]
        std = self.feature_stats["std"]

        return (features - mean) / std

    def _load_cached_features(self, scene_id: str) -> SceneFeatures | None:
        """Load cached features from disk"""
        if not self.cache_dir:
            return None

        cache_path = Path(self.cache_dir) / f"{scene_id}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading cached features for {scene_id}: {e}")

        return None

    def _save_cached_features(self, scene_id: str, features: SceneFeatures):
        """Save features to cache"""
        if not self.cache_dir:
            return

        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_path = cache_dir / f"{scene_id}.pkl"
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(features, f)
        except Exception as e:
            logger.warning(f"Error saving cached features for {scene_id}: {e}")

    def _precompute_all_features(self):
        """Precompute features for all scenes"""
        logger.info("Precomputing features for all scenes...")

        for i in range(len(self.valid_scenes)):
            try:
                self.__getitem__(i)
            except Exception as e:
                logger.warning(f"Error precomputing features for scene {i}: {e}")

        logger.info("Feature precomputation completed")


def collate_scene_batch(batch: list[SceneFeatures]) -> dict[str, torch.Tensor]:
    """
    Custom collate function for batching variable-sized scenes.
    """
    if not batch:
        return {}

    # Separate different types of features
    node_features_list = []
    node_positions_list = []
    node_velocities_list = []
    node_types_list = []
    edge_indices_list = []
    edge_features_list = []
    graph_features_list = []
    history_features_list = []
    future_features_list = []

    # Metadata
    scene_types = []
    vessel_counts = []
    scene_ids = []
    batch_indices = []

    node_offset = 0

    for i, scene in enumerate(batch):
        n_nodes = scene.node_features.shape[0]

        # Node features
        node_features_list.append(scene.node_features)
        node_positions_list.append(scene.node_positions)
        node_velocities_list.append(scene.node_velocities)
        node_types_list.append(scene.node_types)

        # Edge features (adjust indices for batching)
        if scene.edge_index.shape[1] > 0:
            edge_index_adjusted = scene.edge_index + node_offset
            edge_indices_list.append(edge_index_adjusted)
            edge_features_list.append(scene.edge_features)

        # Graph features
        graph_features_list.append(scene.graph_features)

        # Temporal features
        history_features_list.append(scene.history_features)
        future_features_list.append(scene.future_features)

        # Metadata
        scene_types.append(scene.scene_type)
        vessel_counts.append(scene.vessel_count)
        scene_ids.append(scene.scene_id)
        batch_indices.extend([i] * n_nodes)

        node_offset += n_nodes

    # Concatenate features
    batched_data = {
        "node_features": torch.cat(node_features_list, dim=0),
        "node_positions": torch.cat(node_positions_list, dim=0),
        "node_velocities": torch.cat(node_velocities_list, dim=0),
        "node_types": torch.cat(node_types_list, dim=0),
        "graph_features": torch.stack(graph_features_list, dim=0),
        "scene_types": torch.tensor(scene_types, dtype=torch.long),
        "vessel_counts": torch.tensor(vessel_counts, dtype=torch.long),
        "batch_indices": torch.tensor(batch_indices, dtype=torch.long),
        "scene_ids": scene_ids,
    }

    # Handle edges
    if edge_indices_list:
        batched_data["edge_index"] = torch.cat(edge_indices_list, dim=1)
        batched_data["edge_features"] = torch.cat(edge_features_list, dim=0)
    else:
        batched_data["edge_index"] = torch.zeros(2, 0, dtype=torch.long)
        batched_data["edge_features"] = torch.zeros(
            0, edge_features_list[0].shape[1] if edge_features_list else 1
        )

    # Handle temporal features (pad to same length)
    if history_features_list:
        max_nodes = max(h.shape[0] for h in history_features_list)
        history_length = history_features_list[0].shape[1]
        n_features = history_features_list[0].shape[2]

        padded_history = torch.zeros(len(batch), max_nodes, history_length, n_features)
        padded_future = torch.zeros(
            len(batch), max_nodes, future_features_list[0].shape[1], n_features
        )

        for i, (hist, fut) in enumerate(
            zip(history_features_list, future_features_list)
        ):
            n_nodes = hist.shape[0]
            padded_history[i, :n_nodes] = hist
            padded_future[i, :n_nodes] = fut

        batched_data["history_features"] = padded_history
        batched_data["future_features"] = padded_future

    return batched_data


def create_scene_dataloader(
    dataset: SceneDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for scene dataset with custom collate function"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_scene_batch,
        **kwargs,
    )
