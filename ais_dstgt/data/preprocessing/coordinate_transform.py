"""Coordinate Transformation Module for AIS Data.

This module provides coordinate transformation capabilities for AIS data,
including conversion between different coordinate systems and projections.
"""

import logging

import numpy as np

# Optional pyproj imports
try:
    import pyproj  # noqa: F401
    from pyproj import CRS, Transformer
except ImportError:  # pragma: no cover
    CRS = None  # type: ignore
    Transformer = None  # type: ignore

from ..schemas import AISDataFrame

logger = logging.getLogger(__name__)


class CoordinateTransformer:
    """Coordinate transformation utilities for AIS data."""

    def __init__(self, target_crs: str | CRS | None = None):
        """Initialize coordinate transformer.

        Args:
            target_crs: Target coordinate reference system.
                       If None, will use appropriate UTM zone based on data.
        """
        if CRS is None:
            raise ImportError(
                "pyproj is required for CoordinateTransformer but is not installed."
            )

        self.source_crs = CRS.from_epsg(4326)  # WGS84
        self.target_crs = target_crs
        self.transformer = None
        self.inverse_transformer = None

        # Cache for UTM zone detection
        self._utm_zone_cache = {}

    def setup_projection(
        self,
        ais_df: AISDataFrame,
        projection_type: str = "utm",
        custom_crs: str | CRS | None = None,
    ) -> dict:
        """Setup coordinate projection based on data extent.

        Args:
            ais_df: Input AIS DataFrame
            projection_type: Type of projection ('utm', 'mercator', 'custom')
            custom_crs: Custom CRS for projection

        Returns:
            Dict: Projection setup information
        """
        df = ais_df.df

        # Get data bounds
        lon_min, lon_max = df["LON"].min(), df["LON"].max()
        lat_min, lat_max = df["LAT"].min(), df["LAT"].max()

        # Calculate center point
        center_lon = (lon_min + lon_max) / 2
        center_lat = (lat_min + lat_max) / 2

        if projection_type == "utm":
            # Determine appropriate UTM zone
            utm_zone = self._get_utm_zone(center_lon, center_lat)
            self.target_crs = CRS.from_epsg(utm_zone)

        elif projection_type == "mercator":
            # Use Transverse Mercator centered on data
            self.target_crs = CRS.from_proj4(
                f"+proj=tmerc +lat_0={center_lat} +lon_0={center_lon} "
                f"+k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
            )

        elif projection_type == "custom" and custom_crs:
            self.target_crs = CRS.from_user_input(custom_crs)

        else:
            raise ValueError(f"Unsupported projection type: {projection_type}")

        # Create transformers
        self.transformer = Transformer.from_crs(
            self.source_crs, self.target_crs, always_xy=True
        )
        self.inverse_transformer = Transformer.from_crs(
            self.target_crs, self.source_crs, always_xy=True
        )

        projection_info = {
            "projection_type": projection_type,
            "source_crs": self.source_crs.to_string(),
            "target_crs": self.target_crs.to_string(),
            "data_bounds": {
                "lon_min": lon_min,
                "lon_max": lon_max,
                "lat_min": lat_min,
                "lat_max": lat_max,
            },
            "center_point": {"lon": center_lon, "lat": center_lat},
        }

        logger.info(
            f"Setup {projection_type} projection: {self.target_crs.to_string()}"
        )
        return projection_info

    def transform_coordinates(self, ais_df: AISDataFrame) -> AISDataFrame:
        """Transform coordinates from WGS84 to target projection.

        Args:
            ais_df: Input AIS DataFrame with WGS84 coordinates

        Returns:
            AISDataFrame: DataFrame with transformed coordinates
        """
        if self.transformer is None:
            raise ValueError("Projection not set up. Call setup_projection() first.")

        df = ais_df.df.copy()

        logger.info("Transforming coordinates to target projection")

        # Transform coordinates
        x_coords, y_coords = self.transformer.transform(
            df["LON"].values, df["LAT"].values
        )

        # Add transformed coordinates
        df["X"] = x_coords
        df["Y"] = y_coords

        # Store original coordinates
        df["LON_ORIG"] = df["LON"]
        df["LAT_ORIG"] = df["LAT"]

        # Update LAT/LON to transformed coordinates for consistency
        df["LON"] = x_coords
        df["LAT"] = y_coords

        logger.info("Coordinate transformation completed")
        return AISDataFrame(df)

    def inverse_transform_coordinates(self, ais_df: AISDataFrame) -> AISDataFrame:
        """Transform coordinates back to WGS84.

        Args:
            ais_df: Input AIS DataFrame with projected coordinates

        Returns:
            AISDataFrame: DataFrame with WGS84 coordinates
        """
        if self.inverse_transformer is None:
            raise ValueError("Projection not set up. Call setup_projection() first.")

        df = ais_df.df.copy()

        logger.info("Transforming coordinates back to WGS84")

        # Use X,Y coordinates if available, otherwise use LAT/LON
        if "X" in df.columns and "Y" in df.columns:
            x_coords = df["X"].values
            y_coords = df["Y"].values
        else:
            x_coords = df["LON"].values
            y_coords = df["LAT"].values

        # Transform back to WGS84
        lon_coords, lat_coords = self.inverse_transformer.transform(x_coords, y_coords)

        # Update coordinates
        df["LON"] = lon_coords
        df["LAT"] = lat_coords

        logger.info("Inverse coordinate transformation completed")
        return AISDataFrame(df)

    def calculate_distances(self, ais_df: AISDataFrame) -> AISDataFrame:
        """Calculate distances and speeds using projected coordinates.

        Args:
            ais_df: Input AIS DataFrame with projected coordinates

        Returns:
            AISDataFrame: DataFrame with calculated distances and speeds
        """
        df = ais_df.df.copy()

        # Ensure data is sorted by MMSI and time
        df = df.sort_values(["MMSI", "BaseDateTime"])

        # Use projected coordinates if available
        if "X" in df.columns and "Y" in df.columns:
            x_col, y_col = "X", "Y"
        else:
            x_col, y_col = "LON", "LAT"

        # Calculate distances and speeds for each vessel
        df["Distance"] = 0.0
        df["CalculatedSpeed"] = 0.0
        df["TimeDelta"] = 0.0

        for mmsi in df["MMSI"].unique():
            mask = df["MMSI"] == mmsi
            vessel_data = df[mask].copy()

            if len(vessel_data) < 2:
                continue

            # Calculate distances between consecutive points
            x_diff = vessel_data[x_col].diff()
            y_diff = vessel_data[y_col].diff()
            distances = np.sqrt(x_diff**2 + y_diff**2)

            # Calculate time differences (in seconds)
            time_diff = vessel_data["BaseDateTime"].diff().dt.total_seconds()

            # Calculate speeds (m/s)
            speeds = distances / time_diff
            speeds = speeds.fillna(0.0)

            # Convert to knots if needed
            speeds_knots = speeds * 1.94384  # m/s to knots

            # Update DataFrame
            df.loc[mask, "Distance"] = distances.fillna(0.0)
            df.loc[mask, "CalculatedSpeed"] = speeds_knots.fillna(0.0)
            df.loc[mask, "TimeDelta"] = time_diff.fillna(0.0)

        logger.info("Distance and speed calculations completed")
        return AISDataFrame(df)

    def _get_utm_zone(self, lon: float, lat: float) -> int:
        """Get appropriate UTM zone EPSG code for given coordinates.

        Args:
            lon: Longitude in decimal degrees
            lat: Latitude in decimal degrees

        Returns:
            int: UTM zone EPSG code
        """
        # Cache key
        cache_key = (round(lon, 2), round(lat, 2))

        if cache_key in self._utm_zone_cache:
            return self._utm_zone_cache[cache_key]

        # Calculate UTM zone
        zone_number = int((lon + 180) / 6) + 1

        # Handle special cases
        if 56 <= lat < 64 and 3 <= lon < 12:
            zone_number = 32
        elif 72 <= lat < 84:
            if 0 <= lon < 9:
                zone_number = 31
            elif 9 <= lon < 21:
                zone_number = 33
            elif 21 <= lon < 33:
                zone_number = 35
            elif 33 <= lon < 42:
                zone_number = 37

        # Determine hemisphere
        if lat >= 0:
            # Northern hemisphere
            epsg_code = 32600 + zone_number
        else:
            # Southern hemisphere
            epsg_code = 32700 + zone_number

        self._utm_zone_cache[cache_key] = epsg_code
        return epsg_code

    def get_projection_info(self) -> dict:
        """Get information about current projection setup.

        Returns:
            Dict: Projection information
        """
        if self.target_crs is None:
            return {"status": "No projection set up"}

        return {
            "source_crs": self.source_crs.to_string(),
            "target_crs": self.target_crs.to_string(),
            "target_crs_name": self.target_crs.name,
            "target_crs_type": self.target_crs.coordinate_system.name,
            "units": self.target_crs.axis_info[0].unit_name,
        }

    def create_local_mercator(
        self,
        center_lon: float,
        center_lat: float,
        false_easting: float = 0.0,
        false_northing: float = 0.0,
    ) -> CRS:
        """Create a local Transverse Mercator projection.

        Args:
            center_lon: Central longitude
            center_lat: Central latitude
            false_easting: False easting value
            false_northing: False northing value

        Returns:
            CRS: Local Transverse Mercator CRS
        """
        proj_string = (
            f"+proj=tmerc +lat_0={center_lat} +lon_0={center_lon} "
            f"+k=1 +x_0={false_easting} +y_0={false_northing} "
            f"+datum=WGS84 +units=m +no_defs"
        )

        return CRS.from_proj4(proj_string)
