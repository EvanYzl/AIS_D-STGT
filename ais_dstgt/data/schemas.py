"""AIS Data Schemas and Type Definitions.

This module defines the data schemas, types, and validation rules for AIS data
processing. It ensures data consistency and type safety throughout the pipeline.
"""

from datetime import datetime
from enum import Enum

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class VesselType(Enum):
    """Standard vessel type codes according to AIS specifications."""

    FISHING = 30
    TOWING = 31
    TOWING_LARGE = 32
    DREDGING = 33
    DIVING = 34
    MILITARY = 35
    SAILING = 36
    PLEASURE_CRAFT = 37
    CARGO = 70
    TANKER = 80
    PASSENGER = 60
    LAW_ENFORCEMENT = 55
    PILOT_VESSEL = 50
    SEARCH_RESCUE = 51
    TUG = 52
    PORT_TENDER = 53
    ANTI_POLLUTION = 54
    MEDICAL = 58
    NOT_PARTY_TO_CONFLICT = 59


class NavigationStatus(Enum):
    """Navigation status codes according to AIS specifications."""

    UNDER_WAY_USING_ENGINE = 0
    AT_ANCHOR = 1
    NOT_UNDER_COMMAND = 2
    RESTRICTED_MANOEUVRABILITY = 3
    CONSTRAINED_BY_DRAUGHT = 4
    MOORED = 5
    AGROUND = 6
    ENGAGED_IN_FISHING = 7
    UNDER_WAY_SAILING = 8
    AIS_SART = 14
    UNDEFINED = 15


class TransceiverClass(Enum):
    """AIS transceiver class."""

    CLASS_A = "A"
    CLASS_B = "B"


class AISRecord(BaseModel):
    """Individual AIS record with validation."""

    mmsi: int = Field(
        ..., ge=100000000, le=999999999, description="Maritime Mobile Service Identity"
    )
    base_datetime: datetime = Field(..., description="Timestamp of the AIS message")
    lat: float = Field(
        ..., ge=-90.0, le=90.0, description="Latitude in decimal degrees"
    )
    lon: float = Field(
        ..., ge=-180.0, le=180.0, description="Longitude in decimal degrees"
    )
    sog: float = Field(..., ge=0.0, le=102.3, description="Speed over ground in knots")
    cog: float = Field(
        ..., ge=0.0, lt=360.0, description="Course over ground in degrees"
    )
    heading: float | None = Field(
        None, ge=0.0, lt=360.0, description="True heading in degrees"
    )
    vessel_name: str | None = Field(None, max_length=20, description="Vessel name")
    imo: str | None = Field(
        None, description="International Maritime Organization number"
    )
    call_sign: str | None = Field(None, max_length=7, description="Radio call sign")
    vessel_type: int | None = Field(None, ge=0, le=99, description="Type of vessel")
    status: int | None = Field(None, ge=0, le=15, description="Navigation status")
    length: float | None = Field(None, ge=0.0, le=400.0, description="Length in meters")
    width: float | None = Field(None, ge=0.0, le=63.0, description="Width in meters")
    draft: float | None = Field(
        None, ge=0.0, le=25.5, description="Maximum draft in meters"
    )
    cargo: int | None = Field(None, description="Cargo type")
    transceiver_class: str | None = Field(None, description="AIS transceiver class")

    @field_validator("heading")
    @classmethod
    def validate_heading(cls, v):
        """Validate heading value, allowing 511 as 'not available'."""
        if v is not None and v == 511.0:
            return None
        return v

    @field_validator("mmsi")
    @classmethod
    def validate_mmsi(cls, v):
        """Validate MMSI format and range."""
        if not (100000000 <= v <= 999999999):
            raise ValueError(f"Invalid MMSI: {v}")
        return v

    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)


class AISDataFrame:
    """Wrapper for AIS DataFrame with validation and utility methods."""

    REQUIRED_COLUMNS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG"]

    OPTIONAL_COLUMNS = [
        "Heading",
        "VesselName",
        "IMO",
        "CallSign",
        "VesselType",
        "Status",
        "Length",
        "Width",
        "Draft",
        "Cargo",
        "TransceiverClass",
    ]

    DTYPE_MAPPING = {
        "MMSI": "int64",
        "BaseDateTime": "datetime64[ns]",
        "LAT": "float64",
        "LON": "float64",
        "SOG": "float64",
        "COG": "float64",
        "Heading": "float64",
        "VesselName": "string",
        "IMO": "string",
        "CallSign": "string",
        "VesselType": "Int64",
        "Status": "Int64",
        "Length": "float64",
        "Width": "float64",
        "Draft": "float64",
        "Cargo": "Int64",
        "TransceiverClass": "string",
    }

    def __init__(self, df: pd.DataFrame):
        """Initialize with validation."""
        self.df = self._validate_and_normalize(df)

    def _validate_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize the DataFrame."""
        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Normalize column names if needed
        df = df.copy()

        # Apply data types
        for col, dtype in self.DTYPE_MAPPING.items():
            if col in df.columns:
                try:
                    if dtype == "datetime64[ns]" and df[col].dtype == "object":
                        df[col] = pd.to_datetime(df[col])
                    else:
                        df[col] = df[col].astype(dtype)
                except Exception as e:
                    raise ValueError(f"Failed to convert column {col} to {dtype}: {e}")

        return df

    def validate_ranges(self) -> pd.DataFrame:
        """Validate data ranges and return invalid records."""
        invalid_records = []

        # Latitude range
        invalid_lat = self.df[(self.df["LAT"] < -90) | (self.df["LAT"] > 90)]
        if not invalid_lat.empty:
            invalid_records.append(("LAT", invalid_lat))

        # Longitude range
        invalid_lon = self.df[(self.df["LON"] < -180) | (self.df["LON"] > 180)]
        if not invalid_lon.empty:
            invalid_records.append(("LON", invalid_lon))

        # Speed over ground
        invalid_sog = self.df[(self.df["SOG"] < 0) | (self.df["SOG"] > 102.3)]
        if not invalid_sog.empty:
            invalid_records.append(("SOG", invalid_sog))

        # Course over ground
        invalid_cog = self.df[(self.df["COG"] < 0) | (self.df["COG"] >= 360)]
        if not invalid_cog.empty:
            invalid_records.append(("COG", invalid_cog))

        return invalid_records

    def get_summary_stats(self) -> dict:
        """Get summary statistics for the dataset."""
        return {
            "total_records": len(self.df),
            "unique_vessels": self.df["MMSI"].nunique(),
            "time_range": {
                "start": self.df["BaseDateTime"].min(),
                "end": self.df["BaseDateTime"].max(),
            },
            "geographic_bounds": {
                "lat_min": self.df["LAT"].min(),
                "lat_max": self.df["LAT"].max(),
                "lon_min": self.df["LON"].min(),
                "lon_max": self.df["LON"].max(),
            },
            "missing_values": self.df.isnull().sum().to_dict(),
        }

    def to_parquet(self, path: str, **kwargs) -> None:
        """Save to Parquet format."""
        self.df.to_parquet(path, **kwargs)

    def to_csv(self, path: str, **kwargs) -> None:
        """Save to CSV format."""
        self.df.to_csv(path, index=False, **kwargs)
