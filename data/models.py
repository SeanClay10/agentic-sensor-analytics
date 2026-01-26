"""
Pydantic models for sensor data validation and type safety.
"""

from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, List
from enum import Enum


class SensorType(str, Enum):
    """Supported sensor types."""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    CO2 = "co2"
    MOISTURE = "moisture"
    LOAD = "load"
    STRAIN = "strain"


class SensorReading(BaseModel):
    """Single sensor reading with timestamp and value."""
    timestamp: datetime = Field(description="Timestamp of the reading")
    value: float = Field(description="Sensor reading value")
    unit: str = Field(description="Unit of measurement (e.g., °C, %, ppm)")
    raw_value: Optional[float] = Field(default=None, description="Raw sensor value before conversion")
    quality_flag: int = Field(default=0, description="Data quality flag (0=valid, 1=suspect, 2=missing)")
    
    @field_validator('quality_flag')
    @classmethod
    def validate_quality_flag(cls, v: int) -> int:
        """Ensure quality flag is 0, 1, or 2."""
        if v not in [0, 1, 2]:
            raise ValueError("quality_flag must be 0 (valid), 1 (suspect), or 2 (missing)")
        return v


class SensorMetadata(BaseModel):
    """Metadata about a sensor."""
    sensor_id: int = Field(description="Internal database identifier for sensor")
    name: str = Field(description="Descriptive name of the sensor")
    sensor_type: str = Field(description="Type of sensor (temperature, humidity, etc.)")
    location: str = Field(description="Physical location of the sensor")
    unit: str = Field(description="Unit of measurement")
    node_id: int = Field(description="Parent node/device ID")
    input_channel: Optional[int] = Field(default=None, description="Input channel on the node")
    created: Optional[datetime] = Field(default=None, description="Creation timestamp")
    modified: Optional[datetime] = Field(default=None, description="Last modification timestamp")


class NodeMetadata(BaseModel):
    """Metadata about a node (WIDAQ device)."""
    node_id: int = Field(description="Internal database identifier for node")
    physical_id: int = Field(description="Physical device ID printed on WIDAQ")
    name: str = Field(description="Descriptive name for the node")
    location: str = Field(description="Physical location of the node")
    created: Optional[datetime] = Field(default=None)
    modified: Optional[datetime] = Field(default=None)


class TimeRange(BaseModel):
    """Time range for data queries."""
    start_time: datetime = Field(description="Start of time range")
    end_time: datetime = Field(description="End of time range")
    
    @field_validator('end_time')
    @classmethod
    def end_after_start(cls, v: datetime, info) -> datetime:
        """Validate that end_time is after start_time."""
        if 'start_time' in info.data and v <= info.data['start_time']:
            raise ValueError("end_time must be after start_time")
        return v


class DataQuery(BaseModel):
    """Query parameters for sensor data retrieval."""
    sensor_type: str = Field(description="Type of sensor to query")
    location: str | List[str] = Field(description="Location(s) to query")
    start_time: datetime = Field(description="Start of time range")
    end_time: datetime = Field(description="End of time range")
    operation: Optional[str] = Field(default=None, description="Optional operation to perform")
    
    def get_locations_list(self) -> List[str]:
        """Get locations as a list regardless of input type."""
        return [self.location] if isinstance(self.location, str) else self.location


class DataQueryResult(BaseModel):
    """Result of a data query."""
    sensor_metadata: SensorMetadata
    readings: List[SensorReading]
    query_params: DataQuery
    total_readings: int = Field(description="Total number of readings returned")
    has_quality_issues: bool = Field(default=False, description="Whether data has quality flags")
    
    @field_validator('total_readings')
    @classmethod
    def validate_count(cls, v: int, info) -> int:
        """Ensure total_readings matches actual readings count."""
        if 'readings' in info.data:
            actual_count = len(info.data['readings'])
            if v != actual_count:
                raise ValueError(f"total_readings ({v}) doesn't match actual count ({actual_count})")
        return v


class SystemState(BaseModel):
    """Current state of the data system."""
    available_sensors: List[str] = Field(description="List of available sensor types")
    available_locations: List[str] = Field(description="List of available locations")
    time_range: TimeRange = Field(description="Available data time range")
    total_sensors: int = Field(description="Total number of sensors in system")
    total_nodes: int = Field(description="Total number of nodes in system")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update time")