"""
Configuration management for data module.
"""

import yaml
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class APISettings(BaseModel):
    """API connection settings."""
    username: str = Field(description="API username")
    password: str = Field(description="API password")
    base_url: str = Field(description="Base URL for API")
    job_id: int = Field(description="Job ID for Peavy Hall project")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")


class CacheSettings(BaseModel):
    """Data caching settings."""
    enabled: bool = Field(default=True, description="Enable caching")
    ttl_seconds: int = Field(default=3600, description="Cache time-to-live in seconds")
    max_size_mb: int = Field(default=100, description="Maximum cache size in MB")


class ValidationSettings(BaseModel):
    """Data validation settings."""
    check_ranges: bool = Field(default=True, description="Validate sensor value ranges")
    flag_outliers: bool = Field(default=True, description="Flag statistical outliers")
    outlier_std_devs: float = Field(default=3.0, description="Standard deviations for outlier detection")


class SensorMapping(BaseModel):
    """Mapping between sensor names and types."""
    temperature_keywords: list[str] = Field(
        default=['temp', 'temperature', 'tmp'],
        description="Keywords to identify temperature sensors"
    )
    humidity_keywords: list[str] = Field(
        default=['humidity', 'rh', 'relative humidity'],
        description="Keywords to identify humidity sensors"
    )
    co2_keywords: list[str] = Field(
        default=['co2', 'carbon dioxide', 'carbondioxide'],
        description="Keywords to identify CO2 sensors"
    )
    moisture_keywords: list[str] = Field(
        default=['moisture', 'mc', 'moisture content'],
        description="Keywords to identify moisture sensors"
    )


class DataConfig(BaseModel):
    """Complete data module configuration."""
    api: APISettings
    cache: CacheSettings = Field(default_factory=CacheSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    sensor_mapping: SensorMapping = Field(default_factory=SensorMapping)
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> 'DataConfig':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            DataConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                raise ValueError("Empty configuration file")
            
            return cls(**config_data)
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def save_yaml(self, config_path: str | Path) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path where to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict, excluding password for security
        config_dict = self.model_dump()
        config_dict['api']['password'] = '***REDACTED***'
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def load_config(config_path: Optional[str | Path] = None) -> DataConfig:
    """
    Load configuration from file or return defaults.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        DataConfig instance
    """
    if config_path is None:
        # Try to find config in common locations
        possible_paths = [
            Path('data_config.yaml'),
            Path('config/data_config.yaml'),
            Path('../config/data_config.yaml'),
        ]
        
        for path in possible_paths:
            if path.exists():
                return DataConfig.from_yaml(path)
        
        # No config found - user must provide credentials
        raise FileNotFoundError(
            "No configuration file found. Please create data_config.yaml with API credentials."
        )
    
    return DataConfig.from_yaml(config_path)