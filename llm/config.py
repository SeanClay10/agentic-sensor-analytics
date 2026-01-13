"""
Configuration management for LLM module.
Handles loading and validation of settings from YAML files.
"""

import yaml
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class LLMSettings(BaseModel):
    """LLM-specific configuration settings."""
    model_name: str = Field(default="llama3.1:8b")
    base_url: str = Field(default="http://localhost:11434")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    timeout: int = Field(default=30, gt=0)
    max_retries: int = Field(default=3, ge=0)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class LoggingSettings(BaseModel):
    """Logging configuration settings."""
    level: str = Field(default="INFO")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file: str = Field(default="logs/llm.log")
    
    @field_validator('level')
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v_upper


class PerformanceSettings(BaseModel):
    """Performance-related configuration settings."""
    enable_streaming: bool = Field(default=True)
    cache_prompts: bool = Field(default=False)


class LLMConfig(BaseModel):
    """Complete LLM module configuration."""
    llm: LLMSettings = Field(default_factory=LLMSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> 'LLMConfig':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            LLMConfig instance
            
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
                config_data = {}
            
            return cls(**config_data)
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'LLMConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            LLMConfig instance
        """
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return self.model_dump()
    
    def save_yaml(self, config_path: str | Path) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path where to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def get_default_config() -> LLMConfig:
    """
    Get default configuration.
    
    Returns:
        LLMConfig with default settings
    """
    return LLMConfig()


def load_config(config_path: Optional[str | Path] = None) -> LLMConfig:
    """
    Load configuration from file or return defaults.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        LLMConfig instance
    """
    if config_path is None:
        # Try to find config in common locations
        possible_paths = [
            Path('llm_config.yaml'),
            Path('config/llm_config.yaml'),
            Path('../config/llm_config.yaml'),
        ]
        
        for path in possible_paths:
            if path.exists():
                return LLMConfig.from_yaml(path)
        
        # No config file found, use defaults
        return get_default_config()
    
    return LLMConfig.from_yaml(config_path)