"""
Base classes for analytics tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel
import pandas as pd


class AnalyticsResult(BaseModel):
    """Result from analytics tool execution."""
    value: Any
    unit: Optional[str]
    metadata: dict
    success: bool
    error_message: Optional[str] = None
    execution_time_ms: float


class AnalyticsTool(ABC):
    """
    Abstract base class for analytics tools.
    All tools must define name, description, parameters and implement execute().
    """
    
    def __init__(self):
        self.name: str = ""
        self.description: str = ""
        self.parameters: list[str] = []
    
    @abstractmethod
    def execute(self, data: pd.DataFrame, **kwargs) -> AnalyticsResult:
        """
        Execute the analytics operation.
        
        Args:
            data: DataFrame with columns [timestamp, value, unit]
            **kwargs: Additional parameters
            
        Returns:
            AnalyticsResult with value, unit, metadata, success, error_message, execution_time_ms
        """
        pass