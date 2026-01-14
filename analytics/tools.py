"""
Core analytics tool implementations.
"""

import time
import pandas as pd
from scipy import stats

from .base import AnalyticsTool, AnalyticsResult


class TemporalMeanTool(AnalyticsTool):
    """Calculate mean value over time range."""
    
    def __init__(self):
        super().__init__()
        self.name = "temporal_mean"
        self.description = "Calculate mean (average) value"
        self.parameters = []
    
    def execute(self, data: pd.DataFrame, **kwargs) -> AnalyticsResult:
        start_time = time.time()
        
        # Validate DataFrame has required columns: timestamp, value, unit
        required_cols = {'timestamp', 'value', 'unit'}
        if not required_cols.issubset(data.columns):
            return AnalyticsResult(
                value=None,
                unit=None,
                metadata={},
                success=False,
                error_message=f"Missing required columns: {required_cols - set(data.columns)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Calculate mean using pandas .mean()
        mean_value = data['value'].mean()
        
        # Compute metadata: std dev, min, max, sample size
        metadata = {
            "std_dev": float(data['value'].std()),
            "min": float(data['value'].min()),
            "max": float(data['value'].max()),
            "sample_size": len(data)
        }
        
        # Return AnalyticsResult with statistics
        return AnalyticsResult(
            value=float(mean_value),
            unit=data['unit'].iloc[0],
            metadata=metadata,
            success=True,
            execution_time_ms=(time.time() - start_time) * 1000
        )


class TemporalAggregationTool(AnalyticsTool):
    """Aggregate data by time periods."""
    
    def __init__(self):
        super().__init__()
        self.name = "temporal_aggregation"
        self.description = "Aggregate data by hourly/daily/weekly periods"
        self.parameters = ["aggregation_level", "operation"]
    
    def execute(self, data: pd.DataFrame, **kwargs) -> AnalyticsResult:
        start_time = time.time()
        
        # Validate required columns
        required_cols = {'timestamp', 'value', 'unit'}
        if not required_cols.issubset(data.columns):
            return AnalyticsResult(
                value=None,
                unit=None,
                metadata={},
                success=False,
                error_message=f"Missing required columns: {required_cols - set(data.columns)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Accept aggregation_level: "hourly", "daily", or "weekly"
        aggregation_level = kwargs.get('aggregation_level')
        operation = kwargs.get('operation', 'mean')
        
        freq_map = {
            'hourly': 'H',
            'daily': 'D',
            'weekly': 'W'
        }
        
        if aggregation_level not in freq_map:
            return AnalyticsResult(
                value=None,
                unit=None,
                metadata={},
                success=False,
                error_message=f"Invalid aggregation_level: {aggregation_level}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Use pandas resample() method with appropriate frequency
        data_indexed = data.set_index('timestamp').sort_index()
        
        # Apply aggregation function to each time group
        aggregated = data_indexed['value'].resample(freq_map[aggregation_level]).agg(operation)
        
        # Return DataFrame with aggregated values
        result_data = [
            {"timestamp": ts.isoformat(), "value": float(val)}
            for ts, val in aggregated.items() if pd.notna(val)
        ]
        
        # Calculate overall aggregate across entire period
        if operation == 'mean':
            overall_aggregate = data['value'].mean()
        elif operation == 'min':
            overall_aggregate = data['value'].min()
        elif operation == 'max':
            overall_aggregate = data['value'].max()
        elif operation == 'sum':
            overall_aggregate = data['value'].sum()
        else:
            overall_aggregate = data['value'].agg(operation)
        
        return AnalyticsResult(
            value=result_data,
            unit=data['unit'].iloc[0],
            metadata={
                "aggregation_level": aggregation_level,
                "operation": operation,
                "num_periods": len(result_data),
                "overall_aggregate": float(overall_aggregate)
            },
            success=True,
            execution_time_ms=(time.time() - start_time) * 1000
        )


class SpatialComparisonTool(AnalyticsTool):
    """Compare statistics across multiple locations."""
    
    def __init__(self):
        super().__init__()
        self.name = "spatial_comparison"
        self.description = "Compare values across multiple locations"
        self.parameters = ["operation"]
    
    def execute(self, data: pd.DataFrame, **kwargs) -> AnalyticsResult:
        start_time = time.time()
        
        # Validate required columns
        required_cols = {'timestamp', 'value', 'unit', 'location'}
        if not required_cols.issubset(data.columns):
            return AnalyticsResult(
                value=None,
                unit=None,
                metadata={},
                success=False,
                error_message=f"Missing required columns: {required_cols - set(data.columns)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        operation = kwargs.get('operation', 'mean')
        
        # Compute statistics for each location
        comparison_results = []
        for location in data['location'].unique():
            location_data = data[data['location'] == location]
            
            if operation == 'mean':
                value = location_data['value'].mean()
            elif operation == 'min':
                value = location_data['value'].min()
            elif operation == 'max':
                value = location_data['value'].max()
            else:
                return AnalyticsResult(
                    value=None,
                    unit=None,
                    metadata={},
                    success=False,
                    error_message=f"Invalid operation: {operation}",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            comparison_results.append({
                "location": location,
                "value": float(value)
            })
        
        # Calculate relative differences and rankings
        comparison_results.sort(key=lambda x: x['value'], reverse=True)
        for i, result in enumerate(comparison_results, 1):
            result['rank'] = i
        
        highest_value = comparison_results[0]['value']
        for result in comparison_results:
            if highest_value != 0:
                result['percent_of_highest'] = (result['value'] / highest_value) * 100
        
        # Return structured comparison results
        return AnalyticsResult(
            value=comparison_results,
            unit=data['unit'].iloc[0],
            metadata={
                "operation": operation,
                "num_locations": len(comparison_results)
            },
            success=True,
            execution_time_ms=(time.time() - start_time) * 1000
        )


class StatisticalSummaryTool(AnalyticsTool):
    """Generate comprehensive statistical summary."""
    
    def __init__(self):
        super().__init__()
        self.name = "statistical_summary"
        self.description = "Generate comprehensive statistical summary"
        self.parameters = []
    
    def execute(self, data: pd.DataFrame, **kwargs) -> AnalyticsResult:
        start_time = time.time()
        
        # Validate required columns
        required_cols = {'timestamp', 'value', 'unit'}
        if not required_cols.issubset(data.columns):
            return AnalyticsResult(
                value=None,
                unit=None,
                metadata={},
                success=False,
                error_message=f"Missing required columns: {required_cols - set(data.columns)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Use pandas .describe() for basic statistics
        desc = data['value'].describe()
        
        # Calculate additional metrics using scipy.stats (skewness, quartiles)
        summary = {
            "count": int(desc['count']),
            "mean": float(desc['mean']),
            "std": float(desc['std']),
            "min": float(desc['min']),
            "q1": float(desc['25%']),
            "median": float(desc['50%']),
            "q3": float(desc['75%']),
            "max": float(desc['max']),
            "skewness": float(stats.skew(data['value'].dropna())),
            "kurtosis": float(stats.kurtosis(data['value'].dropna()))
        }
        
        # Package all statistics into result metadata
        return AnalyticsResult(
            value=summary,
            unit=data['unit'].iloc[0],
            metadata={"operation": "summary"},
            success=True,
            execution_time_ms=(time.time() - start_time) * 1000
        )