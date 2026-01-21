"""
Unit tests for analytics tools.
Tests each tool in isolation with various edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analytics import (
    TemporalMeanTool,
    TemporalAggregationTool,
    SpatialComparisonTool,
    StatisticalSummaryTool
)


class TestTemporalMeanTool:
    """Unit tests for TemporalMeanTool."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data fixture."""
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        timestamps = [start + timedelta(hours=i) for i in range(10)]
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'value': [20.0, 22.0, 21.0, 23.0, 25.0, 24.0, 26.0, 27.0, 28.0, 30.0],
            'unit': '°C'
        })
    
    def test_mean_calculation(self, sample_data):
        """Test basic mean calculation."""
        tool = TemporalMeanTool()
        result = tool.execute(sample_data, operation='mean')
        
        assert result.success is True
        assert result.value == pytest.approx(24.6, rel=1e-2)
        assert result.unit == '°C'
        assert result.metadata['sample_size'] == 10
    
    def test_min_calculation(self, sample_data):
        """Test min operation."""
        tool = TemporalMeanTool()
        result = tool.execute(sample_data, operation='min')
        
        assert result.success is True
        assert result.value == 20.0
        assert result.metadata['operation'] == 'min'
    
    def test_max_calculation(self, sample_data):
        """Test max operation."""
        tool = TemporalMeanTool()
        result = tool.execute(sample_data, operation='max')
        
        assert result.success is True
        assert result.value == 30.0
        assert result.metadata['operation'] == 'max'
    
    def test_default_operation(self, sample_data):
        """Test default operation (should be mean)."""
        tool = TemporalMeanTool()
        result = tool.execute(sample_data)
        
        assert result.success is True
        assert result.metadata['operation'] == 'mean'
    
    def test_invalid_operation(self, sample_data):
        """Test invalid operation handling."""
        tool = TemporalMeanTool()
        result = tool.execute(sample_data, operation='invalid')
        
        assert result.success is False
        assert 'Invalid operation' in result.error_message
    
    def test_missing_columns(self):
        """Test error handling for missing columns."""
        tool = TemporalMeanTool()
        data = pd.DataFrame({'value': [1, 2, 3]})
        result = tool.execute(data)
        
        assert result.success is False
        assert 'Missing required columns' in result.error_message
    
    def test_metadata_includes_statistics(self, sample_data):
        """Test that metadata includes std, min, max."""
        tool = TemporalMeanTool()
        result = tool.execute(sample_data, operation='mean')
        
        assert 'std_dev' in result.metadata
        assert 'min' in result.metadata
        assert 'max' in result.metadata
        assert result.metadata['min'] == 20.0
        assert result.metadata['max'] == 30.0
    
    def test_single_value(self):
        """Test with single data point."""
        tool = TemporalMeanTool()
        data = pd.DataFrame({
            'timestamp': [datetime(2025, 1, 1, tzinfo=timezone.utc)],
            'value': [25.0],
            'unit': '°C'
        })
        result = tool.execute(data, operation='mean')
        
        assert result.success is True
        assert result.value == 25.0
        assert result.metadata['sample_size'] == 1


class TestTemporalAggregationTool:
    """Unit tests for TemporalAggregationTool."""
    
    @pytest.fixture
    def hourly_data(self):
        """Create hourly data for 3 days."""
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        timestamps = [start + timedelta(hours=i) for i in range(72)]
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'value': [20.0 + (i % 24) for i in range(72)],
            'unit': '°C'
        })
    
    def test_daily_aggregation_mean(self, hourly_data):
        """Test daily aggregation with mean."""
        tool = TemporalAggregationTool()
        result = tool.execute(hourly_data, aggregation_level='daily', operation='mean')
        
        assert result.success is True
        assert result.metadata['num_periods'] == 3
        assert result.metadata['aggregation_level'] == 'daily'
        assert result.metadata['operation'] == 'mean'
    
    def test_hourly_aggregation(self, hourly_data):
        """Test hourly aggregation."""
        tool = TemporalAggregationTool()
        result = tool.execute(hourly_data, aggregation_level='hourly', operation='mean')
        
        assert result.success is True
        assert result.metadata['num_periods'] == 72
    
    def test_weekly_aggregation(self, hourly_data):
        """Test weekly aggregation."""
        tool = TemporalAggregationTool()
        result = tool.execute(hourly_data, aggregation_level='weekly', operation='mean')
        
        assert result.success is True
        assert isinstance(result.value, list)
    
    def test_aggregation_max_operation(self, hourly_data):
        """Test aggregation with max operation."""
        tool = TemporalAggregationTool()
        result = tool.execute(hourly_data, aggregation_level='daily', operation='max')
        
        assert result.success is True
        assert result.metadata['operation'] == 'max'
    
    def test_invalid_aggregation_level(self, hourly_data):
        """Test invalid aggregation level."""
        tool = TemporalAggregationTool()
        result = tool.execute(hourly_data, aggregation_level='monthly', operation='mean')
        
        assert result.success is False
        assert 'Invalid aggregation_level' in result.error_message
    
    def test_missing_columns(self):
        """Test missing required columns."""
        tool = TemporalAggregationTool()
        data = pd.DataFrame({'value': [1, 2, 3]})
        result = tool.execute(data, aggregation_level='daily')
        
        assert result.success is False
        assert 'Missing required columns' in result.error_message
    
    def test_result_structure(self, hourly_data):
        """Test that result has correct structure."""
        tool = TemporalAggregationTool()
        result = tool.execute(hourly_data, aggregation_level='daily', operation='mean')
        
        assert isinstance(result.value, list)
        assert all('timestamp' in item for item in result.value)
        assert all('value' in item for item in result.value)


class TestSpatialComparisonTool:
    """Unit tests for SpatialComparisonTool."""
    
    @pytest.fixture
    def multi_location_data(self):
        """Create data for multiple locations."""
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        timestamps = [start + timedelta(hours=i) for i in range(24)]
        
        data1 = pd.DataFrame({
            'timestamp': timestamps,
            'value': [20.0] * 24,
            'unit': '°C',
            'location': 'Room201'
        })
        
        data2 = pd.DataFrame({
            'timestamp': timestamps,
            'value': [25.0] * 24,
            'unit': '°C',
            'location': 'Room202'
        })
        
        data3 = pd.DataFrame({
            'timestamp': timestamps,
            'value': [22.0] * 24,
            'unit': '°C',
            'location': 'Room203'
        })
        
        return pd.concat([data1, data2, data3], ignore_index=True)
    
    def test_mean_comparison(self, multi_location_data):
        """Test mean comparison across locations."""
        tool = SpatialComparisonTool()
        result = tool.execute(multi_location_data, operation='mean')
        
        assert result.success is True
        assert len(result.value) == 3
        assert result.metadata['num_locations'] == 3
    
    def test_ranking_order(self, multi_location_data):
        """Test that locations are ranked correctly."""
        tool = SpatialComparisonTool()
        result = tool.execute(multi_location_data, operation='mean')
        
        # Should be sorted by value descending
        assert result.value[0]['location'] == 'Room202'  # 25.0
        assert result.value[0]['rank'] == 1
        assert result.value[1]['location'] == 'Room203'  # 22.0
        assert result.value[1]['rank'] == 2
        assert result.value[2]['location'] == 'Room201'  # 20.0
        assert result.value[2]['rank'] == 3
    
    def test_percent_of_highest(self, multi_location_data):
        """Test percent_of_highest calculation."""
        tool = SpatialComparisonTool()
        result = tool.execute(multi_location_data, operation='mean')
        
        assert result.value[0]['percent_of_highest'] == 100.0
        assert result.value[2]['percent_of_highest'] == pytest.approx(80.0, rel=1e-2)
    
    def test_min_comparison(self, multi_location_data):
        """Test min comparison."""
        tool = SpatialComparisonTool()
        result = tool.execute(multi_location_data, operation='min')
        
        assert result.success is True
        assert result.metadata['operation'] == 'min'
    
    def test_invalid_operation(self, multi_location_data):
        """Test invalid operation."""
        tool = SpatialComparisonTool()
        result = tool.execute(multi_location_data, operation='median')
        
        assert result.success is False
        assert 'Invalid operation' in result.error_message
    
    def test_missing_location_column(self):
        """Test missing location column."""
        tool = SpatialComparisonTool()
        data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'value': [20.0],
            'unit': '°C'
        })
        result = tool.execute(data)
        
        assert result.success is False
        assert 'Missing required columns' in result.error_message
    
    def test_single_location(self):
        """Test with single location."""
        tool = SpatialComparisonTool()
        data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'value': [20.0],
            'unit': '°C',
            'location': 'Room201'
        })
        result = tool.execute(data, operation='mean')
        
        assert result.success is True
        assert len(result.value) == 1
        assert result.value[0]['rank'] == 1


class TestStatisticalSummaryTool:
    """Unit tests for StatisticalSummaryTool."""
    
    @pytest.fixture
    def normal_data(self):
        """Create normally distributed data."""
        np.random.seed(42)
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        timestamps = [start + timedelta(hours=i) for i in range(100)]
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'value': np.random.normal(25.0, 5.0, 100),
            'unit': '°C'
        })
    
    def test_summary_statistics(self, normal_data):
        """Test that all summary statistics are present."""
        tool = StatisticalSummaryTool()
        result = tool.execute(normal_data)
        
        assert result.success is True
        
        required_stats = ['count', 'mean', 'std', 'min', 'q1', 'median', 'q3', 'max', 'skewness', 'kurtosis']
        for stat in required_stats:
            assert stat in result.value, f"Missing statistic: {stat}"
    
    def test_count_matches_data(self, normal_data):
        """Test that count matches data length."""
        tool = StatisticalSummaryTool()
        result = tool.execute(normal_data)
        
        assert result.value['count'] == len(normal_data)
    
    def test_quartile_ordering(self, normal_data):
        """Test that quartiles are ordered correctly."""
        tool = StatisticalSummaryTool()
        result = tool.execute(normal_data)
        
        assert result.value['min'] <= result.value['q1']
        assert result.value['q1'] <= result.value['median']
        assert result.value['median'] <= result.value['q3']
        assert result.value['q3'] <= result.value['max']
    
    def test_missing_columns(self):
        """Test missing required columns."""
        tool = StatisticalSummaryTool()
        data = pd.DataFrame({'value': [1, 2, 3]})
        result = tool.execute(data)
        
        assert result.success is False
        assert 'Missing required columns' in result.error_message
    
    def test_small_dataset(self):
        """Test with small dataset."""
        tool = StatisticalSummaryTool()
        data = pd.DataFrame({
            'timestamp': [datetime.now()] * 5,
            'value': [20.0, 21.0, 22.0, 23.0, 24.0],
            'unit': '°C'
        })
        result = tool.execute(data)
        
        assert result.success is True
        assert result.value['count'] == 5
        assert result.value['median'] == 22.0


class TestExecutionTime:
    """Test that all tools track execution time."""
    
    def test_temporal_mean_tracks_time(self):
        """Test TemporalMeanTool tracks execution time."""
        tool = TemporalMeanTool()
        data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'value': [20.0],
            'unit': '°C'
        })
        result = tool.execute(data, operation='mean')
        
        assert result.execution_time_ms > 0
    
    def test_aggregation_tracks_time(self):
        """Test TemporalAggregationTool tracks execution time."""
        tool = TemporalAggregationTool()
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        data = pd.DataFrame({
            'timestamp': [start + timedelta(hours=i) for i in range(24)],
            'value': [20.0] * 24,
            'unit': '°C'
        })
        result = tool.execute(data, aggregation_level='daily', operation='mean')
        
        assert result.execution_time_ms > 0
    
    def test_spatial_comparison_tracks_time(self):
        """Test SpatialComparisonTool tracks execution time."""
        tool = SpatialComparisonTool()
        data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'value': [20.0],
            'unit': '°C',
            'location': 'Room201'
        })
        result = tool.execute(data, operation='mean')
        
        assert result.execution_time_ms > 0
    
    def test_summary_tracks_time(self):
        """Test StatisticalSummaryTool tracks execution time."""
        tool = StatisticalSummaryTool()
        data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'value': [20.0],
            'unit': '°C'
        })
        result = tool.execute(data)
        
        assert result.execution_time_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])