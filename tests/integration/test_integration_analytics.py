"""
Integration tests for analytics module.
Tests the complete analytics system including registry and tool interactions.
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analytics import get_registry


def create_test_data(hours=24, start_value=20.0):
    """Create simple test data with configurable parameters."""
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i) for i in range(hours)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'value': [start_value + i * 0.5 for i in range(hours)],
        'unit': '°C'
    })


def test_temporal_mean():
    """Test temporal mean tool through registry."""
    print("\nTesting TemporalMeanTool...")
    
    data = create_test_data()
    registry = get_registry()
    tool = registry.get_tool('temporal_mean')
    
    # Test mean operation
    result = tool.execute(data, operation='mean')
    assert result.success, f"Mean failed: {result.error_message}"
    assert result.value is not None
    assert result.unit == '°C'
    assert 'sample_size' in result.metadata
    assert result.metadata['operation'] == 'mean'
    
    print(f"✓ Mean: {result.value:.2f} {result.unit}")
    print(f"✓ Sample size: {result.metadata['sample_size']}")
    
    # Test min operation
    result_min = tool.execute(data, operation='min')
    assert result_min.success, f"Min failed: {result_min.error_message}"
    assert result_min.value == 20.0
    print(f"✓ Min: {result_min.value:.2f}")
    
    # Test max operation
    result_max = tool.execute(data, operation='max')
    assert result_max.success, f"Max failed: {result_max.error_message}"
    assert result_max.value == 31.5  # 20.0 + 23 * 0.5
    print(f"✓ Max: {result_max.value:.2f}")
    
    # Test execution time is tracked
    assert result.execution_time_ms > 0
    print(f"✓ Execution time: {result.execution_time_ms:.2f}ms")


def test_temporal_aggregation():
    """Test temporal aggregation tool."""
    print("\nTesting TemporalAggregationTool...")
    
    # Create 3 days of data
    data = create_test_data(hours=72)
    registry = get_registry()
    tool = registry.get_tool('temporal_aggregation')
    
    # Test daily aggregation
    result = tool.execute(data, aggregation_level='daily', operation='mean')
    
    assert result.success, f"Failed: {result.error_message}"
    assert isinstance(result.value, list)
    assert result.metadata['num_periods'] == 3
    assert result.metadata['aggregation_level'] == 'daily'
    assert result.metadata['operation'] == 'mean'
    assert 'overall_aggregate' in result.metadata
    
    print(f"✓ Aggregated {result.metadata['num_periods']} periods")
    print(f"✓ Level: {result.metadata['aggregation_level']}")
    print(f"✓ Overall aggregate: {result.metadata['overall_aggregate']:.2f}")
    
    # Test hourly aggregation
    result_hourly = tool.execute(data, aggregation_level='hourly', operation='max')
    assert result_hourly.success
    assert result_hourly.metadata['aggregation_level'] == 'hourly'
    print(f"✓ Hourly aggregation with max: {result_hourly.metadata['num_periods']} periods")
    
    # Test weekly aggregation
    result_weekly = tool.execute(data, aggregation_level='weekly', operation='min')
    assert result_weekly.success
    print(f"✓ Weekly aggregation with min: {result_weekly.metadata['num_periods']} periods")


def test_spatial_comparison():
    """Test spatial comparison tool."""
    print("\nTesting SpatialComparisonTool...")
    
    # Create multi-location data with different values
    data1 = create_test_data(start_value=20.0)
    data1['location'] = 'Room201'
    
    data2 = create_test_data(start_value=22.0)
    data2['location'] = 'Room202'
    
    data3 = create_test_data(start_value=18.0)
    data3['location'] = 'Room203'
    
    data = pd.concat([data1, data2, data3], ignore_index=True)
    
    registry = get_registry()
    tool = registry.get_tool('spatial_comparison')
    
    # Test mean comparison
    result = tool.execute(data, operation='mean')
    
    assert result.success, f"Failed: {result.error_message}"
    assert len(result.value) == 3
    assert result.metadata['num_locations'] == 3
    assert result.metadata['operation'] == 'mean'
    
    # Verify ranking order (highest to lowest)
    assert result.value[0]['rank'] == 1
    assert result.value[1]['rank'] == 2
    assert result.value[2]['rank'] == 3
    
    # Verify percent_of_highest exists
    assert 'percent_of_highest' in result.value[0]
    assert result.value[0]['percent_of_highest'] == 100.0
    
    print(f"✓ Compared {result.metadata['num_locations']} locations")
    for loc in result.value:
        print(f"  {loc['location']}: Rank {loc['rank']}, Value: {loc['value']:.2f}, {loc['percent_of_highest']:.1f}% of highest")
    
    # Test max comparison
    result_max = tool.execute(data, operation='max')
    assert result_max.success
    print(f"✓ Max comparison successful")
    
    # Test min comparison
    result_min = tool.execute(data, operation='min')
    assert result_min.success
    print(f"✓ Min comparison successful")


def test_statistical_summary():
    """Test statistical summary tool."""
    print("\nTesting StatisticalSummaryTool...")
    
    data = create_test_data(hours=100)
    registry = get_registry()
    tool = registry.get_tool('statistical_summary')
    
    result = tool.execute(data)
    
    assert result.success, f"Failed: {result.error_message}"
    
    # Verify all required statistics are present
    required_stats = ['count', 'mean', 'std', 'min', 'q1', 'median', 'q3', 'max', 'skewness', 'kurtosis']
    for stat in required_stats:
        assert stat in result.value, f"Missing statistic: {stat}"
    
    # Verify count matches data length
    assert result.value['count'] == 100
    
    # Verify quartile ordering
    assert result.value['min'] <= result.value['q1']
    assert result.value['q1'] <= result.value['median']
    assert result.value['median'] <= result.value['q3']
    assert result.value['q3'] <= result.value['max']
    
    print(f"✓ Summary generated with {len(result.value)} statistics")
    print(f"  Count: {result.value['count']}")
    print(f"  Mean: {result.value['mean']:.2f}")
    print(f"  Median: {result.value['median']:.2f}")
    print(f"  Std Dev: {result.value['std']:.2f}")
    print(f"  Range: [{result.value['min']:.2f}, {result.value['max']:.2f}]")
    print(f"  Skewness: {result.value['skewness']:.4f}")
    print(f"  Kurtosis: {result.value['kurtosis']:.4f}")


def test_registry():
    """Test registry functions."""
    print("\nTesting ToolRegistry...")
    
    registry = get_registry()
    
    # Test list_tools
    tools = registry.list_tools()
    assert len(tools) == 4, f"Expected 4 tools, got {len(tools)}"
    
    # Verify each tool has required metadata
    for tool_info in tools:
        assert 'name' in tool_info
        assert 'description' in tool_info
        assert 'parameters' in tool_info
    
    print(f"✓ Registry contains {len(tools)} tools")
    for tool_info in tools:
        print(f"  - {tool_info['name']}: {tool_info['description']}")
        if tool_info['parameters']:
            print(f"    Parameters: {', '.join(tool_info['parameters'])}")
    
    # Test get_tool
    tool = registry.get_tool('temporal_mean')
    assert tool is not None
    assert tool.name == 'temporal_mean'
    print(f"✓ get_tool('temporal_mean') successful")
    
    # Test get_tool_by_operation
    tool = registry.get_tool_by_operation('mean')
    assert tool is not None
    assert tool.name == 'temporal_mean'
    print(f"✓ get_tool_by_operation('mean') returns temporal_mean")
    
    tool = registry.get_tool_by_operation('aggregation')
    assert tool is not None
    assert tool.name == 'temporal_aggregation'
    print(f"✓ get_tool_by_operation('aggregation') returns temporal_aggregation")
    
    tool = registry.get_tool_by_operation('comparison')
    assert tool is not None
    assert tool.name == 'spatial_comparison'
    print(f"✓ get_tool_by_operation('comparison') returns spatial_comparison")
    
    tool = registry.get_tool_by_operation('summary')
    assert tool is not None
    assert tool.name == 'statistical_summary'
    print(f"✓ get_tool_by_operation('summary') returns statistical_summary")
    
    # Test non-existent tool
    tool = registry.get_tool('non_existent')
    assert tool is None
    print(f"✓ get_tool returns None for non-existent tool")


def test_error_handling():
    """Test error handling across tools."""
    print("\nTesting Error Handling...")
    
    registry = get_registry()
    
    # Test with missing columns
    invalid_data = pd.DataFrame({'value': [1, 2, 3]})
    
    tool = registry.get_tool('temporal_mean')
    result = tool.execute(invalid_data)
    assert result.success is False
    assert 'Missing required columns' in result.error_message
    print(f"✓ TemporalMeanTool handles missing columns")
    
    tool = registry.get_tool('temporal_aggregation')
    result = tool.execute(invalid_data, aggregation_level='daily')
    assert result.success is False
    print(f"✓ TemporalAggregationTool handles missing columns")
    
    tool = registry.get_tool('spatial_comparison')
    result = tool.execute(invalid_data)
    assert result.success is False
    print(f"✓ SpatialComparisonTool handles missing columns")
    
    tool = registry.get_tool('statistical_summary')
    result = tool.execute(invalid_data)
    assert result.success is False
    print(f"✓ StatisticalSummaryTool handles missing columns")
    
    # Test invalid aggregation level
    valid_data = create_test_data()
    tool = registry.get_tool('temporal_aggregation')
    result = tool.execute(valid_data, aggregation_level='monthly')
    assert result.success is False
    assert 'Invalid aggregation_level' in result.error_message
    print(f"✓ TemporalAggregationTool handles invalid aggregation level")
    
    # Test invalid operation for temporal_mean
    tool = registry.get_tool('temporal_mean')
    result = tool.execute(valid_data, operation='invalid')
    assert result.success is False
    assert 'Invalid operation' in result.error_message
    print(f"✓ TemporalMeanTool handles invalid operation")
    
    # Test invalid operation for spatial_comparison
    data_with_location = create_test_data()
    data_with_location['location'] = 'Room201'
    tool = registry.get_tool('spatial_comparison')
    result = tool.execute(data_with_location, operation='median')
    assert result.success is False
    assert 'Invalid operation' in result.error_message
    print(f"✓ SpatialComparisonTool handles invalid operation")


def test_edge_cases():
    """Test edge cases."""
    print("\nTesting Edge Cases...")
    
    registry = get_registry()
    
    # Test with single data point
    single_data = pd.DataFrame({
        'timestamp': [datetime(2025, 1, 1, tzinfo=timezone.utc)],
        'value': [25.0],
        'unit': '°C'
    })
    
    tool = registry.get_tool('temporal_mean')
    result = tool.execute(single_data, operation='mean')
    assert result.success
    assert result.value == 25.0
    print(f"✓ TemporalMeanTool handles single data point")
    
    # Test with single location
    single_location = single_data.copy()
    single_location['location'] = 'Room201'
    
    tool = registry.get_tool('spatial_comparison')
    result = tool.execute(single_location, operation='mean')
    assert result.success
    assert len(result.value) == 1
    assert result.value[0]['rank'] == 1
    print(f"✓ SpatialComparisonTool handles single location")
    
    # Test with very small dataset for summary
    tool = registry.get_tool('statistical_summary')
    result = tool.execute(single_data)
    assert result.success
    print(f"✓ StatisticalSummaryTool handles small dataset")


if __name__ == "__main__":
    print("="*60)
    print("Analytics Module Integration Tests")
    print("="*60)
    
    try:
        test_registry()
        test_temporal_mean()
        test_temporal_aggregation()
        test_spatial_comparison()
        test_statistical_summary()
        test_error_handling()
        test_edge_cases()
        
        print("\n" + "="*60)
        print("✓ All integration tests passed!")
        print("="*60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)