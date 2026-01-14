"""
Simple test suite for analytics tools.
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analytics import get_registry


def create_test_data():
    """Create simple test data."""
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i) for i in range(24)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'value': [20.0 + i * 0.5 for i in range(24)],
        'unit': '°C'
    })


def test_temporal_mean():
    """Test temporal mean tool."""
    print("\nTesting TemporalMeanTool...")
    
    data = create_test_data()
    registry = get_registry()
    tool = registry.get_tool('temporal_mean')
    
    result = tool.execute(data)
    
    assert result.success, f"Failed: {result.error_message}"
    assert result.value is not None
    assert result.unit == '°C'
    assert 'sample_size' in result.metadata
    
    print(f"✓ Mean: {result.value:.2f} {result.unit}")
    print(f"✓ Sample size: {result.metadata['sample_size']}")
    print(f"✓ Execution time: {result.execution_time_ms:.2f}ms")


def test_temporal_aggregation():
    """Test temporal aggregation tool."""
    print("\nTesting TemporalAggregationTool...")
    
    data = create_test_data()
    registry = get_registry()
    tool = registry.get_tool('temporal_aggregation')
    
    result = tool.execute(data, aggregation_level='daily', operation='mean')
    
    assert result.success, f"Failed: {result.error_message}"
    assert isinstance(result.value, list)
    
    print(f"✓ Aggregated {result.metadata['num_periods']} periods")
    print(f"✓ Level: {result.metadata['aggregation_level']}")


def test_spatial_comparison():
    """Test spatial comparison tool."""
    print("\nTesting SpatialComparisonTool...")
    
    # Create multi-location data
    data1 = create_test_data()
    data1['location'] = 'Room201'
    
    data2 = create_test_data()
    data2['value'] = data2['value'] + 2
    data2['location'] = 'Room202'
    
    data = pd.concat([data1, data2], ignore_index=True)
    
    registry = get_registry()
    tool = registry.get_tool('spatial_comparison')
    
    result = tool.execute(data, operation='mean')
    
    assert result.success, f"Failed: {result.error_message}"
    assert len(result.value) == 2
    
    print(f"✓ Compared {result.metadata['num_locations']} locations")
    for loc in result.value:
        print(f"  {loc['location']}: Rank {loc['rank']}")


def test_statistical_summary():
    """Test statistical summary tool."""
    print("\nTesting StatisticalSummaryTool...")
    
    data = create_test_data()
    registry = get_registry()
    tool = registry.get_tool('statistical_summary')
    
    result = tool.execute(data)
    
    assert result.success, f"Failed: {result.error_message}"
    assert 'mean' in result.value
    assert 'median' in result.value
    
    print(f"✓ Summary generated")
    print(f"  Mean: {result.value['mean']:.2f}")
    print(f"  Median: {result.value['median']:.2f}")


def test_registry():
    """Test registry functions."""
    print("\nTesting ToolRegistry...")
    
    registry = get_registry()
    
    tools = registry.list_tools()
    assert len(tools) == 4, f"Expected 4 tools, got {len(tools)}"
    
    tool = registry.get_tool_by_operation('mean')
    assert tool is not None
    
    print(f"✓ Registry contains {len(tools)} tools")
    for tool_info in tools:
        print(f"  - {tool_info['name']}")


if __name__ == "__main__":
    print("="*60)
    print("Analytics Tools Test Suite")
    print("="*60)
    
    try:
        test_registry()
        test_temporal_mean()
        test_temporal_aggregation()
        test_spatial_comparison()
        test_statistical_summary()
        
        print("\n" + "="*60)
        print("All tests passed!")
        print("="*60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)