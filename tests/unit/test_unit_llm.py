"""
Unit tests for LLM module components.
Tests individual components in isolation with mocking.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import yaml

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm import (
    TaskSpecification,
    IntentType,
    Operation,
    AggregationLevel,
    SystemContext,
    LLMConfig,
    LLMSettings,
    LoggingSettings,
    PerformanceSettings,
    PromptTemplates,
    TaskSpecificationParser,
    RelativeDateParser,
    LLMError,
    LLMParseError,
    LLMGenerationError
)


class TestTaskSpecification:
    """Unit tests for TaskSpecification model."""
    
    @pytest.fixture
    def valid_query_spec(self):
        """Valid simple query specification."""
        return {
            "intent_type": "query",
            "sensor_type": "temperature",
            "location": "Room201",
            "start_time": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "end_time": datetime(2025, 1, 2, tzinfo=timezone.utc),
            "operation": "mean",
            "aggregation_level": None,
            "confidence": 0.95
        }
    
    @pytest.fixture
    def valid_comparison_spec(self):
        """Valid comparison specification."""
        return {
            "intent_type": "comparison",
            "sensor_type": "humidity",
            "location": ["Room201", "Room202"],
            "start_time": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "end_time": datetime(2025, 1, 7, tzinfo=timezone.utc),
            "operation": "mean",
            "aggregation_level": None,
            "confidence": 0.9
        }
    
    @pytest.fixture
    def valid_aggregation_spec(self):
        """Valid aggregation specification."""
        return {
            "intent_type": "aggregation",
            "sensor_type": "co2",
            "location": "Room305",
            "start_time": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "end_time": datetime(2025, 1, 31, tzinfo=timezone.utc),
            "operation": "mean",
            "aggregation_level": "daily",
            "confidence": 0.85
        }
    
    def test_valid_query_creation(self, valid_query_spec):
        """Test creating valid simple query."""
        spec = TaskSpecification(**valid_query_spec)
        
        assert spec.intent_type == IntentType.QUERY
        assert spec.sensor_type == "temperature"
        assert spec.location == "Room201"
        assert spec.operation == Operation.MEAN
        assert spec.aggregation_level is None
        assert spec.confidence == 0.95
    
    def test_valid_comparison_creation(self, valid_comparison_spec):
        """Test creating valid comparison query."""
        spec = TaskSpecification(**valid_comparison_spec)
        
        assert spec.intent_type == IntentType.COMPARISON
        assert spec.location == ["Room201", "Room202"]
        assert isinstance(spec.location, list)
    
    def test_valid_aggregation_creation(self, valid_aggregation_spec):
        """Test creating valid aggregation query."""
        spec = TaskSpecification(**valid_aggregation_spec)
        
        assert spec.intent_type == IntentType.AGGREGATION
        assert spec.aggregation_level == AggregationLevel.DAILY
    
    def test_end_time_validation(self, valid_query_spec):
        """Test that end_time must be after start_time."""
        valid_query_spec["end_time"] = datetime(2024, 12, 31, tzinfo=timezone.utc)
        
        with pytest.raises(ValueError, match="end_time must be after start_time"):
            TaskSpecification(**valid_query_spec)
    
    def test_confidence_bounds(self, valid_query_spec):
        """Test confidence value bounds."""
        # Test confidence > 1.0
        valid_query_spec["confidence"] = 1.5
        with pytest.raises(ValueError):
            TaskSpecification(**valid_query_spec)
        
        # Test confidence < 0.0
        valid_query_spec["confidence"] = -0.1
        with pytest.raises(ValueError):
            TaskSpecification(**valid_query_spec)
        
        # Test valid boundaries
        valid_query_spec["confidence"] = 0.0
        spec = TaskSpecification(**valid_query_spec)
        assert spec.confidence == 0.0
        
        valid_query_spec["confidence"] = 1.0
        spec = TaskSpecification(**valid_query_spec)
        assert spec.confidence == 1.0
    
    def test_comparison_requires_list(self, valid_comparison_spec):
        """Test that comparison intent requires list of locations."""
        valid_comparison_spec["location"] = "Room201"
        
        with pytest.raises(ValueError, match="Comparison queries require multiple locations"):
            TaskSpecification(**valid_comparison_spec)
    
    def test_query_requires_string(self, valid_query_spec):
        """Test that simple query requires single location string."""
        valid_query_spec["location"] = ["Room201", "Room202"]
        
        with pytest.raises(ValueError, match="Simple queries require single location"):
            TaskSpecification(**valid_query_spec)
    
    def test_get_locations_list_string(self, valid_query_spec):
        """Test get_locations_list with string location."""
        spec = TaskSpecification(**valid_query_spec)
        locations = spec.get_locations_list()
        
        assert isinstance(locations, list)
        assert locations == ["Room201"]
    
    def test_get_locations_list_array(self, valid_comparison_spec):
        """Test get_locations_list with list location."""
        spec = TaskSpecification(**valid_comparison_spec)
        locations = spec.get_locations_list()
        
        assert isinstance(locations, list)
        assert locations == ["Room201", "Room202"]
    
    def test_enum_validation(self, valid_query_spec):
        """Test enum field validation."""
        # Invalid intent_type
        valid_query_spec["intent_type"] = "invalid"
        with pytest.raises(ValueError):
            TaskSpecification(**valid_query_spec)
        
        # Invalid operation
        valid_query_spec["intent_type"] = "query"
        valid_query_spec["operation"] = "invalid"
        with pytest.raises(ValueError):
            TaskSpecification(**valid_query_spec)


class TestSystemContext:
    """Unit tests for SystemContext."""
    
    def test_creation(self):
        """Test SystemContext creation."""
        context = SystemContext(
            available_sensors=["temperature", "humidity"],
            available_locations=["Room201", "Room202"],
            time_range=(
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime(2025, 1, 31, tzinfo=timezone.utc)
            )
        )
        
        assert len(context.available_sensors) == 2
        assert len(context.available_locations) == 2
        assert isinstance(context.time_range, tuple)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        context = SystemContext(
            available_sensors=["temperature"],
            available_locations=["Room201"],
            time_range=(
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime(2025, 1, 2, tzinfo=timezone.utc)
            )
        )
        
        context_dict = context.to_dict()
        
        assert "available_sensors" in context_dict
        assert "available_locations" in context_dict
        assert "time_range" in context_dict
        assert len(context_dict["time_range"]) == 2


class TestLLMConfig:
    """Unit tests for LLMConfig and related settings."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()
        
        assert config.llm.model_name == "llama3.1:8b"
        assert config.llm.base_url == "http://localhost:11434"
        assert config.llm.temperature == 0.1
        assert config.llm.max_tokens == 4096
        assert config.llm.timeout == 30
        assert config.llm.max_retries == 3
        assert config.llm.min_confidence == 0.5
    
    def test_llm_settings_validation(self):
        """Test LLMSettings validation."""
        # Valid settings
        settings = LLMSettings(
            model_name="test-model",
            temperature=0.5,
            max_tokens=2048
        )
        assert settings.temperature == 0.5
        
        # Invalid temperature (too high)
        with pytest.raises(ValueError):
            LLMSettings(temperature=3.0)
        
        # Invalid max_tokens (non-positive)
        with pytest.raises(ValueError):
            LLMSettings(max_tokens=0)
    
    def test_logging_settings_validation(self):
        """Test LoggingSettings validation."""
        # Valid log level
        settings = LoggingSettings(level="DEBUG")
        assert settings.level == "DEBUG"
        
        # Invalid log level
        with pytest.raises(ValueError, match="Invalid log level"):
            LoggingSettings(level="INVALID")
        
        # Case insensitive
        settings = LoggingSettings(level="info")
        assert settings.level == "INFO"
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "llm": {
                "model_name": "custom-model",
                "temperature": 0.2
            }
        }
        
        config = LLMConfig.from_dict(config_dict)
        assert config.llm.model_name == "custom-model"
        assert config.llm.temperature == 0.2
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = LLMConfig()
        config_dict = config.to_dict()
        
        assert "llm" in config_dict
        assert "logging" in config_dict
        assert "performance" in config_dict
    
    def test_from_yaml(self):
        """Test loading from YAML file."""
        # Create temporary YAML file
        config_data = {
            "llm": {
                "model_name": "test-model",
                "temperature": 0.3
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = LLMConfig.from_yaml(temp_path)
            assert config.llm.model_name == "test-model"
            assert config.llm.temperature == 0.3
        finally:
            Path(temp_path).unlink()
    
    def test_from_yaml_file_not_found(self):
        """Test error when YAML file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            LLMConfig.from_yaml("nonexistent.yaml")
    
    def test_save_yaml(self):
        """Test saving to YAML file."""
        config = LLMConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config.save_yaml(config_path)
            
            assert config_path.exists()
            
            # Verify we can load it back
            loaded_config = LLMConfig.from_yaml(config_path)
            assert loaded_config.llm.model_name == config.llm.model_name


class TestPromptTemplates:
    """Unit tests for PromptTemplates."""
    
    def test_intent_extraction_prompt(self):
        """Test intent extraction prompt generation."""
        prompt = PromptTemplates.get_intent_extraction_prompt(
            user_query="What was the temperature yesterday?",
            available_sensors=["temperature", "humidity"],
            available_locations=["Room201"],
            time_range=(
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime(2025, 1, 31, tzinfo=timezone.utc)
            )
        )
        
        assert "temperature yesterday" in prompt.lower()
        assert "temperature" in prompt
        assert "humidity" in prompt
        assert "Room201" in prompt
        assert "2025-01-01" in prompt
        assert "2025-01-31" in prompt
    
    def test_result_explanation_prompt(self):
        """Test result explanation prompt generation."""
        task_spec = {
            "intent_type": "query",
            "sensor_type": "temperature",
            "location": "Room201"
        }
        
        results = [{"value": 22.5, "unit": "°C"}]
        
        prompt = PromptTemplates.get_result_explanation_prompt(
            original_query="What was the average temperature?",
            task_spec=task_spec,
            results=results
        )
        
        assert "average temperature" in prompt.lower()
        assert "22.5" in prompt
        assert "°C" in prompt
    
    def test_error_explanation_prompt(self):
        """Test error explanation prompt generation."""
        errors = ["Unknown sensor type 'invalid'"]
        
        prompt = PromptTemplates.get_error_explanation_prompt(
            user_query="Show me invalid sensor data",
            errors=errors
        )
        
        assert "invalid sensor" in prompt.lower()
        assert "unknown sensor" in prompt.lower()


class TestTaskSpecificationParser:
    """Unit tests for TaskSpecificationParser."""
    
    @pytest.fixture
    def valid_spec(self):
        """Create valid task specification."""
        return TaskSpecification(
            intent_type="query",
            sensor_type="temperature",
            location="Room201",
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
            operation="mean"
        )
    
    @pytest.fixture
    def system_context(self):
        """Create system context."""
        return (
            ["temperature", "humidity", "co2"],
            ["Room201", "Room202", "Room305"],
            (datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2025, 12, 31, tzinfo=timezone.utc))
        )
    
    def test_valid_specification(self, valid_spec, system_context):
        """Test validation passes for valid specification."""
        errors = TaskSpecificationParser.validate_against_context(
            task_spec=valid_spec,
            available_sensors=system_context[0],
            available_locations=system_context[1],
            time_range=system_context[2]
        )
        
        assert len(errors) == 0
    
    def test_invalid_sensor(self, valid_spec, system_context):
        """Test validation catches invalid sensor type."""
        valid_spec.sensor_type = "invalid_sensor"
        
        errors = TaskSpecificationParser.validate_against_context(
            task_spec=valid_spec,
            available_sensors=system_context[0],
            available_locations=system_context[1],
            time_range=system_context[2]
        )
        
        assert len(errors) > 0
        assert any("sensor type" in error.lower() for error in errors)
    
    def test_invalid_location(self, valid_spec, system_context):
        """Test validation catches invalid location."""
        valid_spec.location = "InvalidRoom"
        
        errors = TaskSpecificationParser.validate_against_context(
            task_spec=valid_spec,
            available_sensors=system_context[0],
            available_locations=system_context[1],
            time_range=system_context[2]
        )
        
        assert len(errors) > 0
        assert any("location" in error.lower() for error in errors)
    
    def test_time_before_available_data(self, valid_spec, system_context):
        """Test validation catches time before available data."""
        valid_spec.start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        valid_spec.end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)
        
        errors = TaskSpecificationParser.validate_against_context(
            task_spec=valid_spec,
            available_sensors=system_context[0],
            available_locations=system_context[1],
            time_range=system_context[2]
        )
        
        assert len(errors) > 0
        assert any("before available data" in error.lower() for error in errors)
    
    def test_time_after_available_data(self, valid_spec, system_context):
        """Test validation catches time after available data."""
        valid_spec.end_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        
        errors = TaskSpecificationParser.validate_against_context(
            task_spec=valid_spec,
            available_sensors=system_context[0],
            available_locations=system_context[1],
            time_range=system_context[2]
        )
        
        assert len(errors) > 0
        assert any("after available data" in error.lower() for error in errors)
    
    def test_multiple_errors(self, valid_spec, system_context):
        """Test validation catches multiple errors."""
        valid_spec.sensor_type = "invalid"
        valid_spec.location = "InvalidRoom"
        
        errors = TaskSpecificationParser.validate_against_context(
            task_spec=valid_spec,
            available_sensors=system_context[0],
            available_locations=system_context[1],
            time_range=system_context[2]
        )
        
        assert len(errors) >= 2


class TestRelativeDateParser:
    """Unit tests for RelativeDateParser."""
    
    def test_parse_today(self):
        """Test parsing 'today'."""
        ref_date = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        start, end = RelativeDateParser.parse_relative_date("today", ref_date)
        
        assert start.date() == ref_date.date()
        assert start.hour == 0
        assert end.hour == 23
        assert end.minute == 59
    
    def test_parse_yesterday(self):
        """Test parsing 'yesterday'."""
        ref_date = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        start, end = RelativeDateParser.parse_relative_date("yesterday", ref_date)
        
        assert start.date() == (ref_date - timedelta(days=1)).date()
        assert start.hour == 0
        assert end.hour == 23
    
    def test_parse_last_week(self):
        """Test parsing 'last week'."""
        ref_date = datetime(2025, 1, 15, tzinfo=timezone.utc)
        start, end = RelativeDateParser.parse_relative_date("last week", ref_date)
        
        assert (end - start).days == 7
        assert start == ref_date - timedelta(days=7)
    
    def test_parse_last_n_days(self):
        """Test parsing 'last N days'."""
        ref_date = datetime(2025, 1, 15, tzinfo=timezone.utc)
        start, end = RelativeDateParser.parse_relative_date("last 5 days", ref_date)
        
        assert (end.date() - start.date()).days == 5
    
    def test_parse_past_n_hours(self):
        """Test parsing 'past N hours'."""
        ref_date = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        start, end = RelativeDateParser.parse_relative_date("past 3 hours", ref_date)
        
        assert end == ref_date
        assert start == ref_date - timedelta(hours=3)
    
    def test_invalid_expression(self):
        """Test invalid expression raises error."""
        with pytest.raises(ValueError):
            RelativeDateParser.parse_relative_date("invalid expression")
    
    def test_case_insensitive(self):
        """Test parsing is case insensitive."""
        ref_date = datetime(2025, 1, 15, tzinfo=timezone.utc)
        
        start1, end1 = RelativeDateParser.parse_relative_date("TODAY", ref_date)
        start2, end2 = RelativeDateParser.parse_relative_date("today", ref_date)
        
        assert start1 == start2
        assert end1 == end2


class TestLLMExceptions:
    """Unit tests for LLM exception classes."""
    
    def test_llm_error(self):
        """Test LLMError exception."""
        error = LLMError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_llm_parse_error(self):
        """Test LLMParseError exception."""
        error = LLMParseError("Parse failed")
        assert str(error) == "Parse failed"
        assert isinstance(error, LLMError)
    
    def test_llm_generation_error(self):
        """Test LLMGenerationError exception."""
        error = LLMGenerationError("Generation failed")
        assert str(error) == "Generation failed"
        assert isinstance(error, LLMError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])