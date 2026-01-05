"""
Parser for converting LLM outputs into structured TaskSpecification objects.
Handles JSON extraction, validation, and error recovery.
"""

import json
import re
from typing import Optional
from datetime import datetime, timedelta
from dateutil import parser as date_parser

from .interface import (
    TaskSpecification,
    IntentType,
    Operation,
    AggregationLevel,
    LLMParseError
)


class TaskSpecificationParser:
    """
    Parses LLM outputs into validated TaskSpecification objects.
    Handles common LLM output issues like markdown wrapping, formatting errors, etc.
    """
    
    @staticmethod
    def parse(llm_output: str) -> TaskSpecification:
        """
        Parse LLM output string into TaskSpecification.
        
        Args:
            llm_output: Raw string output from LLM
            
        Returns:
            Validated TaskSpecification object
            
        Raises:
            LLMParseError: If parsing fails
        """
        try:
            # Extract JSON from potential markdown wrapping
            json_str = TaskSpecificationParser._extract_json(llm_output)
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Normalize field values
            normalized_data = TaskSpecificationParser._normalize_data(data)
            
            # Validate and construct TaskSpecification
            task_spec = TaskSpecification.model_validate(normalized_data)
            
            return task_spec
            
        except json.JSONDecodeError as e:
            raise LLMParseError(f"Invalid JSON in LLM output: {e}")
        except ValueError as e:
            raise LLMParseError(f"Validation error: {e}")
        except Exception as e:
            raise LLMParseError(f"Unexpected parsing error: {e}")
    
    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Extract JSON from text, handling markdown code blocks and other wrapping.
        """
        text = text.strip()
        
        # Remove markdown code blocks
        if text.startswith("```"):
            # Find the end of the opening backticks line
            first_newline = text.find('\n')
            # Find the closing backticks
            last_backticks = text.rfind("```")
            
            if first_newline != -1 and last_backticks != -1:
                text = text[first_newline+1:last_backticks].strip()
        
        # Find JSON object boundaries
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        raise LLMParseError("No JSON object found in LLM output")
    
    @staticmethod
    def _normalize_data(data: dict) -> dict:
        """
        Normalize and clean extracted data before validation.
        Handles common LLM output variations.
        """
        normalized = data.copy()
        
        # Normalize intent_type
        if 'intent_type' in normalized:
            intent = normalized['intent_type'].lower().strip()
            # Handle variations
            intent_mapping = {
                'simple_query': 'query',
                'single_query': 'query',
                'temporal_query': 'query',
                'compare': 'comparison',
                'comparison_query': 'comparison',
                'temporal_aggregation': 'aggregation',
                'aggregate': 'aggregation'
            }
            normalized['intent_type'] = intent_mapping.get(intent, intent)
        
        # Normalize operation
        if 'operation' in normalized:
            op = normalized['operation'].lower().strip()
            # Handle common variations
            operation_mapping = {
                'average': 'mean',
                'avg': 'mean',
                'maximum': 'max',
                'minimum': 'min',
                'total': 'sum',
                'standard_deviation': 'std',
                'stddev': 'std',
                'num': 'count',
                'number': 'count'
            }
            normalized['operation'] = operation_mapping.get(op, op)
        
        # Normalize aggregation_level
        if 'aggregation_level' in normalized:
            agg = normalized['aggregation_level']
            if agg and isinstance(agg, str):
                agg = agg.lower().strip()
                # Handle "null", "none", etc.
                if agg in ['null', 'none', 'n/a', '']:
                    normalized['aggregation_level'] = None
                else:
                    normalized['aggregation_level'] = agg
        
        # Parse datetime strings
        if 'start_time' in normalized and isinstance(normalized['start_time'], str):
            normalized['start_time'] = TaskSpecificationParser._parse_datetime(
                normalized['start_time']
            )
        
        if 'end_time' in normalized and isinstance(normalized['end_time'], str):
            normalized['end_time'] = TaskSpecificationParser._parse_datetime(
                normalized['end_time']
            )
        
        # Ensure confidence is float
        if 'confidence' in normalized:
            normalized['confidence'] = float(normalized['confidence'])
        
        return normalized
    
    @staticmethod
    def _parse_datetime(dt_str: str) -> datetime:
        """
        Parse datetime string.
        Handles ISO format, common variations, and relative dates.
        """
        dt_str = dt_str.strip()
        
        # Try ISO format first
        try:
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        except ValueError:
            pass
        
        # Try dateutil parser (handles many formats)
        try:
            return date_parser.parse(dt_str)
        except Exception:
            raise LLMParseError(f"Cannot parse datetime: {dt_str}")
    
    @staticmethod
    def validate_against_context(
        task_spec: TaskSpecification,
        available_sensors: list[str],
        available_locations: list[str],
        time_range: tuple[datetime, datetime]
    ) -> list[str]:
        """
        Validate task specification against system context.
        Returns list of validation errors (empty if valid).
        
        Args:
            task_spec: Parsed task specification
            available_sensors: Valid sensor types
            available_locations: Valid location identifiers
            time_range: (min_date, max_date) for available data
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate sensor type
        if task_spec.sensor_type not in available_sensors:
            errors.append(
                f"Unknown sensor type '{task_spec.sensor_type}'. "
                f"Available: {', '.join(available_sensors)}"
            )
        
        # Validate locations
        locations = task_spec.get_locations_list()
        for loc in locations:
            if loc not in available_locations:
                errors.append(
                    f"Unknown location '{loc}'. "
                    f"Available: {', '.join(available_locations[:5])}..."
                )
        
        # Validate time range
        min_date, max_date = time_range
        if task_spec.start_time < min_date:
            errors.append(
                f"Start time {task_spec.start_time.date()} is before available data "
                f"(earliest: {min_date.date()})"
            )
        if task_spec.end_time > max_date:
            errors.append(
                f"End time {task_spec.end_time.date()} is after available data "
                f"(latest: {max_date.date()})"
            )
        
        # Validate time range makes sense
        if task_spec.end_time <= task_spec.start_time:
            errors.append("End time must be after start time")
            
        return errors


class RelativeDateParser:
    """Helper class for parsing relative date expressions."""
    
    @staticmethod
    def parse_relative_date(expression: str, reference_date: Optional[datetime] = None) -> tuple[datetime, datetime]:
        """
        Parse relative date expressions like "today", "yesterday", "last week".
        
        Args:
            expression: Relative date expression
            reference_date: Reference date (default: now)
            
        Returns:
            Tuple of (start_datetime, end_datetime)
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        expression = expression.lower().strip()
        
        # Today
        if expression == "today":
            start = reference_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(hour=23, minute=59, second=59)
            return start, end
        
        # Yesterday
        if expression == "yesterday":
            yesterday = reference_date - timedelta(days=1)
            start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(hour=23, minute=59, second=59)
            return start, end
        
        # Last week
        if expression == "last week":
            end = reference_date.replace(hour=23, minute=59, second=59)
            start = (reference_date - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
            return start, end
        
        # Last month
        if expression == "last month":
            end = reference_date.replace(hour=23, minute=59, second=59)
            start = (reference_date - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0)
            return start, end
        
        # Last N days
        match = re.match(r'last (\d+) days?', expression)
        if match:
            days = int(match.group(1))
            end = reference_date.replace(hour=23, minute=59, second=59)
            start = (reference_date - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
            return start, end
        
        raise ValueError(f"Cannot parse relative date expression: {expression}")