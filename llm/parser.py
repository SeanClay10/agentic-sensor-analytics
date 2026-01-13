"""
Parser for converting LLM outputs into structured TaskSpecification objects.
Handles JSON extraction, validation, and error recovery.
"""

import json
import re
from typing import Optional
from datetime import datetime, timedelta, timezone
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
            raise LLMParseError(f"Invalid JSON in LLM output: {e}\n\nOutput:\n{llm_output}")
        except ValueError as e:
            raise LLMParseError(f"Validation error: {e}\n\nParsed data: {data if 'data' in locals() else 'N/A'}")
        except Exception as e:
            raise LLMParseError(f"Unexpected parsing error: {e}\n\nOutput:\n{llm_output}")
    
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
        
        # Remove any leading/trailing text that isn't part of JSON
        # Find the first { and last }
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            raise LLMParseError("No JSON object found in LLM output")
        
        json_str = text[start_idx:end_idx+1]
        
        return json_str
    
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
                'simple': 'query',
                'compare': 'comparison',
                'comparison_query': 'comparison',
                'spatial_comparison': 'comparison',
                'temporal_aggregation': 'aggregation',
                'aggregate': 'aggregation',
                'time_series': 'aggregation'
            }
            normalized['intent_type'] = intent_mapping.get(intent, intent)
        
        # Normalize sensor_type
        if 'sensor_type' in normalized:
            sensor = normalized['sensor_type'].lower().strip()
            # Handle common variations
            sensor_mapping = {
                'temp': 'temperature',
                'co2_concentration': 'co2',
                'carbon_dioxide': 'co2',
                'power': 'energy',
                'occupant': 'occupancy',
                'people_count': 'occupancy'
            }
            normalized['sensor_type'] = sensor_mapping.get(sensor, sensor)
        
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
                'number': 'count',
                'cnt': 'count'
            }
            normalized['operation'] = operation_mapping.get(op, op)
        
        # Normalize aggregation_level
        if 'aggregation_level' in normalized:
            agg = normalized['aggregation_level']
            if agg and isinstance(agg, str):
                agg = agg.lower().strip()
                # Handle "null", "none", etc.
                if agg in ['null', 'none', 'n/a', '', 'na']:
                    normalized['aggregation_level'] = None
                else:
                    # Handle variations
                    agg_mapping = {
                        'hour': 'hourly',
                        'day': 'daily',
                        'week': 'weekly'
                    }
                    normalized['aggregation_level'] = agg_mapping.get(agg, agg)
        
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
            try:
                normalized['confidence'] = float(normalized['confidence'])
            except (ValueError, TypeError):
                # Default to 0.8 if confidence parsing fails
                normalized['confidence'] = 0.8
        else:
            # Default confidence if not provided
            normalized['confidence'] = 0.85
        
        return normalized
    
    @staticmethod
    def _parse_datetime(dt_str: str) -> datetime:
        dt_str = dt_str.strip()
        try:
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
        
        try:
            dt = date_parser.parse(dt_str)
            # FORCE AWARENESS if date_parser returns a naive object
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            raise LLMParseError(f"Cannot parse datetime: '{dt_str}'")
    
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
                f"Available sensors: {', '.join(available_sensors)}"
            )
        
        # Validate locations
        locations = task_spec.get_locations_list()
        invalid_locations = [loc for loc in locations if loc not in available_locations]
        
        if invalid_locations:
            # Show a few suggestions if there are many locations
            suggestions = available_locations[:5]
            errors.append(
                f"Unknown location(s): {', '.join(invalid_locations)}. "
                f"Available locations include: {', '.join(suggestions)}..."
            )
        
        # Validate time range
        min_date, max_date = time_range
        if task_spec.start_time < min_date:
            errors.append(
                f"Start time ({task_spec.start_time.date()}) is before available data. "
                f"Data starts from {min_date.date()}."
            )
        if task_spec.end_time > max_date:
            errors.append(
                f"End time ({task_spec.end_time.date()}) is after available data. "
                f"Data available until {max_date.date()}."
            )
        
        # Validate time range makes sense (should be caught by Pydantic, but double-check)
        if task_spec.end_time <= task_spec.start_time:
            errors.append("End time must be after start time.")
        

        return errors


class RelativeDateParser:
    """Helper class for parsing relative date expressions."""
    
    @staticmethod
    def parse_relative_date(
        expression: str,
        reference_date: Optional[datetime] = None
    ) -> tuple[datetime, datetime]:
        """
        Parse relative date expressions like "today", "yesterday", "last week".
        
        Args:
            expression: Relative date expression
            reference_date: Reference date (default: now)
            
        Returns:
            Tuple of (start_datetime, end_datetime)
        """
        if reference_date is None:
            reference_date = datetime.now(timezone.utc)
        
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
            if end > reference_date:
                end = reference_date
            return start, end
        
        # This week
        if expression in ["this week", "current week"]:
            # Start of week (Monday)
            days_since_monday = reference_date.weekday()
            start = (reference_date - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            end = reference_date.replace(hour=23, minute=59, second=59)
            return start, end
        
        # Last week
        if expression == "last week":
            end = reference_date.replace(hour=23, minute=59, second=59)
            start = (reference_date - timedelta(days=7)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return start, end
        
        # Last month
        if expression == "last month":
            end = reference_date.replace(hour=23, minute=59, second=59)
            start = (reference_date - timedelta(days=30)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return start, end
        
        # Last N days
        match = re.match(r'last (\d+) days?', expression)
        if match:
            days = int(match.group(1))
            end = reference_date.replace(hour=23, minute=59, second=59)
            start = (reference_date - timedelta(days=days)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return start, end
        
        # Past N hours
        match = re.match(r'past (\d+) hours?', expression)
        if match:
            hours = int(match.group(1))
            end = reference_date
            start = reference_date - timedelta(hours=hours)
            return start, end
        
        raise ValueError(f"Cannot parse relative date expression: '{expression}'")