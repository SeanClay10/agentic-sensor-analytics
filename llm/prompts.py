"""
Prompt templates for LLM intent extraction and result explanation.
"""

from typing import List, Dict
from datetime import datetime


class PromptTemplates:
    """Centralized prompt templates for the LLM component."""
    
    @staticmethod
    def get_intent_extraction_prompt(
        user_query: str,
        available_sensors: List[str],
        available_locations: List[str],
        time_range: tuple[datetime, datetime]
    ) -> str:
        """
        Generate prompt for extracting structured task specification from natural language.
        
        Args:
            user_query: The user's natural language query
            available_sensors: List of valid sensor types in the system
            available_locations: List of valid locations
            time_range: Tuple of (earliest_datetime, latest_datetime) for available data
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a task extraction assistant for a smart building analytics system. 
Your job is to convert natural language queries into structured JSON task specifications.

SYSTEM CONTEXT:
Available sensors: {', '.join(available_sensors)}
Available locations: {', '.join(available_locations)}
Data available from: {time_range[0].strftime('%Y-%m-%d')} to {time_range[1].strftime('%Y-%m-%d')}
Current date: {datetime.now().strftime('%Y-%m-%d')}

USER QUERY:
{user_query}

INSTRUCTIONS:
Extract the following information and return ONLY valid JSON (no markdown, no explanations):

{{
  "intent_type": "<query|comparison|aggregation>",
  "sensor_type": "<temperature|humidity|co2|energy|occupancy>",
  "location": "<single location string OR list of locations for comparison>",
  "start_time": "<ISO 8601 datetime>",
  "end_time": "<ISO 8601 datetime>",
  "operation": "<mean|max|min|sum|std|count>",
  "aggregation_level": "<hourly|daily|weekly|null>",
  "confidence": <0.0-1.0 confidence score>
}}

INTENT TYPES:
- query: Simple statistical query for one location
- comparison: Compare metrics across multiple locations
- aggregation: Temporal aggregation (hourly/daily/weekly summaries)

Return ONLY the JSON object.
"""
        return prompt
    
    @staticmethod
    def get_result_explanation_prompt(
        original_query: str,
        task_spec: Dict,
        results: List[Dict]
    ) -> str:
        """
        Generate prompt for explaining analytics results in natural language.
        
        Args:
            original_query: The user's original question
            task_spec: The structured task specification
            results: List of analytics results with metadata
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are explaining analytics results from a smart building system.

ORIGINAL USER QUERY:
{original_query}

TASK SPECIFICATION:
{task_spec}

ANALYTICS RESULTS:
{results}

INSTRUCTIONS:
Provide a clear, concise natural language explanation of the results.

GUIDELINES:
1. Directly answer the user's question
2. Include the numerical result with units
3. Add relevant context (sample size, time range)
4. Mention any data quality notes if present
5. Keep response to 2-3 sentences
6. Do NOT hallucinate or add information not in the results
7. Use natural, conversational language

Now explain the results above:
"""
        return prompt

    @staticmethod
    def get_error_explanation_prompt(
        user_query: str,
        errors: List[str]
    ) -> str:
        """
        Generate prompt for explaining validation errors to users.
        
        Args:
            user_query: The user's query
            errors: List of validation error messages
            
        Returns:
            Formatted prompt for error explanation
        """
        prompt = f"""Explain validation errors to the user in a helpful way.

USER QUERY:
{user_query}

VALIDATION ERRORS:
{errors}

Generate a user-friendly explanation that:
1. Explains what went wrong clearly
2. Suggests how to fix the query
3. Remains polite and helpful
4. Uses simple language (no technical jargon)

Return only the explanation, 2-3 sentences maximum.
"""
        return prompt


class SystemContext:
    """Container for system context passed to prompts."""
    
    def __init__(
        self,
        available_sensors: List[str],
        available_locations: List[str],
        time_range: tuple[datetime, datetime]
    ):
        self.available_sensors = available_sensors
        self.available_locations = available_locations
        self.time_range = time_range
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "available_sensors": self.available_sensors,
            "available_locations": self.available_locations,
            "time_range": [
                self.time_range[0].isoformat(),
                self.time_range[1].isoformat()
            ]
        }