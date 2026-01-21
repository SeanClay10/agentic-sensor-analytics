"""
LLM module for natural language understanding and explanation generation.

This module provides the interface for interacting with Large Language Models
to extract structured task specifications from natural language queries and
generate human-readable explanations of analytics results.

Architecture:
- Interface: Abstract base class defining LLM contract
- Implementation: Ollama-based local LLM with retry logic
- Prompts: Centralized prompt templates for consistency
- Parser: Robust JSON extraction and validation
- Config: YAML-based configuration management

Example Usage:
    from llm import OllamaLLM, SystemContext
    from datetime import datetime, timedelta
    
    # Initialize LLM with config
    llm = OllamaLLM.from_config('llm_config.yaml')
    
    # Create system context
    context = SystemContext(
        available_sensors=['temperature', 'humidity', 'co2'],
        available_locations=['Room201', 'Room202'],
        time_range=(datetime.now() - timedelta(days=30), datetime.now())
    )
    
    # Extract intent from natural language
    task_spec = llm.extract_intent(
        "What was the average temperature in Room201 yesterday?",
        context
    )
    
    # Generate explanation of results
    explanation = llm.explain_results(query, task_spec, results)
"""

from .interface import (
    LLMInterface,
    TaskSpecification,
    IntentType,
    Operation,
    AggregationLevel,
    LLMError,
    LLMParseError,
    LLMGenerationError
)

from .prompts import (
    PromptTemplates,
    SystemContext
)

from .parser import (
    TaskSpecificationParser,
    RelativeDateParser
)

from .local_llm import OllamaLLM

from .config import (
    LLMConfig,
    LLMSettings,
    LoggingSettings,
    PerformanceSettings
)

__all__ = [
    # Main LLM implementation
    'OllamaLLM',
    
    # Interface and base types
    'LLMInterface',
    'TaskSpecification',
    'IntentType',
    'Operation',
    'AggregationLevel',
    
    # Errors
    'LLMError',
    'LLMParseError',
    'LLMGenerationError',
    
    # Prompts and context
    'PromptTemplates',
    'SystemContext',
    
    # Parsing utilities
    'TaskSpecificationParser',
    'RelativeDateParser',
    
    # Configuration
    'LLMConfig',
    'LLMSettings',
    'LoggingSettings',
    'PerformanceSettings',
]

__version__ = '0.1.0'
__author__ = 'Sean Clayton'
__description__ = 'LLM interface for agentic smart building analytics'