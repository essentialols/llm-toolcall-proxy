#!/usr/bin/env python3
"""
Factory for creating appropriate tool call converters based on model
"""

from typing import Dict, Any, Optional, List
from .base import ToolCallConverter, StreamingToolCallHandler, PassThroughConverter
from .glm import GLMToolCallConverter, GLMStreamingHandler
from .openai import OpenAIToolCallConverter, OpenAIStreamingHandler
from .claude import ClaudeToolCallConverter, ClaudeStreamingHandler
from .deepseek import DeepSeekToolCallConverter, DeepSeekStreamingHandler


class ConverterFactory:
    """Factory class for creating model-specific tool call converters"""
    
    def __init__(self):
        # Register available converters (order matters - most specific first)
        self._converters = [
            DeepSeekToolCallConverter(),
            GLMToolCallConverter(),
            OpenAIToolCallConverter(),
            ClaudeToolCallConverter(),
            PassThroughConverter(),  # Fallback - should be last
        ]
    
    def get_converter(self, model_name: str) -> ToolCallConverter:
        """Get appropriate converter for the given model"""
        if not model_name:
            return PassThroughConverter()
        
        for converter in self._converters:
            if converter.can_handle_model(model_name):
                return converter
        
        # Fallback to pass-through
        return PassThroughConverter()
    
    def get_streaming_handler(self, model_name: str) -> StreamingToolCallHandler:
        """Get appropriate streaming handler for the given model"""
        converter = self.get_converter(model_name)
        
        # Return model-specific streaming handler if available
        if isinstance(converter, DeepSeekToolCallConverter):
            return DeepSeekStreamingHandler()
        elif isinstance(converter, GLMToolCallConverter):
            return GLMStreamingHandler()
        elif isinstance(converter, OpenAIToolCallConverter):
            return OpenAIStreamingHandler()
        elif isinstance(converter, ClaudeToolCallConverter):
            return ClaudeStreamingHandler()
        
        # Generic streaming handler for other converters
        return StreamingToolCallHandler(converter)
    
    def detect_model_from_response(self, response_data: Dict[str, Any]) -> Optional[str]:
        """Try to detect model name from response data"""
        if isinstance(response_data, dict):
            # Check common locations for model name
            model = response_data.get('model')
            if model:
                return model
            
            # Check in choices if available
            choices = response_data.get('choices', [])
            if choices and isinstance(choices[0], dict):
                # Some APIs might have model info in choice metadata
                pass
        
        return None
    
    def detect_model_from_request(self, request_data: Dict[str, Any]) -> Optional[str]:
        """Try to detect model name from request data"""
        if isinstance(request_data, dict):
            return request_data.get('model')
        return None
    
    def register_converter(self, converter: ToolCallConverter, priority: int = None):
        """Register a new converter with optional priority"""
        if priority is None:
            # Add before the PassThroughConverter (which should be last)
            self._converters.insert(-1, converter)
        else:
            self._converters.insert(priority, converter)
    
    def list_supported_models(self) -> List[str]:
        """List all supported model patterns"""
        supported = []
        for converter in self._converters:
            if hasattr(converter, 'GLM_MODEL_PATTERNS'):
                supported.extend(converter.GLM_MODEL_PATTERNS)
            elif hasattr(converter, 'OPENAI_MODEL_PATTERNS'):
                supported.extend(converter.OPENAI_MODEL_PATTERNS)
            elif hasattr(converter, 'CLAUDE_MODEL_PATTERNS'):
                supported.extend(converter.CLAUDE_MODEL_PATTERNS)
            elif not isinstance(converter, PassThroughConverter):
                supported.append(f"{converter.__class__.__name__} patterns")
        return supported


# Global factory instance
converter_factory = ConverterFactory()