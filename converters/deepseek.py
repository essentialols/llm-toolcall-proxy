#!/usr/bin/env python3
"""
DeepSeek Coder V2 Lite tool call converter.

DeepSeek Coder V2 Lite outputs tool calls as JSON in its text content
rather than using the structured tool_calls field. This converter
detects and extracts those JSON tool calls.

Patterns handled:
1. {"name": "func_name", "arguments": {...}}  (bare JSON)
2. {"name": "func_name", "parameters": {...}}  (parameters variant)
3. ```json\n{"name": "...", ...}\n```  (markdown-wrapped)
4. <tool_call>{"name": "...", ...}</tool_call>  (XML-wrapped, from custom templates)
5. func_name({"key": "value"})  (function-call style)
"""

import json
import re
from typing import Dict, Any, List
from .base import ToolCallConverter, StreamingToolCallHandler


class DeepSeekToolCallConverter(ToolCallConverter):
    """Converts DeepSeek Coder V2 text-based tool calls to standard OpenAI format"""

    DEEPSEEK_MODEL_PATTERNS = [
        r'.*deepseek.*coder.*',
        r'.*deepseek.*v2.*lite.*',
        r'.*deepseek-coder.*',
    ]

    # Ordered by specificity (most specific first)
    TOOL_CALL_PATTERNS = [
        # DeepSeek native tokens: <｜tool▁call▁begin｜>function<｜tool▁sep｜>name\n```json\n{...}\n```<｜tool▁call▁end｜>
        (
            'native_ds',
            re.compile(
                r'<｜tool▁call▁begin｜>function<｜tool▁sep｜>(\w+)\s*```(?:json)?\s*(\{.*?\})\s*```\s*<｜tool▁call▁end｜>',
                re.DOTALL
            )
        ),
        # XML-wrapped: <tool_call>...</tool_call>
        (
            'xml',
            re.compile(
                r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
                re.DOTALL
            )
        ),
        # Markdown code block
        (
            'markdown',
            re.compile(
                r'```(?:json)?\s*(\{[^}]*"name"\s*:\s*"[^"]+?".*?\})\s*```',
                re.DOTALL
            )
        ),
        # Bare JSON with "name" + "arguments" or "parameters"
        (
            'bare_json',
            re.compile(
                r'(\{\s*"name"\s*:\s*"[^"]+?"\s*,\s*"(?:arguments|parameters)"\s*:\s*\{.*?\}\s*\})',
                re.DOTALL
            )
        ),
    ]

    # For detecting partial tool calls in streaming
    PARTIAL_PATTERNS = [
        re.compile(r'<tool_call>'),
        re.compile(r'\{\s*"name"\s*:\s*"[^"]*"?\s*,?\s*"?(?:arguments|parameters)'),
        re.compile(r'```json\s*\{'),
    ]

    COMPLETE_PATTERNS = [
        re.compile(r'<tool_call>.*?</tool_call>', re.DOTALL),
        re.compile(r'\{\s*"name"\s*:\s*"[^"]+?"\s*,\s*"(?:arguments|parameters)"\s*:\s*\{.*?\}\s*\}', re.DOTALL),
        re.compile(r'```(?:json)?\s*\{.*?\}\s*```', re.DOTALL),
    ]

    def can_handle_model(self, model_name: str) -> bool:
        if not model_name:
            return False
        model_lower = model_name.lower()
        for pattern in self.DEEPSEEK_MODEL_PATTERNS:
            if re.match(pattern, model_lower):
                return True
        return False

    def parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Extract tool calls from DeepSeek's text output."""
        tool_calls = []

        if not content:
            return tool_calls

        for fmt, pattern in self.TOOL_CALL_PATTERNS:
            for match in pattern.finditer(content):
                groups = match.groups()

                # Handle native DeepSeek format: group(1)=func_name, group(2)=args_json
                if fmt == 'native_ds' and len(groups) == 2:
                    func_name, args_json = groups
                    try:
                        args = json.loads(args_json)
                        tool_call = {
                            "id": f"ds-{hash(f'{func_name}_{len(tool_calls)}') % 1000000000}",
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": json.dumps(args, ensure_ascii=False)
                            }
                        }
                        tool_calls.append(tool_call)
                    except json.JSONDecodeError:
                        continue
                    continue

                # Standard format: group(1)=JSON with name+arguments
                json_str = groups[0]
                try:
                    obj = json.loads(json_str)
                    name = obj.get("name", "")
                    if not name:
                        continue

                    args = obj.get("arguments", obj.get("parameters", {}))
                    if isinstance(args, dict):
                        args_str = json.dumps(args, ensure_ascii=False)
                    elif isinstance(args, str):
                        args_str = args
                    else:
                        args_str = json.dumps(args)

                    tool_call = {
                        "id": f"ds-{hash(f'{name}_{len(tool_calls)}') % 1000000000}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args_str
                        }
                    }
                    tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    continue

            # If we found calls with a more specific pattern, stop
            if tool_calls:
                break

        return tool_calls

    def has_partial_tool_call(self, content: str) -> bool:
        if not content:
            return False
        for pattern in self.PARTIAL_PATTERNS:
            if pattern.search(content):
                return True
        return False

    def is_complete_tool_call(self, content: str) -> bool:
        if not content:
            return False
        for pattern in self.COMPLETE_PATTERNS:
            if pattern.search(content):
                return True
        return False

    def _clean_content(self, content: str) -> str:
        """Remove tool call markup from content, keep surrounding text."""
        if not content:
            return content

        # Remove XML-wrapped tool calls
        content = re.sub(r'<tool_call>\s*\{.*?\}\s*</tool_call>', '', content, flags=re.DOTALL)
        # Remove markdown-wrapped tool calls
        content = re.sub(r'```(?:json)?\s*\{[^}]*"name".*?\}\s*```', '', content, flags=re.DOTALL)
        # Remove bare JSON tool calls (careful not to remove other JSON)
        content = re.sub(
            r'\{\s*"name"\s*:\s*"[^"]+?"\s*,\s*"(?:arguments|parameters)"\s*:\s*\{.*?\}\s*\}',
            '', content, flags=re.DOTALL
        )
        # Clean up whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content.strip()


class DeepSeekStreamingHandler(StreamingToolCallHandler):
    """Streaming handler for DeepSeek tool calls"""

    def __init__(self):
        super().__init__(DeepSeekToolCallConverter())
