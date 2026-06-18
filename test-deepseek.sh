#!/usr/bin/env bash
# Test the DeepSeek converter proxy end-to-end
# Prerequisites: llama-server running DeepSeek on :18080, proxy on :18082

set -euo pipefail

PROXY="${1:-http://127.0.0.1:18082}"
echo "Testing DeepSeek tool-call proxy at $PROXY"

# Test 1: Tool call should be extracted from text
echo -e "\n=== Test 1: Tool calling (auto) ==="
result=$(curl -fsS --max-time 60 "$PROXY/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder-v2-lite-q4_k_m.gguf",
    "messages": [{"role": "user", "content": "Read the file /home/user/main.py"}],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto",
    "max_tokens": 200
  }')

finish=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['finish_reason'])")
has_tc=$(echo "$result" | python3 -c "import sys,json; print(bool(json.load(sys.stdin)['choices'][0]['message'].get('tool_calls')))")
echo "finish_reason: $finish"
echo "has_tool_calls: $has_tc"

if [ "$has_tc" = "True" ]; then
    echo "$result" | python3 -c "
import sys, json
d = json.load(sys.stdin)
tc = d['choices'][0]['message']['tool_calls']
for c in tc:
    print(f'  -> {c[\"function\"][\"name\"]}({c[\"function\"][\"arguments\"]})')
"
    echo "PASS"
else
    echo "FAIL - no tool calls extracted"
    echo "$result" | python3 -m json.tool 2>/dev/null | head -20
fi

# Test 2: Plain text should pass through
echo -e "\n=== Test 2: Plain text (no tools) ==="
result2=$(curl -fsS --max-time 30 "$PROXY/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder-v2-lite-q4_k_m.gguf",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50
  }')
content=$(echo "$result2" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message'].get('content','')[:100])")
echo "Content: $content"
[ -n "$content" ] && echo "PASS" || echo "FAIL"

# Test 3: Multi-tool scenario
echo -e "\n=== Test 3: Multi-tool choice ==="
result3=$(curl -fsS --max-time 60 "$PROXY/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder-v2-lite-q4_k_m.gguf",
    "messages": [{"role": "user", "content": "Create a file called hello.py with print(42) in it"}],
    "tools": [
      {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
      {"type": "function", "function": {"name": "write_file", "description": "Write to a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}
    ],
    "tool_choice": "auto",
    "max_tokens": 300
  }')
finish3=$(echo "$result3" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['finish_reason'])")
has_tc3=$(echo "$result3" | python3 -c "import sys,json; print(bool(json.load(sys.stdin)['choices'][0]['message'].get('tool_calls')))")
echo "finish_reason: $finish3"
echo "has_tool_calls: $has_tc3"
if [ "$has_tc3" = "True" ]; then
    echo "$result3" | python3 -c "
import sys, json
d = json.load(sys.stdin)
tc = d['choices'][0]['message']['tool_calls']
for c in tc:
    print(f'  -> {c[\"function\"][\"name\"]}({c[\"function\"][\"arguments\"]})')
"
    echo "PASS"
else
    echo "FAIL"
    echo "$result3" | python3 -m json.tool 2>/dev/null | head -20
fi

echo -e "\n=== Done ==="
