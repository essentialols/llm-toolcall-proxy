#!/usr/bin/env bash
# Launch the llm-toolcall-proxy configured for DeepSeek on llama-server
# Proxy listens on :18082, forwards to llama-server tunnel on :18080

export BACKEND_HOST=127.0.0.1
export BACKEND_PORT=18080
export BACKEND_PROTOCOL=http
export PROXY_HOST=127.0.0.1
export PROXY_PORT=18082
export DEBUG=true
export LOG_LEVEL=DEBUG
export ENABLE_TOOL_CALL_CONVERSION=true
export REMOVE_THINK_TAGS=true
export REQUEST_TIMEOUT=300

cd "$(dirname "$0")"
echo "llm-toolcall-proxy for DeepSeek Coder V2 Lite"
echo "  Upstream: $BACKEND_PROTOCOL://$BACKEND_HOST:$BACKEND_PORT"
echo "  Proxy:    http://$PROXY_HOST:$PROXY_PORT"
echo "  Point Goose at: OPENAI_HOST=http://127.0.0.1:18082"
echo ""

exec ~/tools/.venv/bin/python app.py
