#!/usr/bin/env bash
# Swap llama-server to DeepSeek Coder V2 Lite + start tool-parse proxy
set -euo pipefail

echo "=== Swapping to DeepSeek Coder V2 Lite ==="

# Update service file
ssh h1 'sudo tee /etc/systemd/system/llama-qwen.service > /dev/null << '\''UNIT'\''
[Unit]
Description=llama.cpp DeepSeek Coder V2 Lite (Vulkan iGPU, 16k ctx)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ingmar
SupplementaryGroups=render
Environment=RADV_PERFTEST=nogttspill
Slice=background.slice
CPUQuota=500%
MemoryMax=14G
MemoryHigh=10G
LimitMEMLOCK=infinity
ExecStart=/home/ingmarsturm/llama.cpp/build-vulkan/bin/.llama-server.real -m /home/ingmarsturm/models/deepseek-coder-v2-lite-q4_k_m.gguf --host 0.0.0.0 --port 38016 --parallel 1 --ctx-size 16384 --n-gpu-layers 99 -t 6 --threads-batch 8 --batch-size 512 --ubatch-size 256 --mmap --mlock --jinja --temp 0.2 --repeat-penalty 1.1
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT'

echo "Service file updated"

# Restart
ssh h1 'sudo systemctl daemon-reload && sudo systemctl restart llama-qwen'
echo "Service restarting..."

# Wait for model to load
echo "Waiting for DeepSeek to load..."
until curl -fsS --max-time 3 http://127.0.0.1:18080/v1/models > /dev/null 2>&1; do
    sleep 2
    printf "."
done
echo ""

# Verify model
MODEL=$(curl -fsS http://127.0.0.1:18080/v1/models 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "Model loaded: $MODEL"

# Quick smoke test
echo "Smoke test..."
RESULT=$(curl -fsS --max-time 30 http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Say hello"}],"max_tokens":20}' 2>/dev/null)
SPEED=$(echo "$RESULT" | python3 -c "import sys,json; t=json.load(sys.stdin).get('timings',{}); print(f'{t.get(\"predicted_per_second\",0):.1f} tok/s')")
echo "Generation speed: $SPEED"

echo ""
echo "=== DeepSeek ready on :18080 ==="
echo "Start proxy:  cd ~/tools/llm-toolcall-proxy && ~/tools/.venv/bin/python app.py"
echo "Or custom:    ~/tools/.venv/bin/python ~/tools/qwen-relay/tool-parse-proxy.py -v"
echo "Test:         ~/tools/llm-toolcall-proxy/test-deepseek.sh http://127.0.0.1:18082"
