#!/usr/bin/env python3
"""Smart DeepSeek proxy: uses tool_choice=required for tool turns only.

Strategy:
- If tools are provided and it's a user message (not a tool result):
  force tool_choice=required so llama-server's constrained decoding
  produces structured tool_calls
- If the last message is a tool result: let model respond freely
  (tool_choice=auto or none) so it can explain or make next call
- No tools in request: pass through unchanged

This avoids the "fragmentation" problem of always-required while
ensuring the model actually calls tools when it should.

Listens on :18082, forwards to llama-server on :18080.
"""
import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import URLError

UPSTREAM = "http://127.0.0.1:18080"
LISTEN_PORT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 18082
VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv


def should_force_required(data):
    """Decide if this request should use tool_choice=required."""
    tools = data.get("tools")
    if not tools:
        return False

    messages = data.get("messages", [])
    if not messages:
        return True  # First message with tools, force it

    last_msg = messages[-1]
    last_role = last_msg.get("role", "")

    # If last message is from user: force required (user is asking for action)
    if last_role == "user":
        return True

    # If last message is a tool result: force required
    # (model should make next tool call or we need it to call a "done" signal)
    # Actually, after tool result the model might want to explain or call another tool.
    # Force required here too since DeepSeek can't do auto reliably
    if last_role == "tool":
        return True

    # Assistant messages (continuation): force required
    return True


class ProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._forward()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len else b""

        if self.path.endswith("/chat/completions") and body:
            try:
                data = json.loads(body)

                if data.get("tools") and should_force_required(data):
                    if VERBOSE:
                        last_role = data.get("messages", [{}])[-1].get("role", "?")
                        print(f"[proxy] FORCING required (last_role={last_role})")
                    data["tool_choice"] = "required"
                    body = json.dumps(data).encode()

            except (json.JSONDecodeError, KeyError, IndexError):
                pass

        self._forward(body)

    def _forward(self, body=None):
        if body is None:
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len) if content_len else b""
        url = UPSTREAM + self.path
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length")}
        if body:
            headers["Content-Length"] = str(len(body))
        req = Request(url, data=body if body else None, headers=headers,
                      method=self.command)
        try:
            with urlopen(req, timeout=300) as resp:
                self._send_response(resp.read(), resp.status)
        except URLError as e:
            self._send_response(str(e).encode(), 502)

    def _send_response(self, body, status):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        if VERBOSE:
            print(f"[proxy] {format % args}")


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", LISTEN_PORT), ProxyHandler)
    print(f"deepseek-smart-proxy: :{LISTEN_PORT} -> {UPSTREAM}")
    print(f"  Strategy: force tool_choice=required when tools present")
    print(f"  Verbose: {VERBOSE}")
    server.serve_forever()
