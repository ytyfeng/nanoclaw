#!/usr/bin/env python3
"""
gcloud daemon for NanoClaw's container gcloud skill (container/skills/gcloud/SKILL.md).

Runs on the HOST (not in a container) and exposes gcloud over HTTP so agent
containers can call it via host.docker.internal:7475. Containers never get
direct gcloud credentials — only this daemon does.

Endpoints:
  GET  /status  -> {account, project, ok}
  POST /run      body: {"args": [...], "timeout": <seconds, optional>}
                 -> {"returncode", "stdout", "stderr", "success", "command"}
"""
import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GCLOUD_BIN = os.environ.get("GCLOUD_BIN", "gcloud")
# 0.0.0.0 (not 127.0.0.1) so Docker containers can reach this via
# host.docker.internal — see container/skills/gcloud/SKILL.md. This does
# expose the daemon on all interfaces; there is no auth on /run, so treat
# it as trusted-network-only (fine for a single-host dev setup, not for a
# shared or internet-facing box).
HOST = os.environ.get("GCLOUD_DAEMON_HOST", "0.0.0.0")
PORT = int(os.environ.get("GCLOUD_DAEMON_PORT", "7475"))
DEFAULT_TIMEOUT = 60
MAX_TIMEOUT = 600


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/status":
            self._send_json(404, {"error": "not found"})
            return
        try:
            account = subprocess.run(
                [GCLOUD_BIN, "config", "get-value", "account"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            project = subprocess.run(
                [GCLOUD_BIN, "config", "get-value", "project"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            self._send_json(200, {"ok": True, "account": account, "project": project})
        except Exception as e:
            self._send_json(500, {"ok": False, "error": str(e)})

    def do_POST(self) -> None:
        if self.path != "/run":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        args = body.get("args")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            self._send_json(400, {"error": "'args' must be a list of strings"})
            return

        timeout = body.get("timeout", DEFAULT_TIMEOUT)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            timeout = DEFAULT_TIMEOUT
        timeout = min(timeout, MAX_TIMEOUT)

        cmd = [GCLOUD_BIN, *args]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
            self._send_json(200, {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
                "command": " ".join(cmd),
            })
        except subprocess.TimeoutExpired:
            self._send_json(200, {
                "returncode": -1,
                "stdout": "",
                "stderr": f"command timed out after {timeout}s",
                "success": False,
                "command": " ".join(cmd),
            })
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        print(f"[gcloud-daemon] {self.address_string()} - {fmt % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"gcloud daemon listening on {HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
