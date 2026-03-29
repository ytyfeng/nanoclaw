---
name: ml-research
description: Set up Karpathy's autoresearch framework for autonomous ML research on this machine. Clones the repo, prepares data, creates a research daemon (works with CUDA GPU, Apple Silicon MPS, or CPU), mounts it into the NanoClaw agent container, and installs the agent-facing skill. Use when the user wants to run ML research, train models, or run overnight experiments.
---

# Set Up ML Research (autoresearch)

Sets up [karpathy/autoresearch](https://github.com/karpathy/autoresearch) on the host and connects it to NanoClaw container agents via a lightweight HTTP daemon. Training runs on whatever hardware is available: NVIDIA GPU, Apple Silicon, or CPU.

## Phase 1: Pre-flight

### Check if already set up

```bash
test -f ~/autoresearch/research-daemon.py && echo "ALREADY_SETUP" || echo "FRESH_INSTALL"
```

If `ALREADY_SETUP`, skip to Phase 5 (Verify).

### Check prerequisites

```bash
# Git
git --version 2>/dev/null || echo "MISSING: git"

# uv package manager
uv --version 2>/dev/null || echo "MISSING: uv"

# Python 3.10+
python3 --version 2>/dev/null || echo "MISSING: python3"
```

If `uv` is missing, tell the user:

> `uv` is required. Install it with:
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> source ~/.bashrc  # or restart your shell
> ```

### Detect hardware

```bash
# NVIDIA GPU
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null && echo "DEVICE: cuda" || true

# Apple Silicon
uname -m 2>/dev/null | grep -q arm64 && echo "DEVICE: mps" || true

# CPU fallback info
python3 -c "import multiprocessing; print('CPU cores:', multiprocessing.cpu_count())" 2>/dev/null || true
```

Report detected hardware to the user before proceeding.

**CPU note:** Training will work but is slow (~1 experiment/hour vs ~12 on GPU). The agent will automatically use a small model configuration.

## Phase 2: Clone and Set Up autoresearch

### Clone the repository

```bash
cd ~
if [ -d ~/autoresearch/.git ]; then
  echo "Repo already exists, pulling latest..."
  git -C ~/autoresearch pull
else
  git clone https://github.com/karpathy/autoresearch ~/autoresearch
fi
```

### Install Python dependencies

```bash
cd ~/autoresearch
uv sync
```

### Prepare training data (one-time, ~2 minutes)

```bash
cd ~/autoresearch
uv run prepare.py
```

This downloads and tokenizes the training dataset. It only needs to run once.

### Verify baseline training works

```bash
cd ~/autoresearch
echo "Running a quick 60-second test (press Ctrl+C after you see the first val_bpb)..."
timeout 90 uv run train.py 2>&1 | head -30 || true
```

This confirms the training environment is working. If it fails, stop and report the error.

## Phase 3: Create the Research Daemon

Create `~/autoresearch/research-daemon.py`:

```bash
cat > ~/autoresearch/research-daemon.py << 'DAEMON_EOF'
#!/usr/bin/env python3
"""NanoClaw ML Research Daemon

HTTP API for triggering autoresearch training experiments from inside containers.
Training runs on the host using whatever GPU/MPS/CPU is available.

Port: 7474
Endpoints:
  GET  /status  - daemon status, device info, last result
  GET  /output  - raw stdout from last training run
  POST /run     - start a training run (non-blocking)
  POST /cancel  - cancel current run
"""
import json
import multiprocessing
import os
import platform
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

AUTORESEARCH_DIR = Path(os.environ.get("AUTORESEARCH_DIR", Path.home() / "autoresearch"))
PORT = int(os.environ.get("RESEARCH_DAEMON_PORT", "7474"))


def detect_device():
    """Detect best available compute device: cuda > mps > cpu."""
    # NVIDIA GPU
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return "cuda", r.stdout.strip().split("\n")[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Apple Silicon MPS
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3,
            ).stdout.strip()
        except Exception:
            chip = "Apple Silicon"
        return "mps", chip or "Apple Silicon"

    # CPU fallback
    cpu_count = multiprocessing.cpu_count()
    try:
        cpu_brand = subprocess.run(
            ["cat", "/proc/cpuinfo"],
            capture_output=True, text=True, timeout=3,
        ).stdout
        m = re.search(r"model name\s*:\s*(.+)", cpu_brand)
        cpu_name = m.group(1).strip() if m else f"CPU ({cpu_count} cores)"
    except Exception:
        cpu_name = f"CPU ({cpu_count} cores)"
    return "cpu", cpu_name


DEVICE, DEVICE_NAME = detect_device()


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.process = None
        self.run_count = 0
        self.last_result = None
        self.last_stdout = ""
        self.start_time = None


state = State()


def extract_val_bpb(text):
    """Extract val_bpb from training output (last occurrence wins)."""
    found = None
    for line in text.split("\n"):
        m = re.search(r"val[_\s]?bpb[:\s=]+([0-9]+\.[0-9]+)", line, re.IGNORECASE)
        if m:
            found = float(m.group(1))
    return found


def run_training_bg():
    """Run uv run train.py in background, stream output into state."""
    with state.lock:
        state.running = True
        state.start_time = time.time()
        state.last_stdout = ""

    env = os.environ.copy()
    # Ensure uv is findable (common install locations)
    for uv_path in ["~/.local/bin", "~/.cargo/bin"]:
        expanded = os.path.expanduser(uv_path)
        if os.path.isdir(expanded) and expanded not in env.get("PATH", ""):
            env["PATH"] = expanded + ":" + env.get("PATH", "")

    # Device-specific hints
    if DEVICE == "mps":
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    elif DEVICE == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    try:
        proc = subprocess.Popen(
            ["uv", "run", "train.py"],
            cwd=AUTORESEARCH_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        with state.lock:
            state.process = proc

        lines = []
        for line in proc.stdout:
            lines.append(line)
            with state.lock:
                # Keep rolling last 500 lines for live output
                state.last_stdout = "".join(lines[-500:])

        proc.wait()
        full_output = "".join(lines)
        elapsed = time.time() - state.start_time
        val_bpb = extract_val_bpb(full_output)

        with state.lock:
            state.running = False
            state.process = None
            state.run_count += 1
            state.last_result = {
                "run_count": state.run_count,
                "val_bpb": val_bpb,
                "returncode": proc.returncode,
                "elapsed_seconds": round(elapsed, 1),
                "device": DEVICE,
                "device_name": DEVICE_NAME,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "success": proc.returncode == 0,
            }
            state.last_stdout = full_output[-20000:]  # last ~20KB

    except Exception as e:
        with state.lock:
            elapsed = time.time() - (state.start_time or time.time())
            state.running = False
            state.process = None
            state.run_count += 1
            state.last_result = {
                "run_count": state.run_count,
                "val_bpb": None,
                "returncode": -1,
                "elapsed_seconds": round(elapsed, 1),
                "device": DEVICE,
                "device_name": DEVICE_NAME,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "success": False,
                "error": str(e),
            }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # quiet

    def send_json(self, data, code=200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/status":
            with state.lock:
                self.send_json({
                    "daemon": "running",
                    "device": DEVICE,
                    "device_name": DEVICE_NAME,
                    "autoresearch_dir": str(AUTORESEARCH_DIR),
                    "running": state.running,
                    "run_count": state.run_count,
                    "last_result": state.last_result,
                    "live_output_lines": state.last_stdout.count("\n") if state.running else 0,
                })
        elif self.path == "/output":
            with state.lock:
                self.send_json({"output": state.last_stdout})
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == "/run":
            with state.lock:
                if state.running:
                    self.send_json(
                        {"error": "Experiment already running. Check GET /status."},
                        409,
                    )
                    return
            t = threading.Thread(target=run_training_bg, daemon=True)
            t.start()
            self.send_json({
                "status": "started",
                "device": DEVICE,
                "device_name": DEVICE_NAME,
                "message": f"Training started on {DEVICE_NAME}. Poll GET /status for results.",
            })
        elif self.path == "/cancel":
            with state.lock:
                if state.process:
                    state.process.terminate()
                    self.send_json({"status": "cancelled"})
                else:
                    self.send_json({"error": "No experiment running"}, 404)
        else:
            self.send_json({"error": "Not found"}, 404)


if __name__ == "__main__":
    print(f"[research-daemon] port={PORT} device={DEVICE} ({DEVICE_NAME})", flush=True)
    print(f"[research-daemon] autoresearch_dir={AUTORESEARCH_DIR}", flush=True)
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[research-daemon] Shutting down", flush=True)
DAEMON_EOF
chmod +x ~/autoresearch/research-daemon.py
echo "Daemon script created."
```

### Test the daemon manually

```bash
cd ~/autoresearch
python3 research-daemon.py &
DAEMON_PID=$!
sleep 1
curl -s http://127.0.0.1:7474/status | python3 -m json.tool
kill $DAEMON_PID
```

Confirm the status response shows the correct device.

## Phase 4: Register Daemon as a System Service

### Linux (systemd)

```bash
# Resolve uv path for the service
UV_PATH=$(which uv 2>/dev/null || echo "$HOME/.local/bin/uv")
UV_DIR=$(dirname "$UV_PATH")

mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/nanoclaw-research.service << SERVICE_EOF
[Unit]
Description=NanoClaw ML Research Daemon
After=network.target

[Service]
Type=simple
WorkingDirectory=%h/autoresearch
Environment="PATH=${UV_DIR}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/usr/bin/env python3 %h/autoresearch/research-daemon.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
SERVICE_EOF

systemctl --user daemon-reload
systemctl --user enable nanoclaw-research
systemctl --user start nanoclaw-research
sleep 2
systemctl --user status nanoclaw-research
```

### macOS (launchd)

```bash
UV_PATH=$(which uv 2>/dev/null || echo "$HOME/.local/bin/uv")
UV_DIR=$(dirname "$UV_PATH")

cat > ~/Library/LaunchAgents/com.nanoclaw.research.plist << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.nanoclaw.research</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/env</string>
    <string>python3</string>
    <string>${HOME}/autoresearch/research-daemon.py</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${HOME}/autoresearch</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>${UV_DIR}:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${HOME}/autoresearch/daemon.log</string>
  <key>StandardErrorPath</key>
  <string>${HOME}/autoresearch/daemon.log</string>
</dict>
</plist>
PLIST_EOF

launchctl load ~/Library/LaunchAgents/com.nanoclaw.research.plist
sleep 2
curl -s http://127.0.0.1:7474/status | python3 -m json.tool
```

## Phase 5: Configure Mount Allowlist

The autoresearch directory must be added to the mount allowlist before containers can access it.

```bash
python3 - << 'PYEOF'
import json, os, sys
from pathlib import Path

allowlist_path = Path.home() / ".config" / "nanoclaw" / "mount-allowlist.json"
autoresearch_dir = str(Path.home() / "autoresearch")

allowlist_path.parent.mkdir(parents=True, exist_ok=True)

# Load existing or create fresh
if allowlist_path.exists():
    allowlist = json.loads(allowlist_path.read_text())
else:
    allowlist = {
        "allowedRoots": [],
        "blockedPatterns": ["password", "secret", "token"],
        "nonMainReadOnly": True,
    }

# Check if already present
for root in allowlist.get("allowedRoots", []):
    if root.get("path") in [autoresearch_dir, "~/autoresearch"]:
        print(f"Already in allowlist: {autoresearch_dir}")
        sys.exit(0)

# Add it
allowlist.setdefault("allowedRoots", []).append({
    "path": autoresearch_dir,
    "allowReadWrite": True,
    "description": "autoresearch ML experiments",
})

allowlist_path.write_text(json.dumps(allowlist, indent=2) + "\n")
print(f"Added to allowlist: {autoresearch_dir}")
PYEOF
```

## Phase 6: Configure the Main Group Mount

Add the autoresearch mount to the main group's container configuration in the NanoClaw database.

```bash
python3 - << 'PYEOF'
import json, os, sqlite3, sys
from pathlib import Path

db_path = os.path.join(os.getcwd(), "store", "messages.db")
autoresearch_dir = str(Path.home() / "autoresearch")

if not os.path.exists(db_path):
    print("ERROR: Database not found. Is NanoClaw set up?", file=sys.stderr)
    sys.exit(1)

conn = sqlite3.connect(db_path)
row = conn.execute(
    "SELECT jid, container_config FROM registered_groups WHERE is_main = 1 OR folder = 'main' LIMIT 1"
).fetchone()

if not row:
    print("ERROR: No main group found. Register your main chat with /setup first.", file=sys.stderr)
    conn.close()
    sys.exit(1)

jid, config_json = row
config = json.loads(config_json) if config_json else {}
mounts = config.get("additionalMounts", [])

# Check if already configured
for m in mounts:
    if m.get("hostPath") in [autoresearch_dir, "~/autoresearch"]:
        print(f"Mount already configured: {autoresearch_dir} -> /workspace/extra/autoresearch")
        conn.close()
        sys.exit(0)

mounts.append({
    "hostPath": autoresearch_dir,
    "containerPath": "autoresearch",
    "readonly": False,
})
config["additionalMounts"] = mounts

conn.execute(
    "UPDATE registered_groups SET container_config = ? WHERE jid = ?",
    (json.dumps(config), jid),
)
conn.commit()
conn.close()
print(f"Mount configured: {autoresearch_dir} -> /workspace/extra/autoresearch (read-write)")
PYEOF
```

## Phase 7: For CPU-Only Machines — Adjust program.md

If no GPU was detected in Phase 1, update the research goal to account for CPU constraints:

```bash
# Only do this if DEVICE is cpu
if ! nvidia-smi &>/dev/null && ! (uname -m | grep -q arm64); then
  cat >> ~/autoresearch/program.md << 'CPU_NOTE'

## Hardware Note
This machine has no GPU. Training runs on CPU — significantly slower.
Keep models very small to fit within the 5-minute budget:
- n_layer: 2-3
- n_head: 4
- n_embd: 128-256
- batch_size: 32-64

Focus experiments on algorithmic changes (optimizer, schedule, architecture ratios)
rather than scaling. val_bpb values will be higher than GPU runs — that's expected.
CPU_NOTE
  echo "Added CPU guidance to program.md"
fi
```

## Phase 8: Restart NanoClaw

```bash
# Linux
systemctl --user restart nanoclaw 2>/dev/null || true

# macOS
launchctl kickstart -k gui/$(id -u)/com.nanoclaw 2>/dev/null || true
```

## Phase 9: Verify

### Daemon is reachable

```bash
curl -s http://127.0.0.1:7474/status | python3 -m json.tool
```

### Mount is active (after NanoClaw restarts)

Tell the user:

> Send a message to your main chat:
> ```
> check /workspace/extra/autoresearch/program.md
> ```
>
> The agent should be able to read the file. If it returns "not found", the mount hasn't taken effect yet — restart NanoClaw and try again.

### Run a test experiment

Tell the user:

> Send to your main chat:
> ```
> kick off an autoresearch experiment — read program.md and run one training iteration
> ```

The agent will read `program.md`, check the daemon, start training, and report val_bpb.

## Troubleshooting

**"No main group found"** — Run `/setup` first to register your main chat.

**Daemon not starting (Linux):**
```bash
systemctl --user status nanoclaw-research
journalctl --user -u nanoclaw-research -n 50
```

**Daemon not starting (macOS):**
```bash
cat ~/autoresearch/daemon.log
launchctl unload ~/Library/LaunchAgents/com.nanoclaw.research.plist
launchctl load ~/Library/LaunchAgents/com.nanoclaw.research.plist
```

**uv not found by daemon:**
Check that uv's directory is in the service PATH. Run `which uv` to find it, then update the service file with the correct path.

**Mount rejected (allowlist error):**
Check `~/.config/nanoclaw/mount-allowlist.json` — confirm `~/autoresearch` (or its expanded absolute path) is in `allowedRoots` with `"allowReadWrite": true`.

**prepare.py fails:**
This usually means missing internet access or a corrupted download. Try:
```bash
cd ~/autoresearch && rm -rf data/ && uv run prepare.py
```
