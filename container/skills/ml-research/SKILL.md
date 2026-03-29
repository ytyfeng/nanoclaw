---
name: ml-research
description: Run autonomous ML research experiments using Karpathy's autoresearch framework. The agent modifies train.py, runs timed experiments, evaluates val_bpb, and iterates. Use when the user asks to run ML experiments, improve a model, try a new architecture, or do overnight research.
allowed-tools: Bash
---

# ML Research with autoresearch

Autonomous ML research loop: read instructions → modify training code → run experiment → evaluate → iterate.

The research daemon runs on the host at `host.docker.internal:7474` and handles all GPU/MPS/CPU training.
Your workspace is at `/workspace/extra/autoresearch` (read-write).

## Quick start

```bash
# 1. Check daemon is up
curl -s http://host.docker.internal:7474/status | python3 -m json.tool

# 2. Read the research goals
cat /workspace/extra/autoresearch/program.md

# 3. Read current training code
cat /workspace/extra/autoresearch/train.py

# 4. Modify train.py with your improvement, then kick off a run
curl -s -X POST http://host.docker.internal:7474/run | python3 -m json.tool

# 5. Poll until done (training takes ~5 minutes)
watch -n 30 'curl -s http://host.docker.internal:7474/status | python3 -m json.tool'

# 6. Read results
curl -s http://host.docker.internal:7474/status | python3 -c "import json,sys; d=json.load(sys.stdin); print('val_bpb:', d['last_result']['val_bpb'])"
```

## Daemon API

| Method | Endpoint   | Description                                      |
|--------|------------|--------------------------------------------------|
| GET    | `/status`  | Running state, device info, last result          |
| GET    | `/output`  | Raw stdout/stderr from last training run         |
| POST   | `/run`     | Start a training experiment (non-blocking)       |
| POST   | `/cancel`  | Cancel current run                               |

## Interpreting status

```bash
curl -s http://host.docker.internal:7474/status
```

Key fields:
- `running` — `true` while training is in progress
- `device` — `"cuda"`, `"mps"`, or `"cpu"`
- `device_name` — human-readable hardware name
- `last_result.val_bpb` — validation bits-per-byte (lower is better)
- `last_result.elapsed_seconds` — how long the run took

## Research workflow

### 1. Read the goal

```bash
cat /workspace/extra/autoresearch/program.md
```

This describes what to optimize and any constraints.

### 2. Review current code

```bash
cat /workspace/extra/autoresearch/train.py
```

Study the model architecture, optimizer, and training loop.

### 3. Check device and calibrate expectations

```bash
curl -s http://host.docker.internal:7474/status | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print(d['device_name'], '-', d['device'])"
```

- **CUDA (GPU)**: full experiments, H100/A100 typical; ~12 experiments/hour
- **MPS (Apple Silicon)**: works well for small models; ~3-6 experiments/hour
- **CPU**: slow — reduce model size significantly (fewer layers, smaller d_model); ~1 experiment/hour

For CPU runs, always reduce model complexity first:
```python
# Suggested CPU-friendly settings in train.py
n_layer = 2
n_head = 4
n_embd = 128
```

### 4. Propose an experiment

Based on `program.md` and current val_bpb, pick ONE change to test. Good candidates:
- Learning rate schedule (cosine vs linear, warmup steps)
- Optimizer (AdamW betas, weight decay)
- Architecture (depth vs width, attention vs MLP ratio)
- Batch size / gradient accumulation
- Regularization (dropout, weight tying)

### 5. Edit train.py

```bash
# Read current train.py
cat /workspace/extra/autoresearch/train.py

# Edit with your change
# Use Write or Edit tool directly on the file
```

Keep a comment in train.py noting what you changed and why.

### 6. Run the experiment

```bash
curl -s -X POST http://host.docker.internal:7474/run
```

Wait for the job to complete:

```bash
# Poll every 30 seconds
while true; do
  STATUS=$(curl -s http://host.docker.internal:7474/status)
  RUNNING=$(echo "$STATUS" | python3 -c "import json,sys; print(json.load(sys.stdin)['running'])")
  if [ "$RUNNING" = "False" ]; then
    echo "$STATUS" | python3 -m json.tool
    break
  fi
  echo "Training in progress... ($(echo "$STATUS" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('live_output_lines',0), 'lines so far')"))"
  sleep 30
done
```

### 7. Evaluate results

```bash
curl -s http://host.docker.internal:7474/status | python3 -c "
import json, sys
d = json.load(sys.stdin)
r = d.get('last_result', {})
print(f\"Run #{r.get('run_count')}: val_bpb={r.get('val_bpb')} ({r.get('elapsed_seconds')}s on {r.get('device_name')})\")
print(f\"Success: {r.get('success')}\")
"
```

If the run failed, check the output:
```bash
curl -s http://host.docker.internal:7474/output | python3 -c "import json,sys; print(json.load(sys.stdin)['output'][-3000:])"
```

### 8. Log and iterate

Keep a research log in `/workspace/group/research-log.md`:

```markdown
## Run N - [description of change]
- val_bpb: X.XXXX
- Change: [what you tried]
- Result: [better/worse/same]
- Next: [what to try next]
```

Then propose the next experiment based on what you learned.

## Overnight research loop

For extended autonomous research, repeat the workflow continuously:
1. Check last val_bpb
2. Hypothesize one improvement
3. Edit train.py
4. Run experiment
5. Log result
6. Repeat

Send periodic updates via `mcp__nanoclaw__send_message` with best val_bpb achieved.

## Troubleshooting

**Daemon not reachable:**
```bash
curl -s http://host.docker.internal:7474/status || echo "Daemon offline"
```
Ask the user to check: `systemctl --user status nanoclaw-research` (Linux) or `launchctl list | grep research` (macOS).

**Training crashes immediately:**
Check output for CUDA/MPS errors:
```bash
curl -s http://host.docker.internal:7474/output | python3 -c "import json,sys; print(json.load(sys.stdin)['output'][:2000])"
```
Common fix: reduce batch size or model size in train.py.

**val_bpb is None:**
The metric wasn't printed in the expected format. Check raw output and look for the actual metric name used.

**Previous run still running:**
```bash
curl -s -X POST http://host.docker.internal:7474/cancel
```
