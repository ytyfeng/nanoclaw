---
name: research-paper
description: Autonomously write a full academic research paper on any topic. Runs the 23-stage AutoResearchClaw pipeline — literature review, hypothesis generation, experiment design, paper writing, and citation verification — using local Ollama models. Use whenever the user asks to research a topic, write a paper, or do a literature review.
allowed-tools: Bash
---

# Autonomous Research Paper Pipeline

Powered by [AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw): 23-stage pipeline that turns a topic into a complete academic paper (5,000–6,500 words) with literature review, experiments, and verified citations. Uses local Ollama models — no external API keys required.

**Estimated time:** 30–60 minutes depending on topic complexity.

## Quick start

```bash
# 1. Install (one-time, ~10 seconds)
pip install researchclaw --quiet 2>&1 | tail -1

# 2. Write Ollama config (one-time)
research-paper-setup

# 3. Run pipeline
cd /workspace/group
export OLLAMA_API_KEY=ollama
researchclaw run --topic "Your research topic" \
  --config /workspace/group/config.researchclaw.yaml \
  --auto-approve
```

## Setup (first time only)

Run this once to write the Ollama-backed config to your workspace:

```bash
mkdir -p /workspace/group
cat > /workspace/group/config.researchclaw.yaml << 'RCEOF'
project:
  name: "research"
  mode: "full-auto"

research:
  domains: [machine-learning, computer-science]
  daily_paper_count: 10
  quality_threshold: 3.5

runtime:
  timezone: "America/Los_Angeles"
  max_parallel_tasks: 2
  retry_attempts: 2

llm:
  provider: "openai-compatible"
  base_url: "http://host.docker.internal:11434/v1"
  api_key_env: "OLLAMA_API_KEY"
  primary_model: "glm-4.7-flash:latest"
  fallback_models:
    - "qwen3.5:latest"

experiment:
  mode: "simulated"
  time_budget_minutes: 30
  hardware_requirements: cpu

security:
  hitl_stages: []
  auto_approve: true

notifications:
  console: true
RCEOF
echo "Config written to /workspace/group/config.researchclaw.yaml"
```

## Running the pipeline

```bash
cd /workspace/group
export OLLAMA_API_KEY=ollama
researchclaw run \
  --topic "TOPIC" \
  --config /workspace/group/config.researchclaw.yaml \
  --auto-approve 2>&1
```

Replace `TOPIC` with the user's research topic. Quote it if it contains spaces.

## Reading results

After the pipeline finishes, artifacts are in `/workspace/group/artifacts/`:

```bash
# Full paper (markdown)
ls /workspace/group/artifacts/stage_17_*/paper_draft.md 2>/dev/null || \
  find /workspace/group/artifacts -name "paper_draft.md" | head -1 | xargs cat

# Conference-ready LaTeX
find /workspace/group/artifacts -name "paper.tex" | head -1

# All generated artifacts
ls /workspace/group/artifacts/
```

## Full workflow

```bash
# Step 1: Install if needed
python3 -c "import researchclaw" 2>/dev/null || pip install researchclaw --quiet

# Step 2: Write config if not present
[ -f /workspace/group/config.researchclaw.yaml ] || cat > /workspace/group/config.researchclaw.yaml << 'RCEOF'
project:
  name: "research"
  mode: "full-auto"
research:
  domains: [machine-learning, computer-science]
  daily_paper_count: 10
  quality_threshold: 3.5
runtime:
  timezone: "America/Los_Angeles"
  max_parallel_tasks: 2
  retry_attempts: 2
llm:
  provider: "openai-compatible"
  base_url: "http://host.docker.internal:11434/v1"
  api_key_env: "OLLAMA_API_KEY"
  primary_model: "glm-4.7-flash:latest"
  fallback_models:
    - "qwen3.5:latest"
experiment:
  mode: "simulated"
  time_budget_minutes: 30
  hardware_requirements: cpu
security:
  hitl_stages: []
  auto_approve: true
notifications:
  console: true
RCEOF

# Step 3: Run pipeline
cd /workspace/group
export OLLAMA_API_KEY=ollama
researchclaw run --topic "$TOPIC" --config config.researchclaw.yaml --auto-approve

# Step 4: Find and read the paper
PAPER=$(find /workspace/group/artifacts -name "paper_draft.md" | sort | tail -1)
if [ -n "$PAPER" ]; then
  echo "=== PAPER DRAFT ==="
  cat "$PAPER"
else
  echo "Pipeline may still be running or failed. Check artifacts/:"
  ls /workspace/group/artifacts/ 2>/dev/null || echo "No artifacts yet."
fi
```

## Sending progress updates

Since this runs for 30–60 minutes, send the user an acknowledgment first:

```
mcp__nanoclaw__send_message with content: "Starting research pipeline on '[topic]'. This will take 30–60 minutes. I'll send you the paper when it's done."
```

Then run the pipeline, and when complete, send the paper via `mcp__nanoclaw__send_message`.

## Troubleshooting

**Ollama connection refused**: Verify Ollama is running on the host (`ollama list`). The container reaches it at `host.docker.internal:11434`.

**Model not found**: The pipeline uses `glm-4.7-flash:latest` — verify it's installed with `ollama list` on the host.

**Pipeline hangs at literature search**: arXiv/Semantic Scholar may be rate-limiting. Wait a few minutes and retry, or reduce `daily_paper_count` to 5.

**pip install fails**: Try `pip install researchclaw --break-system-packages --quiet` if running in a system Python environment.
