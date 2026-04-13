---
name: gcloud
description: Run Google Cloud (gcloud) commands from inside the container. Use when the user asks to manage GCP resources — create/list/delete VMs, check GPU availability, manage Cloud Storage, query billing, etc. Default project is courseassistai.
allowed-tools: Bash
---

# GCloud via Host Daemon

The gcloud daemon runs on the host at `host.docker.internal:7475`. It runs gcloud commands on your behalf using the host's authenticated credentials (tyfeng7@gmail.com, default project: courseassistai).

## Quick start

```bash
# Check daemon and current account/project
curl -s http://host.docker.internal:7475/status | python3 -m json.tool

# Run any gcloud command
curl -s -X POST http://host.docker.internal:7475/run \
  -H "Content-Type: application/json" \
  -d '{"args": ["compute", "instances", "list"]}' | python3 -m json.tool
```

## API

**POST /run** — run a gcloud command

Request body:
```json
{
  "args": ["compute", "instances", "list", "--project", "courseassistai"],
  "timeout": 60
}
```

Response:
```json
{
  "returncode": 0,
  "stdout": "...",
  "stderr": "...",
  "success": true,
  "command": "gcloud compute instances list ..."
}
```

`args` is the argument list after `gcloud` — do not include `gcloud` itself.

**GET /status** — check daemon health, active account, default project.

## Common operations

### List VMs
```bash
curl -s -X POST http://host.docker.internal:7475/run \
  -d '{"args":["compute","instances","list","--format","table(name,zone,machineType,status)"]}'
```

### Create a GPU VM
```bash
curl -s -X POST http://host.docker.internal:7475/run \
  -H "Content-Type: application/json" \
  -d '{
    "args": [
      "compute", "instances", "create", "gpu-vm-1",
      "--zone", "us-central1-a",
      "--machine-type", "n1-standard-8",
      "--accelerator", "type=nvidia-tesla-t4,count=1",
      "--image-family", "pytorch-latest-gpu",
      "--image-project", "deeplearning-platform-release",
      "--boot-disk-size", "100GB",
      "--maintenance-policy", "TERMINATE",
      "--metadata", "install-nvidia-driver=True"
    ],
    "timeout": 120
  }'
```

### List available GPU types in a zone
```bash
curl -s -X POST http://host.docker.internal:7475/run \
  -d '{"args":["compute","accelerator-types","list","--filter","zone:us-central1-a"]}'
```

### Stop / start / delete a VM
```bash
# Stop
curl -s -X POST http://host.docker.internal:7475/run \
  -d '{"args":["compute","instances","stop","gpu-vm-1","--zone","us-central1-a"]}'

# Start
curl -s -X POST http://host.docker.internal:7475/run \
  -d '{"args":["compute","instances","start","gpu-vm-1","--zone","us-central1-a"]}'

# Delete
curl -s -X POST http://host.docker.internal:7475/run \
  -d '{"args":["compute","instances","delete","gpu-vm-1","--zone","us-central1-a","--quiet"]}'
```

### SSH into a VM
```bash
curl -s -X POST http://host.docker.internal:7475/run \
  -d '{"args":["compute","ssh","gpu-vm-1","--zone","us-central1-a","--command","nvidia-smi"],"timeout":30}'
```

### List Cloud Storage buckets
```bash
curl -s -X POST http://host.docker.internal:7475/run \
  -d '{"args":["storage","buckets","list","--format","value(name)"]}'
```

### Switch project
```bash
curl -s -X POST http://host.docker.internal:7475/run \
  -d '{"args":["config","set","project","my-other-project"]}'
```

## Tips

- Default project is **courseassistai** — no need to specify `--project` for most commands
- VM creation takes 30-90s — use `"timeout": 120`
- For long operations (e.g. cluster creation), use `"timeout": 300`
- Always confirm with the user before deleting resources
- Use `--format json` to get machine-readable output for parsing
- GPU quota may be limited — check with `gcloud compute regions describe us-central1 --format="json(quotas)"`

## Check quota
```bash
curl -s -X POST http://host.docker.internal:7475/run \
  -d '{"args":["compute","regions","describe","us-central1","--format","json(quotas)"],"timeout":15}' \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
quotas = json.loads(d['stdout']).get('quotas',[])
gpu = [q for q in quotas if 'GPU' in q.get('metric','')]
for q in gpu: print(q['metric'], q['usage'],'/',q['limit'])
"
```
