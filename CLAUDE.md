# NanoClaw

Personal Claude assistant. See [README.md](README.md) for philosophy and setup. See [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) for architecture decisions.

## Quick Context

Single Node.js process with skill-based channel system. Channels (WhatsApp, Telegram, Slack, Discord, Gmail) are skills that self-register at startup. Messages route to Claude Agent SDK running in containers (Linux VMs). Each group has isolated filesystem and memory.

## Key Files

| File | Purpose |
|------|---------|
| `src/index.ts` | Orchestrator: state, message loop, agent invocation |
| `src/channels/registry.ts` | Channel registry (self-registration at startup) |
| `src/ipc.ts` | IPC watcher and task processing |
| `src/router.ts` | Message formatting and outbound routing |
| `src/config.ts` | Trigger pattern, paths, intervals |
| `src/container-runner.ts` | Spawns agent containers with mounts |
| `src/task-scheduler.ts` | Runs scheduled tasks |
| `src/db.ts` | SQLite operations |
| `groups/{name}/CLAUDE.md` | Per-group memory (isolated) |
| `container/skills/` | Skills loaded inside agent containers (browser, status, formatting) |

## Secrets / Credentials / Proxy (OneCLI)

API keys, secret keys, OAuth tokens, and auth credentials are managed by the OneCLI gateway — which handles secret injection into containers at request time, so no keys or tokens are ever passed to containers directly. Run `onecli --help`.

## Skills

Four types of skills exist in NanoClaw. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full taxonomy and guidelines.

- **Feature skills** — merge a `skill/*` branch to add capabilities (e.g. `/add-telegram`, `/add-slack`)
- **Utility skills** — ship code files alongside SKILL.md (e.g. `/claw`)
- **Operational skills** — instruction-only workflows, always on `main` (e.g. `/setup`, `/debug`)
- **Container skills** — loaded inside agent containers at runtime (`container/skills/`)

| Skill | When to Use |
|-------|-------------|
| `/setup` | First-time installation, authentication, service configuration |
| `/customize` | Adding channels, integrations, changing behavior |
| `/debug` | Container issues, logs, troubleshooting |
| `/update-nanoclaw` | Bring upstream NanoClaw updates into a customized install |
| `/init-onecli` | Install OneCLI Agent Vault and migrate `.env` credentials to it |
| `/qodo-pr-resolver` | Fetch and fix Qodo PR review issues interactively or in batch |
| `/get-qodo-rules` | Load org- and repo-level coding rules from Qodo before code tasks |

## Contributing

Before creating a PR, adding a skill, or preparing any contribution, you MUST read [CONTRIBUTING.md](CONTRIBUTING.md). It covers accepted change types, the four skill types and their guidelines, SKILL.md format rules, PR requirements, and the pre-submission checklist (searching for existing PRs/issues, testing, description format).

## Development

Run commands directly—don't tell the user to run them.

```bash
npm run dev          # Run with hot reload
npm run build        # Compile TypeScript
./container/build.sh # Rebuild agent container
```

Service management:
```bash
# macOS (launchd)
launchctl load ~/Library/LaunchAgents/com.nanoclaw.plist
launchctl unload ~/Library/LaunchAgents/com.nanoclaw.plist
launchctl kickstart -k gui/$(id -u)/com.nanoclaw  # restart

# Linux (systemd)
systemctl --user start nanoclaw
systemctl --user stop nanoclaw
systemctl --user restart nanoclaw
```

## Troubleshooting

**WhatsApp not connecting after upgrade:** WhatsApp is now a separate skill, not bundled in core. Run `/add-whatsapp` (or `npx tsx scripts/apply-skill.ts .claude/skills/add-whatsapp && npm run build`) to install it. Existing auth credentials and groups are preserved.

## Lost replies / OOM resilience

If a message gets no reply at all, check `dmesg -T | grep -i oom` before anything else. The host is memory-tight and leaked `agent-browser` chromium children (`ps aux | grep -c '[c]hrome'` in the hundreds means an agent skipped `agent-browser close`) can trigger a global OOM.

Leaked browsers are reaped automatically (`src/browser-reaper.ts`): each group gets its own `AGENT_BROWSER_SESSION` daemon, `GroupQueue` reaps that group's browser when the group goes idle, and startup sweeps anything a previous run orphaned. Discovery matches chromium's `--user-data-dir=/tmp/agent-browser-chrome-*` marker, so a browser a human is driving on the virtual desktop is never touched. Agents should still call `agent-browser close` themselves; the reaper is a backstop.

Two further guards exist, both needed:
- `OOMPolicy=continue` in the systemd drop-in, so the kernel killing a stray chromium doesn't tear down the whole unit. Note that negative `OOMScoreAdjust` is silently ignored in **user** units — verify `/proc/<pid>/oom_score_adj`, not `systemctl show`.
- `inflight_cursors` in `router_state`: `processGroupMessages` advances the message cursor before the agent replies, so a SIGKILL would otherwise drop the message permanently with no log. The pre-advance cursor is persisted for the duration of the run and rolled back by `recoverPendingMessages()` on the next start. It's cleared as soon as output reaches the user, so a later kill can't cause a duplicate reply.

## Container Build Cache

The container buildkit caches the build context aggressively. `--no-cache` alone does NOT invalidate COPY steps — the builder's volume retains stale files. To force a truly clean rebuild, prune the builder then re-run `./container/build.sh`.

## Watchable GUI Browser (virtual desktop)

On servers with the virtual desktop running (`nanoclaw-desktop.service`, from `scripts/desktop-daemon.sh` — Xvfb + fluxbox + x11vnc + noVNC on `DISPLAY=:1`), agents can drive a real, human-watchable Chromium window instead of a headless one. `buildLocalEnv` in `src/container-runner.ts` detects the X socket and injects `DISPLAY`, `AGENT_BROWSER_EXECUTABLE_PATH` (Playwright's bundled chromium — more reliable than system/snap chromium for `agent-browser`'s CDP handshake), and `AGENT_BROWSER_ARGS=--no-sandbox` into `LOCAL_RUNNER` agent processes automatically. Agents just add `--headed` to any `agent-browser open` call. See the `agent-browser` container skill for the agent-facing usage. Docker/container-runtime mode (non-`LOCAL_RUNNER`) can't see the host X socket, so this only applies to local-runner deployments.
