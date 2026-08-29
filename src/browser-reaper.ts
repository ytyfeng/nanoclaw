/**
 * Reaps orphaned agent-browser chromium processes.
 *
 * The agent-browser daemon re-parents itself to PID 1, so it deliberately
 * outlives the agent process that spawned it. When an agent finishes without
 * calling `agent-browser close`, the daemon and its whole chromium tree keep
 * running. On 2026-08-28 that leaked 334 chrome processes (~2GB) and triggered
 * a global OOM that took NanoClaw down mid-reply.
 */
import { execFileSync, spawnSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { logger } from './logger.js';

// Chromium launched by agent-browser always carries a profile dir of the form
// --user-data-dir=/tmp/agent-browser-chrome-<uuid>. Verified present on every
// process in the tree (root browser plus renderer/GPU/zygote children), which
// makes it a reliable discriminator against any unrelated chromium on the host.
const CHROME_PROFILE_MARKER = 'agent-browser-chrome-';

/**
 * agent-browser session name for a group. Each group gets its own daemon so a
 * reap for one group can't kill a browser another group's agent is driving.
 * Sanitized because the name becomes a filename in $XDG_RUNTIME_DIR.
 */
export function browserSessionName(groupFolder: string): string {
  return `nanoclaw-${groupFolder.replace(/[^A-Za-z0-9_-]/g, '_')}`;
}

function agentBrowserRuntimeDir(): string | undefined {
  const xdg = process.env.XDG_RUNTIME_DIR;
  if (!xdg) return undefined;
  const dir = path.join(xdg, 'agent-browser');
  return fs.existsSync(dir) ? dir : undefined;
}

/** PIDs of live agent-browser daemons, keyed by session name from its pidfile. */
function readDaemonPids(): Array<{ session: string; pid: number }> {
  const dir = agentBrowserRuntimeDir();
  if (!dir) return [];
  const found: Array<{ session: string; pid: number }> = [];
  for (const file of fs.readdirSync(dir)) {
    if (!file.endsWith('.pid')) continue;
    const raw = fs.readFileSync(path.join(dir, file), 'utf-8').trim();
    const pid = Number.parseInt(raw, 10);
    if (Number.isInteger(pid) && pid > 1) {
      found.push({ session: file.replace(/\.pid$/, ''), pid });
    }
  }
  return found;
}

function isAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function readProcessTable(): Array<{
  pid: number;
  ppid: number;
  args: string;
}> {
  let out: string;
  try {
    // `-ww` so long chromium cmdlines aren't truncated before the marker.
    out = execFileSync('ps', ['-eo', 'pid=,ppid=,args=', '-ww'], {
      encoding: 'utf-8',
      maxBuffer: 8 * 1024 * 1024,
    });
  } catch {
    return [];
  }

  const rows: Array<{ pid: number; ppid: number; args: string }> = [];
  for (const line of out.split('\n')) {
    const m = line.trim().match(/^(\d+)\s+(\d+)\s+(.*)$/);
    if (!m) continue;
    rows.push({ pid: Number(m[1]), ppid: Number(m[2]), args: m[3] });
  }
  return rows;
}

/**
 * PIDs of every agent-browser chromium process, oldest ancestor first.
 * Matches the profile-dir marker rather than the process name so unrelated
 * chromium (a human's browser on the virtual desktop) is never touched.
 */
function findAgentBrowserChromePids(): number[] {
  const rows = readProcessTable().filter(
    (r) =>
      r.args.includes(CHROME_PROFILE_MARKER) && /chrome|chromium/i.test(r.args),
  );

  // Kill parents before children so a dying parent takes its subtree with it
  // and we don't waste signals on already-reaped PIDs.
  const pids = new Set(rows.map((r) => r.pid));
  return rows
    .sort((a, b) => Number(pids.has(a.ppid)) - Number(pids.has(b.ppid)))
    .map((r) => r.pid);
}

/**
 * Chromium PIDs under a specific daemon, deepest-last.
 *
 * Walks ppid links from the daemon rather than matching the profile marker
 * globally, so only this session's browser is returned even when several
 * groups have browsers open at once. Must be called while the daemon is still
 * alive — once it dies its children re-parent to PID 1 and the link is lost.
 */
function descendantChromePids(daemonPid: number): number[] {
  const rows = readProcessTable();
  const byParent = new Map<number, number[]>();
  for (const r of rows) {
    const siblings = byParent.get(r.ppid);
    if (siblings) siblings.push(r.pid);
    else byParent.set(r.ppid, [r.pid]);
  }

  const found: number[] = [];
  const queue = [daemonPid];
  const seen = new Set<number>([daemonPid]);
  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const child of byParent.get(current) ?? []) {
      if (seen.has(child)) continue;
      seen.add(child);
      found.push(child);
      queue.push(child);
    }
  }
  return found.reverse();
}

/** Remove stale /tmp/agent-browser-chrome-* profile dirs left by dead browsers. */
function removeStaleProfileDirs(): number {
  let removed = 0;
  let entries: string[];
  try {
    entries = fs.readdirSync(os.tmpdir());
  } catch {
    return 0;
  }
  for (const entry of entries) {
    if (!entry.startsWith('agent-browser-chrome-')) continue;
    const full = path.join(os.tmpdir(), entry);
    try {
      fs.rmSync(full, { recursive: true, force: true });
      removed++;
    } catch {
      // Still in use or racing another reap — leave it for the next pass.
    }
  }
  return removed;
}

/**
 * Ask a daemon to shut its browser down gracefully.
 *
 * Best-effort only, with a short timeout: session-scoped `close` was observed
 * hanging indefinitely against a daemon busy mid-navigation. The SIGKILL below
 * is what actually guarantees the processes are gone.
 */
function tryGracefulClose(session: string): void {
  const result = spawnSync('agent-browser', ['close', '--all'], {
    timeout: 10000,
    stdio: 'ignore',
    env: { ...process.env, AGENT_BROWSER_SESSION: session },
  });
  if (result.error) {
    logger.debug(
      { session, err: result.error.message },
      'agent-browser close unavailable, falling back to signals',
    );
  }
}

/** Remove a session's pidfile/socket so the next `open` doesn't wait on a dead daemon. */
function removeSessionRuntimeFiles(session: string): void {
  const dir = agentBrowserRuntimeDir();
  if (!dir) return;
  for (const file of fs.readdirSync(dir)) {
    if (file !== session && !file.startsWith(`${session}.`)) continue;
    try {
      fs.rmSync(path.join(dir, file), { force: true });
    } catch {
      // Leave it; a live daemon will rewrite it.
    }
  }
}

/**
 * Kill the agent-browser daemon and chromium tree belonging to one group.
 *
 * Scoped to the group's own session so concurrently-running agents in other
 * groups keep their browsers. Safe to call when the group never opened one:
 * it no-ops.
 */
export function reapGroupBrowser(groupFolder: string): void {
  const session = browserSessionName(groupFolder);
  const daemon = readDaemonPids().find(
    (d) => d.session === session && isAlive(d.pid),
  );
  const ownedBefore = daemon ? descendantChromePids(daemon.pid) : [];
  if (!daemon && ownedBefore.length === 0) return;

  logger.info(
    { groupFolder, session, chromeCount: ownedBefore.length },
    'Reaping group browser',
  );

  tryGracefulClose(session);

  // Daemons ignore SIGTERM (verified — the process stays alive), so use
  // SIGKILL. This cascade-reaps most of the chromium tree with it.
  if (daemon && isAlive(daemon.pid)) {
    try {
      process.kill(daemon.pid, 'SIGKILL');
    } catch (err) {
      logger.warn({ session, pid: daemon.pid, err }, 'Failed to kill daemon');
    }
  }

  // Chromium children run in their own process group and can survive as PPID-1
  // orphans, so sweep the PIDs recorded before the daemon died.
  let killed = 0;
  for (const pid of ownedBefore) {
    if (!isAlive(pid)) continue;
    try {
      process.kill(pid, 'SIGKILL');
      killed++;
    } catch {
      // Already gone — the parent took it down.
    }
  }

  removeSessionRuntimeFiles(session);
  logger.info({ groupFolder, chromeKilled: killed }, 'Reaped group browser');
}

/**
 * Startup sweep: kill every agent-browser daemon and chromium tree, then clear
 * stale runtime and profile files.
 *
 * Only safe at startup, when no agent of ours can be driving a browser — any
 * daemon alive at that point outlived the process that owned it.
 */
export function reapAllOrphanedBrowsers(reason: string): void {
  const daemons = readDaemonPids().filter((d) => isAlive(d.pid));
  const chromePids = findAgentBrowserChromePids();
  if (daemons.length === 0 && chromePids.length === 0) {
    removeStaleProfileDirs();
    return;
  }

  logger.warn(
    { reason, chromeCount: chromePids.length, daemonCount: daemons.length },
    'Reaping orphaned agent-browser processes left by a previous run',
  );

  for (const { session } of daemons) tryGracefulClose(session);

  for (const { session, pid } of daemons) {
    if (!isAlive(pid)) continue;
    try {
      process.kill(pid, 'SIGKILL');
    } catch (err) {
      logger.warn({ session, pid, err }, 'Failed to kill daemon');
    }
  }

  let killed = 0;
  for (const pid of findAgentBrowserChromePids()) {
    if (!isAlive(pid)) continue;
    try {
      process.kill(pid, 'SIGKILL');
      killed++;
    } catch {
      // Already gone.
    }
  }

  const dir = agentBrowserRuntimeDir();
  if (dir) {
    for (const file of fs.readdirSync(dir)) {
      try {
        fs.rmSync(path.join(dir, file), { force: true });
      } catch {
        // Leave it.
      }
    }
  }

  const profileDirs = removeStaleProfileDirs();
  logger.info(
    {
      reason,
      chromeKilled: killed,
      daemonsKilled: daemons.length,
      profileDirs,
    },
    'Reaped orphaned agent-browser processes',
  );
}

/** Number of live agent-browser chromium processes (for monitoring/tests). */
export function countAgentBrowserChromeProcesses(): number {
  return findAgentBrowserChromePids().length;
}
