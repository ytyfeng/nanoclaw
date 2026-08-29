import { spawn } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { describe, it, expect, beforeEach, afterEach } from 'vitest';

import {
  browserSessionName,
  countAgentBrowserChromeProcesses,
  reapAllOrphanedBrowsers,
  reapGroupBrowser,
} from './browser-reaper.js';

// Stand-ins for chromium: sleeping processes whose argv carries the same
// --user-data-dir=/tmp/agent-browser-chrome-<uuid> marker real agent-browser
// chromium has, launched from a script named "chrome" so the /chrome|chromium/
// filter matches. This exercises the reaper's real discovery and kill paths
// without launching a 200MB browser.
const TMP = os.tmpdir();
let scratchDirs: string[] = [];
let spawned: number[] = [];
let fakeChromePath: string;
let runtimeDir: string;
let originalXdg: string | undefined;

// A wrapper script rather than a symlink to sleep: sleep rejects unknown
// flags, and `exec sleep` would replace argv and drop the marker the reaper
// matches on. Calling sleep as a child keeps the full argv visible in ps.
function makeFakeChromeBinary(): string {
  const dir = fs.mkdtempSync(path.join(TMP, 'reaper-bin-'));
  scratchDirs.push(dir);
  const script = path.join(dir, 'chrome');
  fs.writeFileSync(script, '#!/bin/sh\nsleep "$1"\n', { mode: 0o755 });
  return script;
}

function spawnFakeChrome(profileDir: string): number {
  const proc = spawn(fakeChromePath, ['300', `--user-data-dir=${profileDir}`], {
    stdio: 'ignore',
    detached: true,
  });
  proc.unref();
  spawned.push(proc.pid!);
  return proc.pid!;
}

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function waitForExit(pid: number, timeoutMs = 4000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (alive(pid) && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 25));
  }
}

beforeEach(() => {
  scratchDirs = [];
  spawned = [];
  fakeChromePath = makeFakeChromeBinary();
  originalXdg = process.env.XDG_RUNTIME_DIR;
  const xdg = fs.mkdtempSync(path.join(TMP, 'reaper-xdg-'));
  scratchDirs.push(xdg);
  process.env.XDG_RUNTIME_DIR = xdg;
  runtimeDir = path.join(xdg, 'agent-browser');
  fs.mkdirSync(runtimeDir, { recursive: true });
});

afterEach(() => {
  for (const pid of spawned) {
    try {
      process.kill(pid, 'SIGKILL');
    } catch {
      /* already gone */
    }
  }
  for (const dir of scratchDirs) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
  if (originalXdg === undefined) delete process.env.XDG_RUNTIME_DIR;
  else process.env.XDG_RUNTIME_DIR = originalXdg;
});

describe('browserSessionName', () => {
  it('namespaces per group so each gets its own daemon', () => {
    expect(browserSessionName('slack_claw4')).toBe('nanoclaw-slack_claw4');
    expect(browserSessionName('slack_claw4')).not.toBe(
      browserSessionName('slack_claw8'),
    );
  });

  it('strips characters that would escape the runtime dir', () => {
    expect(browserSessionName('../../etc/passwd')).toBe(
      'nanoclaw-______etc_passwd',
    );
    expect(browserSessionName('a b/c')).not.toMatch(/[/\s]/);
  });
});

describe('reapGroupBrowser', () => {
  it('kills the daemon and the chromium tree it owns', async () => {
    const profile = path.join(TMP, 'agent-browser-chrome-owned-by-daemon');
    fs.mkdirSync(profile, { recursive: true });
    scratchDirs.push(profile);

    // Daemon stand-in. Its chromium subtree is covered by the sweep test
    // below; here the point is that the daemon itself dies.
    const daemon = spawn('/bin/sh', ['-c', `exec sleep 300`], {
      stdio: 'ignore',
      detached: true,
    });
    daemon.unref();
    spawned.push(daemon.pid!);
    fs.writeFileSync(
      path.join(runtimeDir, `${browserSessionName('g1')}.pid`),
      String(daemon.pid),
    );

    reapGroupBrowser('g1');

    await waitForExit(daemon.pid!);
    expect(alive(daemon.pid!)).toBe(false);
  });

  it('removes the session pidfile so the next open does not wait on a dead daemon', () => {
    const pidfile = path.join(runtimeDir, `${browserSessionName('g1')}.pid`);
    const sock = path.join(runtimeDir, `${browserSessionName('g1')}.sock`);
    const daemon = spawn('/bin/sh', ['-c', 'exec sleep 300'], {
      stdio: 'ignore',
      detached: true,
    });
    daemon.unref();
    spawned.push(daemon.pid!);
    fs.writeFileSync(pidfile, String(daemon.pid));
    fs.writeFileSync(sock, '');

    reapGroupBrowser('g1');

    expect(fs.existsSync(pidfile)).toBe(false);
    expect(fs.existsSync(sock)).toBe(false);
  });

  it("leaves another group's daemon and browser untouched", () => {
    const otherDaemon = spawn('/bin/sh', ['-c', 'exec sleep 300'], {
      stdio: 'ignore',
      detached: true,
    });
    otherDaemon.unref();
    spawned.push(otherDaemon.pid!);
    const otherPidfile = path.join(
      runtimeDir,
      `${browserSessionName('other')}.pid`,
    );
    fs.writeFileSync(otherPidfile, String(otherDaemon.pid));

    reapGroupBrowser('g1');

    expect(alive(otherDaemon.pid!)).toBe(true);
    expect(fs.existsSync(otherPidfile)).toBe(true);
  });

  it('no-ops for a group that never opened a browser', () => {
    expect(() => reapGroupBrowser('never-browsed')).not.toThrow();
  });

  it('tolerates a stale pidfile pointing at a dead process', async () => {
    const dead = spawn('/bin/sh', ['-c', 'exit 0'], { stdio: 'ignore' });
    const deadPid = dead.pid!;
    await waitForExit(deadPid);
    fs.writeFileSync(
      path.join(runtimeDir, `${browserSessionName('g1')}.pid`),
      String(deadPid),
    );

    expect(() => reapGroupBrowser('g1')).not.toThrow();
  });
});

describe('reapAllOrphanedBrowsers', () => {
  it('kills orphaned chromium identified by the profile-dir marker', async () => {
    const profile = path.join(TMP, 'agent-browser-chrome-orphan-test');
    fs.mkdirSync(profile, { recursive: true });
    const pid = spawnFakeChrome(profile);

    expect(countAgentBrowserChromeProcesses()).toBeGreaterThan(0);

    reapAllOrphanedBrowsers('test');

    await waitForExit(pid);
    expect(alive(pid)).toBe(false);
  });

  it('ignores chromium that is not agent-browser (no marker)', () => {
    // Same fake binary, but a profile dir outside agent-browser's namespace:
    // this stands in for a browser a human is running on the virtual desktop.
    const unrelated = fs.mkdtempSync(path.join(TMP, 'someones-own-chrome-'));
    scratchDirs.push(unrelated);
    const pid = spawnFakeChrome(unrelated);

    reapAllOrphanedBrowsers('test');

    expect(alive(pid)).toBe(true);
  });

  it('clears stale profile dirs left behind by dead browsers', () => {
    const stale = path.join(TMP, 'agent-browser-chrome-stale-dir-test');
    fs.mkdirSync(stale, { recursive: true });
    fs.writeFileSync(path.join(stale, 'Cookies'), 'x');

    reapAllOrphanedBrowsers('test');

    expect(fs.existsSync(stale)).toBe(false);
  });

  it('no-ops when there is nothing to reap', () => {
    expect(() => reapAllOrphanedBrowsers('test')).not.toThrow();
  });

  it('does not throw when XDG_RUNTIME_DIR is unset', () => {
    delete process.env.XDG_RUNTIME_DIR;
    expect(() => reapAllOrphanedBrowsers('test')).not.toThrow();
    expect(() => reapGroupBrowser('g1')).not.toThrow();
  });
});
