import fs from 'fs';
import path from 'path';

import { GROUPS_DIR } from './config.js';
import { logger } from './logger.js';
import { Channel, NewMessage } from './types.js';
import { formatLocalTime } from './timezone.js';

export function escapeXml(s: string): string {
  if (!s) return '';
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function formatMessages(
  messages: NewMessage[],
  timezone: string,
): string {
  const lines = messages.map((m) => {
    const displayTime = formatLocalTime(m.timestamp, timezone);
    return `<message sender="${escapeXml(m.sender_name)}" time="${escapeXml(displayTime)}">${escapeXml(m.content)}</message>`;
  });

  const header = `<context timezone="${escapeXml(timezone)}" />\n`;

  return `${header}<messages>\n${lines.join('\n')}\n</messages>`;
}

export function stripInternalTags(text: string): string {
  return text.replace(/<internal>[\s\S]*?<\/internal>/g, '').trim();
}

/**
 * Resolve one [SEND_FILE:] path to a host path, or null if it escapes the
 * group's sandbox. Accepts container paths (/workspace/group/, /workspace/extra/)
 * and, for LOCAL_RUNNER mode where the agent's cwd is the real group dir,
 * host-absolute or group-relative paths.
 */
function resolveSendFilePath(
  rawPath: string,
  groupFolder: string,
): string | null {
  const groupRoot = path.resolve(GROUPS_DIR, groupFolder);
  const extraRoot = path.resolve(process.cwd(), 'extra');

  let candidate: string;
  if (rawPath.startsWith('/workspace/group/')) {
    candidate = path.resolve(
      groupRoot,
      rawPath.slice('/workspace/group/'.length),
    );
  } else if (rawPath.startsWith('/workspace/extra/')) {
    candidate = path.resolve(
      extraRoot,
      rawPath.slice('/workspace/extra/'.length),
    );
  } else if (path.isAbsolute(rawPath)) {
    candidate = path.resolve(rawPath);
  } else {
    candidate = path.resolve(groupRoot, rawPath);
  }

  // Confine to the group's own folder (or a validated extra mount). Use
  // realpath so symlinks inside the group dir can't point outside it.
  let real: string;
  try {
    real = fs.realpathSync(candidate);
  } catch {
    return null; // missing file — nothing to send
  }

  const withinRoot = (root: string) => {
    let realRoot: string;
    try {
      realRoot = fs.realpathSync(root);
    } catch {
      return false;
    }
    const rel = path.relative(realRoot, real);
    return rel !== '' && !rel.startsWith('..') && !path.isAbsolute(rel);
  };

  if (!withinRoot(groupRoot) && !withinRoot(extraRoot)) return null;
  if (!fs.statSync(real).isFile()) return null;
  return real;
}

/**
 * Extract [SEND_FILE: path] tags from agent output.
 * Returns the cleaned text and a list of resolved host file paths.
 * Paths that don't exist or escape the group's folder are dropped.
 */
export function extractSendFileTags(
  text: string,
  groupFolder: string,
): { cleanText: string; filePaths: string[] } {
  const filePaths: string[] = [];
  const cleanText = text
    .replace(/\[SEND_FILE:\s*([^\]]+)\]/g, (_match, p: string) => {
      const resolved = resolveSendFilePath(p.trim(), groupFolder);
      if (resolved) {
        filePaths.push(resolved);
      } else {
        logger.warn(
          { rawPath: p.trim(), groupFolder },
          'SEND_FILE path rejected (missing or outside group folder)',
        );
      }
      return '';
    })
    .trim();
  return { cleanText, filePaths };
}

export function formatOutbound(rawText: string): string {
  const text = stripInternalTags(rawText);
  if (!text) return '';
  return text;
}

export function routeOutbound(
  channels: Channel[],
  jid: string,
  text: string,
): Promise<void> {
  const channel = channels.find((c) => c.ownsJid(jid) && c.isConnected());
  if (!channel) throw new Error(`No channel for JID: ${jid}`);
  return channel.sendMessage(jid, text);
}

export function findChannel(
  channels: Channel[],
  jid: string,
): Channel | undefined {
  return channels.find((c) => c.ownsJid(jid));
}
