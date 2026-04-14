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
 * Extract [SEND_FILE: /workspace/group/foo.csv] tags from agent output.
 * Returns the cleaned text and a list of resolved host file paths.
 * Container path /workspace/group/ maps to groups/{groupFolder}/ on host.
 */
export function extractSendFileTags(
  text: string,
  groupFolder: string,
): { cleanText: string; filePaths: string[] } {
  const filePaths: string[] = [];
  const cleanText = text
    .replace(/\[SEND_FILE:\s*([^\]]+)\]/g, (_match, p: string) => {
      const containerPath = p.trim();
      const hostPath = containerPath.startsWith('/workspace/group/')
        ? containerPath.replace('/workspace/group/', `groups/${groupFolder}/`)
        : containerPath.startsWith('/workspace/extra/')
          ? containerPath.replace('/workspace/extra/', 'extra/')
          : null;
      if (hostPath) filePaths.push(hostPath);
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
