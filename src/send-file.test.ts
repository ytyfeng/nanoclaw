import fs from 'fs';
import path from 'path';

import { describe, it, expect, beforeAll, afterAll } from 'vitest';

import { GROUPS_DIR } from './config.js';
import { extractSendFileTags } from './router.js';

const GROUP = '__sendfile_test__';
const groupDir = path.join(GROUPS_DIR, GROUP);

beforeAll(() => {
  fs.mkdirSync(path.join(groupDir, 'out'), { recursive: true });
  fs.writeFileSync(path.join(groupDir, 'report.csv'), 'a,b\n');
  fs.writeFileSync(path.join(groupDir, 'out', 'cv.pdf'), '%PDF-1.4\n');
});

afterAll(() => {
  fs.rmSync(groupDir, { recursive: true, force: true });
});

describe('extractSendFileTags', () => {
  it('resolves a container path', () => {
    const { cleanText, filePaths } = extractSendFileTags(
      'Done [SEND_FILE: /workspace/group/report.csv]',
      GROUP,
    );
    expect(cleanText).toBe('Done');
    expect(filePaths).toEqual([
      fs.realpathSync(path.join(groupDir, 'report.csv')),
    ]);
  });

  // Local-runner mode: the agent's cwd is the real group dir, so it emits
  // host-absolute paths. These were previously dropped silently.
  it('resolves a host-absolute path inside the group', () => {
    const abs = path.join(groupDir, 'out', 'cv.pdf');
    const { filePaths } = extractSendFileTags(`[SEND_FILE: ${abs}]`, GROUP);
    expect(filePaths).toEqual([fs.realpathSync(abs)]);
  });

  it('resolves a group-relative path', () => {
    const { filePaths } = extractSendFileTags('[SEND_FILE: out/cv.pdf]', GROUP);
    expect(filePaths).toEqual([
      fs.realpathSync(path.join(groupDir, 'out', 'cv.pdf')),
    ]);
  });

  it('extracts multiple tags', () => {
    const { cleanText, filePaths } = extractSendFileTags(
      'Two files [SEND_FILE: report.csv] [SEND_FILE: out/cv.pdf]',
      GROUP,
    );
    expect(cleanText).toBe('Two files');
    expect(filePaths).toHaveLength(2);
  });

  it('rejects traversal out of the group folder', () => {
    const { filePaths } = extractSendFileTags(
      '[SEND_FILE: /workspace/group/../../.env] [SEND_FILE: ../../package.json]',
      GROUP,
    );
    expect(filePaths).toEqual([]);
  });

  it('rejects unrelated absolute host paths', () => {
    const { filePaths } = extractSendFileTags(
      '[SEND_FILE: /etc/passwd]',
      GROUP,
    );
    expect(filePaths).toEqual([]);
  });

  it('rejects a symlink pointing outside the group folder', () => {
    const link = path.join(groupDir, 'escape.txt');
    fs.symlinkSync('/etc/passwd', link);
    try {
      const { filePaths } = extractSendFileTags(
        '[SEND_FILE: escape.txt]',
        GROUP,
      );
      expect(filePaths).toEqual([]);
    } finally {
      fs.rmSync(link, { force: true });
    }
  });

  it('drops missing files and directories', () => {
    const { filePaths } = extractSendFileTags(
      '[SEND_FILE: nope.csv] [SEND_FILE: out]',
      GROUP,
    );
    expect(filePaths).toEqual([]);
  });

  it('leaves text without tags untouched', () => {
    const { cleanText, filePaths } = extractSendFileTags('just text', GROUP);
    expect(cleanText).toBe('just text');
    expect(filePaths).toEqual([]);
  });
});
