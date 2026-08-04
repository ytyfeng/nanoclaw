#!/bin/bash
# Virtual desktop for host-level "computer use": Xvfb -> fluxbox -> x11vnc -> noVNC.
# Runs on the HOST (not in a NanoClaw agent container) so a Claude Code session with
# a Bash tool can drive it directly via `DISPLAY=:1 xdotool ...` / `scrot`, and a human
# can watch the same desktop live over noVNC (tunneled via SSH, never publicly exposed).
set -e

DISPLAY_NUM="${NANOCLAW_DESKTOP_DISPLAY:-:1}"
WIDTH="${NANOCLAW_DESKTOP_WIDTH:-1280}"
HEIGHT="${NANOCLAW_DESKTOP_HEIGHT:-800}"
VNC_PORT="${NANOCLAW_DESKTOP_VNC_PORT:-5900}"
NOVNC_PORT="${NANOCLAW_DESKTOP_NOVNC_PORT:-6080}"
VNC_PASSWD_FILE="${NANOCLAW_DESKTOP_VNC_PASSWD_FILE:-$HOME/.vnc/passwd}"
NOVNC_WEBROOT="${NANOCLAW_DESKTOP_NOVNC_WEBROOT:-/usr/share/novnc}"

cleanup() {
  jobs -p | xargs -r kill 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "Starting Xvfb on $DISPLAY_NUM (${WIDTH}x${HEIGHT})..."
Xvfb "$DISPLAY_NUM" -screen 0 "${WIDTH}x${HEIGHT}x24" -nolisten tcp &

# Wait for the X socket to exist before starting anything that connects to it.
SOCKET_NUM="${DISPLAY_NUM#:}"
for i in $(seq 1 50); do
  [ -e "/tmp/.X11-unix/X${SOCKET_NUM}" ] && break
  sleep 0.1
done

export DISPLAY="$DISPLAY_NUM"

echo "Starting fluxbox..."
fluxbox &
sleep 1

echo "Starting x11vnc on 127.0.0.1:$VNC_PORT..."
x11vnc -display "$DISPLAY_NUM" -rfbauth "$VNC_PASSWD_FILE" -rfbport "$VNC_PORT" \
  -localhost -forever -shared -quiet &

echo "Starting noVNC (websockify) on 127.0.0.1:$NOVNC_PORT..."
websockify --web "$NOVNC_WEBROOT" "127.0.0.1:$NOVNC_PORT" "127.0.0.1:$VNC_PORT" &

wait
