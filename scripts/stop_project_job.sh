#!/usr/bin/env bash
set -euo pipefail

PID="${1:?usage: stop_project_job.sh PID}"
if [[ ! "${PID}" =~ ^[0-9]+$ ]]; then
  echo "PID must be numeric" >&2
  exit 1
fi

stop_tree() {
  local pid="$1"
  local child
  for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
    stop_tree "${child}"
  done
  kill -TERM "${pid}" 2>/dev/null || true
}

if ! kill -0 "${PID}" 2>/dev/null; then
  echo "process ${PID} is not running"
  exit 0
fi
stop_tree "${PID}"
sleep 2
if kill -0 "${PID}" 2>/dev/null; then
  echo "process ${PID} is still running; inspect before using KILL" >&2
  exit 1
fi
echo "process tree ${PID} stopped"
