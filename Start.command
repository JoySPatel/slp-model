#!/bin/zsh
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then
  .venv/bin/python app.py &
else
  python3 app.py &
fi
app_pid=$!
trap 'kill "$app_pid" 2>/dev/null' EXIT INT TERM
sleep 1
if kill -0 "$app_pid" 2>/dev/null; then
  open http://127.0.0.1:8765
  wait "$app_pid"
fi
