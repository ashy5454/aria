#!/usr/bin/env bash
# watchdog.sh — restarts loop_v2.py if it exits for any reason
# Run in a separate tmux window: tmux new-window -n watchdog "bash watchdog.sh"

LOOP_SCRIPT="loop_v2.py"
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/results/watchdog.log"
RESTART_DELAY=10  # seconds between crash and restart

mkdir -p "$DIR/results"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] watchdog started — watching $LOOP_SCRIPT" | tee -a "$LOG"

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting loop_v2.py ..." | tee -a "$LOG"
    cd "$DIR" && python3 "$LOOP_SCRIPT"
    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] loop exited with code $EXIT_CODE" | tee -a "$LOG"

    # Exit cleanly if loop wrote CONCLUSION.md (goal reached / ceiling found)
    if [[ -f "$DIR/results/CONCLUSION.md" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] CONCLUSION.md found — loop is done, watchdog exiting." | tee -a "$LOG"
        break
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] restarting in ${RESTART_DELAY}s ..." | tee -a "$LOG"
    sleep $RESTART_DELAY
done
