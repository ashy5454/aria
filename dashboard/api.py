"""
Crucible dashboard API — serves the phone UI and exposes loop controls.

Usage:
    pip install flask
    python dashboard/api.py

Then open http://YOUR_VM_IP:8080 on your phone.
"""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

ROOT = Path(__file__).parent.parent
RESULTS_TSV  = ROOT / "results" / "results.tsv"
SESSION_LOG  = ROOT / "results" / "session.log"
CONCLUSION   = ROOT / "results" / "CONCLUSION.md"
RESEARCH_CFG = ROOT / "research.yaml"
STATIC_DIR   = Path(__file__).parent / "static"

PIN = os.environ.get("CRUCIBLE_PIN", "1234")

app = Flask(__name__, static_folder=str(STATIC_DIR))

_loop_proc: subprocess.Popen | None = None


# ── helpers ──────────────────────────────────────────────────────────────────

def _read_tail(path: Path, n: int = 50) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()
    return lines[-n:]


def _parse_results() -> list[dict]:
    if not RESULTS_TSV.exists():
        return []
    rows = []
    for line in RESULTS_TSV.read_text(errors="replace").splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            rows.append({"exp": parts[0], "metric": parts[1], "tag": parts[2] if len(parts) > 2 else ""})
    return rows[-20:]


def _best_metric(rows: list[dict]) -> str | None:
    vals = []
    for r in rows:
        try:
            vals.append(float(r["metric"]))
        except ValueError:
            pass
    if not vals:
        return None
    try:
        import yaml
        cfg = yaml.safe_load(RESEARCH_CFG.read_text())
        direction = cfg.get("eval", {}).get("direction", "lower")
        return str(min(vals) if direction == "lower" else max(vals))
    except Exception:
        return str(min(vals))


def _loop_running() -> bool:
    global _loop_proc
    if _loop_proc is None:
        return False
    if _loop_proc.poll() is not None:
        _loop_proc = None
        return False
    return True


def _require_pin() -> bool:
    data = request.get_json(silent=True) or {}
    return str(data.get("pin", "")) == PIN


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/status")
def status():
    rows = _parse_results()
    conclusion_text = CONCLUSION.read_text(errors="replace") if CONCLUSION.exists() else None
    return jsonify({
        "running":    _loop_running(),
        "total_exps": len(rows),
        "best":       _best_metric(rows),
        "conclusion": conclusion_text,
    })


@app.route("/results")
def results():
    return jsonify(_parse_results())


@app.route("/log")
def log():
    return jsonify({"lines": _read_tail(SESSION_LOG, 60)})


@app.route("/conclusion")
def conclusion():
    if not CONCLUSION.exists():
        return jsonify({"text": None})
    return jsonify({"text": CONCLUSION.read_text(errors="replace")})


@app.route("/start", methods=["POST"])
def start():
    if not _require_pin():
        return jsonify({"error": "wrong PIN"}), 403
    global _loop_proc
    if _loop_running():
        return jsonify({"ok": False, "msg": "already running"})
    _loop_proc = subprocess.Popen(
        [sys.executable, str(ROOT / "loop_v2.py")],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return jsonify({"ok": True, "pid": _loop_proc.pid})


@app.route("/stop", methods=["POST"])
def stop():
    if not _require_pin():
        return jsonify({"error": "wrong PIN"}), 403
    global _loop_proc
    if not _loop_running():
        return jsonify({"ok": False, "msg": "not running"})
    _loop_proc.send_signal(signal.SIGTERM)
    _loop_proc = None
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Crucible dashboard → http://0.0.0.0:{port}")
    print(f"  → http://0.0.0.0:{port}  (phone: use your VM external IP)")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
