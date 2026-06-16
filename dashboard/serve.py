"""
dashboard/serve.py — Live leaderboard at http://localhost:8000

Shows: val_bpb chart, experiment log, best result so far.
Auto-refreshes every 60 seconds.

Usage:
    python cmp_autoresearch/dashboard/serve.py
"""

import http.server
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_TSV = ROOT / "results" / "results.tsv"
LAST_RESULT = ROOT / "harness" / "last_result.json"
PORT = 8000


def build_html() -> str:
    rows = []
    if RESULTS_TSV.exists():
        lines = RESULTS_TSV.read_text().strip().splitlines()
        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) >= 5:
                    rows.append({
                        "commit": parts[0], "val_bpb": parts[1],
                        "params_M": parts[2], "status": parts[3], "description": parts[4]
                    })

    last = {}
    if LAST_RESULT.exists():
        try:
            last = json.loads(LAST_RESULT.read_text())
        except Exception:
            pass

    best_keeps = [r for r in rows if r["status"] == "keep"]
    best_bpb = min((float(r["val_bpb"]) for r in best_keeps), default=9.999)
    gap_vs_gate = best_bpb - 1.91

    bpb_values = []
    for r in rows:
        try:
            bpb_values.append({"bpb": float(r["val_bpb"]), "status": r["status"], "desc": r["description"]})
        except Exception:
            pass

    chart_data = json.dumps(bpb_values)

    rows_html = "".join(
        f'<tr class="{r["status"]}">'
        f'<td>{r["commit"]}</td>'
        f'<td><b>{r["val_bpb"]}</b></td>'
        f'<td>{r["params_M"]}</td>'
        f'<td class="status-{r["status"]}">{r["status"].upper()}</td>'
        f'<td>{r["description"]}</td>'
        f'</tr>'
        for r in reversed(rows)
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta http-equiv="refresh" content="60">
<title>CMP Autoresearch</title>
<style>
body {{ font-family: monospace; background: #0d0d0d; color: #e0e0e0; padding: 20px; }}
h1 {{ color: #7ec8e3; }}
.stats {{ display: flex; gap: 40px; margin: 20px 0; }}
.stat {{ background: #1a1a2e; padding: 15px 25px; border-radius: 8px; }}
.stat-label {{ font-size: 12px; color: #888; }}
.stat-value {{ font-size: 28px; font-weight: bold; color: #7ec8e3; }}
.gate {{ color: #ff6b6b; }}
.best {{ color: #51cf66; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
th {{ background: #1a1a2e; padding: 8px 12px; text-align: left; color: #7ec8e3; }}
td {{ padding: 6px 12px; border-bottom: 1px solid #222; }}
tr.keep {{ background: #0d1f0d; }}
tr.discard {{ background: #1a0d0d; opacity: 0.7; }}
tr.crash {{ background: #1a1a0d; opacity: 0.5; }}
.status-keep {{ color: #51cf66; }}
.status-discard {{ color: #ff6b6b; }}
.status-crash {{ color: #ffd43b; }}
canvas {{ margin: 20px 0; background: #1a1a2e; border-radius: 8px; }}
</style>
</head>
<body>
<h1>CMP Autoresearch Loop</h1>
<div class="stats">
  <div class="stat">
    <div class="stat-label">Experiments run</div>
    <div class="stat-value">{len(rows)}</div>
  </div>
  <div class="stat">
    <div class="stat-label">Best val_bpb</div>
    <div class="stat-value best">{best_bpb:.4f}</div>
  </div>
  <div class="stat">
    <div class="stat-label">vs Transformer gate (1.91)</div>
    <div class="stat-value {'best' if gap_vs_gate < 0 else 'gate'}">{gap_vs_gate:+.4f}</div>
  </div>
  <div class="stat">
    <div class="stat-label">Last eval</div>
    <div class="stat-value">{last.get('val_bpb', '—')}</div>
  </div>
</div>

<canvas id="chart" width="900" height="200"></canvas>

<table>
<tr><th>Commit</th><th>val_bpb</th><th>Params (M)</th><th>Status</th><th>Description</th></tr>
{rows_html}
</table>

<script>
const data = {chart_data};
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;
const PAD = 40;

if (data.length > 0) {{
  const bpbs = data.map(d => d.bpb).filter(b => b < 9);
  const minB = Math.min(...bpbs, 1.91) - 0.05;
  const maxB = Math.max(...bpbs) + 0.05;
  const scaleX = (i) => PAD + i * (W - 2*PAD) / Math.max(data.length - 1, 1);
  const scaleY = (b) => H - PAD - (b - minB) / (maxB - minB) * (H - 2*PAD);

  // gate line
  ctx.strokeStyle = '#ff6b6b'; ctx.lineWidth = 1; ctx.setLineDash([5, 5]);
  const gy = scaleY(1.91);
  ctx.beginPath(); ctx.moveTo(PAD, gy); ctx.lineTo(W - PAD, gy); ctx.stroke();
  ctx.fillStyle = '#ff6b6b'; ctx.font = '11px monospace';
  ctx.fillText('1.91 gate', W - PAD - 55, gy - 4);
  ctx.setLineDash([]);

  // data points
  data.forEach((d, i) => {{
    if (d.bpb >= 9) return;
    const x = scaleX(i), y = scaleY(d.bpb);
    ctx.fillStyle = d.status === 'keep' ? '#51cf66' : d.status === 'crash' ? '#ffd43b' : '#666';
    ctx.beginPath(); ctx.arc(x, y, 4, 0, 2*Math.PI); ctx.fill();
  }});

  // connect keeps
  ctx.strokeStyle = '#51cf66'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  let started = false;
  data.forEach((d, i) => {{
    if (d.status === 'keep' && d.bpb < 9) {{
      if (!started) {{ ctx.moveTo(scaleX(i), scaleY(d.bpb)); started = true; }}
      else ctx.lineTo(scaleX(i), scaleY(d.bpb));
    }}
  }});
  ctx.stroke();
}}
</script>
<p style="color:#555; font-size:11px">Auto-refresh every 60s. Last updated at page load.</p>
</body>
</html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        html = build_html().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, *args):
        pass  # silence request logs


if __name__ == "__main__":
    print(f"Dashboard at http://localhost:{PORT}  (Ctrl+C to stop)")
    httpd = http.server.HTTPServer(("", PORT), Handler)
    httpd.serve_forever()
