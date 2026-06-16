#!/bin/bash
# Crucible dashboard launcher
# Usage: bash dashboard/start.sh
# Optional: CRUCIBLE_PIN=9999 bash dashboard/start.sh

set -e

cd "$(dirname "$0")/.."

pip3 install flask google-genai httpx pyyaml -q --break-system-packages 2>/dev/null || pip3 install flask google-genai httpx pyyaml -q

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Crucible Dashboard"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Get external IP
EXT_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_VM_IP")
echo "  Open on your phone: http://${EXT_IP}:8080"
echo "  PIN: ${CRUCIBLE_PIN:-1234}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 dashboard/api.py
