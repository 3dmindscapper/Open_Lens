#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../Open_Lens_Core"

echo ""
echo " Starting Open Lens Web Translator..."
echo " The server URL will appear below."
echo " Open that URL on any device on your network."
echo ""

python3 web_app.py
