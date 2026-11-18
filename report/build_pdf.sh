#!/bin/sh
# Render report.html to PathSnap_Technical_Report.pdf with headless Chrome (macOS path; adjust on other systems).
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="PathSnap_Technical_Report.pdf" "file://$PWD/report.html"
