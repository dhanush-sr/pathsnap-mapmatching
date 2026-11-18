"""Start the PathSnap API on LAPTOP 1, reachable from other machines on the same network.

    python api/run_api.py                 # port 8000
    python api/run_api.py --port 9000
"""
import argparse
import os
import sys

import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    os.environ["PATHSNAP_PORT"] = str(args.port)
    uvicorn.run("api.app:app", host="0.0.0.0", port=args.port, log_level="info")
