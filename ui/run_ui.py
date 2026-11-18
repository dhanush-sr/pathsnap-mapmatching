"""Serve the PathSnap evaluator UI on LAPTOP 2.

    python ui/run_ui.py --api http://<laptop-1-ip>:8000
    python ui/run_ui.py --api http://<laptop-1-ip>:8000 --port 5500

Opens the UI in the browser with the Laptop 1 API address filled in. Only the Python
standard library is needed on this laptop.
"""
import argparse
import functools
import http.server
import os
import socket
import urllib.parse
import webbrowser


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000", help="address of the API on Laptop 1")
    ap.add_argument("--port", type=int, default=5500)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=here)
    url = f"http://localhost:{args.port}/?api={urllib.parse.quote(args.api, safe=':/')}"
    bar = "=" * 64
    print(bar)
    print("PATHSNAP UI  |  LAPTOP 2 (evaluator interface)")
    print(f"API (Laptop 1) : {args.api}")
    print(f"open           : {url}")
    print(f"from elsewhere : http://{lan_ip()}:{args.port}/?api={args.api}")
    print(bar, flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    http.server.ThreadingHTTPServer(("0.0.0.0", args.port), handler).serve_forever()
