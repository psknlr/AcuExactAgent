"""Loopback-only demo server; no external services, filesystem browsing or upload endpoint."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

from .classification import classify_quote
from .pipeline import analyze, demo

MAX_BODY = 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    def response(self, status, data, content_type="application/json; charset=utf-8"):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.response(200, (Path(__file__).parent / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/demo":
            self.response(200, demo())
        elif self.path == "/api/health":
            self.response(200, {"status": "ok", "mode": "offline"})
        else:
            self.response(404, {"error": "Not found"})

    def do_POST(self):
        # Reject cross-origin browsers; CLI requests without Origin remain usable.
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}:
            self.response(403, {"error": "Cross-origin request denied"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                self.response(413, {"error": "Request body must be 1 byte to 1 MB"})
                return
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            if self.path == "/api/analyze":
                text = payload.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("请粘贴已确认属于目标文章的文本。")
                result = analyze([{"text": text, "page": 1, "method": "pasted_text"}],
                                 {"id": str(payload.get("article_id") or "pasted"), "title": str(payload.get("article_title") or "粘贴文章")})
                result["audit"]["boundary_status"] = "user_scoped_text"
                self.response(200, result)
            elif self.path == "/api/classify":
                quotes = payload.get("quotes")
                if not isinstance(quotes, list) or len(quotes) > 2000 or any(not isinstance(q, dict) or not isinstance(q.get("text"), str) for q in quotes):
                    raise ValueError("Expected up to 2000 quote objects with text")
                self.response(200, {"quotes": [classify_quote(q) for q in quotes]})
            else:
                self.response(404, {"error": "Not found"})
        except (ValueError, TypeError, KeyError) as exc:
            self.response(400, {"error": str(exc)})

    def log_message(self, fmt, *args):
        # Do not log source text.
        pass


def serve(port=8765, host="127.0.0.1"):
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"AcuExactAgent demo: http://127.0.0.1:{port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
