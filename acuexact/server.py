"""Local PDF workbench. Temporary documents and in-memory API credentials."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from .classification import classify_quote
from .pipeline import analyze, demo
from .workspace import Workspace, MAX_PDF_BYTES, available_providers, make_client

MAX_BODY = 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    def reject(self, status, message):
        # Drain small rejected POST bodies before closing. Otherwise Windows may
        # reset the TCP connection and hide the useful HTTP error from the UI.
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if self.command == "POST" and 0 < length <= MAX_BODY:
                self.connection.settimeout(2)
                self.rfile.read(length)
        except (ValueError, OSError):
            pass
        self.close_connection = True
        self.response(status, {"error": message})

    def response(self, status, data, content_type="application/json; charset=utf-8"):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data: blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def trusted_request(self):
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        if self.headers.get("Host", "").lower() not in hosts or (origin and origin not in {f"http://{h}" for h in hosts}):
            self.reject(403, "只允许从 localhost 或 127.0.0.1 同源访问。")
            return False
        return True

    def do_GET(self):
        if not self.trusted_request():
            return
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self.response(200, (Path(__file__).parent / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/demo":
            self.response(200, demo())
        elif path == "/api/health":
            self.response(200, {"status": "ok", "mode": "pdf_agent", "version": "0.2.0"})
        elif path == "/api/providers":
            self.response(200, available_providers())
        elif match := re.fullmatch(r"/api/documents/([a-f0-9]{32})/pages/(\d+)\.png", path):
            try:
                self.response(200, self.server.workspace.preview(match[1], int(match[2])), "image/png")
            except (KeyError, ValueError) as exc:
                self.response(400, {"error": str(exc)})
            except Exception:
                self.response(500, {"error": "无法渲染 PDF 预览。"})
        elif match := re.fullmatch(r"/api/jobs/([a-f0-9]{32})", path):
            try:
                self.response(200, self.server.workspace.get_job(match[1]))
            except KeyError as exc:
                self.response(404, {"error": str(exc)})
        else:
            self.response(404, {"error": "Not found"})

    def do_POST(self):
        if not self.trusted_request():
            return
        path = urlsplit(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            maximum = MAX_PDF_BYTES if path == "/api/uploads" else MAX_BODY
            if not 0 < length <= maximum:
                self.reject(413, f"请求内容须为 1–{maximum} bytes。")
                return
            expected_type = "application/pdf" if path == "/api/uploads" else "application/json"
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != expected_type:
                self.reject(415, f"Content-Type 必须为 {expected_type}。")
                return
            self.connection.settimeout(30)
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("请求内容不完整。")
            if path == "/api/uploads":
                self.response(201, self.server.workspace.upload(body, unquote(self.headers.get("X-Filename", "document.pdf"))))
                return
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            if path == "/api/jobs":
                self.response(202, self.server.workspace.start(payload))
            elif match := re.fullmatch(r"/api/jobs/([a-f0-9]{32})/cancel", path):
                self.response(200, self.server.workspace.cancel(match[1]))
            elif path == "/api/providers/test":
                client = make_client(payload.get("config"))
                client.generate_json("Return JSON with ok=true. This is a connection test.", {},
                                     {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}, schema_name="connection_test")
                self.response(200, {"ok": True, "message": "API 已连接，JSON 输出校验通过。", "calls": client.calls})
            elif path == "/api/analyze":
                text = payload.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("请粘贴已确认属于目标文章的文本。")
                result = analyze([{"text": text, "page": 1, "method": "pasted_text"}],
                                 {"id": str(payload.get("article_id") or "pasted"), "title": str(payload.get("article_title") or "粘贴文章")})
                result["audit"]["boundary_status"] = "user_scoped_text"
                self.response(200, result)
            elif path == "/api/classify":
                quotes = payload.get("quotes")
                if not isinstance(quotes, list) or len(quotes) > 2000 or any(not isinstance(q, dict) or not isinstance(q.get("text"), str) for q in quotes):
                    raise ValueError("Expected up to 2000 quote objects with text")
                self.response(200, {"quotes": [classify_quote(q) for q in quotes]})
            else:
                self.response(404, {"error": "Not found"})
        except (ValueError, TypeError, KeyError) as exc:
            self.response(400, {"error": str(exc)})
        except Exception:
            self.response(500, {"error": "请求处理失败，请检查文件及模型设置。"})

    def log_message(self, fmt, *args):
        # Do not log source text.
        pass


def create_server(port=8765, host="127.0.0.1", workspace=None):
    server = ThreadingHTTPServer((host, port), Handler)
    server.workspace = workspace or Workspace()
    return server


def serve(port=8765, host="127.0.0.1"):
    server = create_server(port, host)
    print(f"AcuExactAgent: http://127.0.0.1:{server.server_port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.workspace.close()
