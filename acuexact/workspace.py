"""Temporary PDF storage and bounded background agent jobs."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from threading import Event, RLock
from uuid import uuid4

from .pdf import file_sha256, open_pdf, text_quality_flags
from .providers import ModelClient, ProviderConfig, provider_options

MAX_PDF_BYTES = 40 * 1024 * 1024
MAX_DOCUMENT_PAGES = 200


def parse_pages(value, total):
    if value is None or value == "":
        return None
    if not isinstance(value, str) or len(value) > 1000:
        raise ValueError("页码应填写 1-3,5 这样的范围。")
    pages = []
    for part in value.replace("，", ",").split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", part)
        if not match:
            raise ValueError("页码应填写 1-3,5 这样的范围。")
        start = int(match[1]); end = int(match[2] or start)
        if not 1 <= start <= end <= total:
            raise ValueError(f"页码必须在 1–{total} 内，起始页不能大于结束页。")
        for page in range(start, end + 1):
            if page not in pages:
                pages.append(page)
    return pages


def make_client(value):
    if not isinstance(value, dict):
        raise ValueError("请配置模型供应商与模型名称。")
    allowed = {"provider", "model", "api_key", "base_url", "timeout", "max_tokens", "vision"}
    if set(value) - allowed:
        raise ValueError("模型配置包含不支持的字段。")
    try:
        client = ModelClient(ProviderConfig(**value))
        if client.config.provider == "minimax" and client.config.vision:
            raise ValueError("MiniMax 文字适配器不支持图片；请关闭主模型视觉并配置独立 GPT/Claude 视觉模型。")
        return client
    except TypeError:
        raise ValueError("模型配置不完整或字段类型不正确。") from None


def available_providers():
    return {"providers": [dict(p, env_key=p["env"], configured=bool(os.environ.get(p["env"], "")) if p["env"] else False)
                           for p in provider_options()], "max_pdf_bytes": MAX_PDF_BYTES}


class Workspace:
    def __init__(self, runner=None):
        self._temp = TemporaryDirectory(prefix="acuexact-")
        self.root = Path(self._temp.name)
        self.documents, self.jobs, self._cancel = {}, {}, {}
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="acuexact")
        self.runner = runner

    def upload(self, content, filename):
        if not content or len(content) > MAX_PDF_BYTES:
            raise ValueError("PDF 文件须为 1 byte–40 MB。")
        if not content[:1024].lstrip().startswith(b"%PDF-"):
            raise ValueError("文件不是有效的 PDF。")
        with self._lock:
            if len(self.documents) >= 10:
                raise ValueError("本次服务最多保留 10 份 PDF，请重启服务清理临时文档。")
            doc_id = uuid4().hex
            path = self.root / f"{doc_id}.pdf"
            path.write_bytes(content)
            try:
                from pypdf import PdfReader
                reader = PdfReader(path)
                if reader.is_encrypted or not 1 <= len(reader.pages) <= MAX_DOCUMENT_PAGES:
                    raise ValueError("PDF 已加密或页数超限。")
                infos = []
                with open_pdf(path) as pdf:
                    for n, page in enumerate(pdf.pages, 1):
                        txt = page.extract_text() or ""
                        infos.append({"page": n, "width": float(page.width), "height": float(page.height),
                                      "char_count": len(txt), "quality_flags": text_quality_flags(txt)})
                name = str(filename).replace("\\", "/").split("/")[-1][:200] or "document.pdf"
                doc = {"document_id": doc_id, "filename": name, "pages": len(infos),
                       "sha256": file_sha256(path), "page_info": infos,
                       "preview_url": f"/api/documents/{doc_id}/pages/1.png"}
                self.documents[doc_id] = doc
                return deepcopy(doc)
            except Exception:
                path.unlink(missing_ok=True)
                raise ValueError("无法读取 PDF；请确认文件未损坏、未加密且不超过 200 页。") from None

    def document(self, doc_id):
        with self._lock:
            if doc_id not in self.documents:
                raise KeyError("文档不存在或服务已重启，请重新导入。")
            return deepcopy(self.documents[doc_id]), self.root / f"{doc_id}.pdf"

    def preview(self, doc_id, page):
        doc, path = self.document(doc_id)
        if not 1 <= page <= doc["pages"]:
            raise ValueError("预览页码超出文档范围。")
        with open_pdf(path) as pdf:
            p = pdf.pages[page - 1]
            png = p.to_image(resolution=min(110, 1800 * 72 / max(p.width, p.height))).original
            out = BytesIO(); png.save(out, format="PNG")
            return out.getvalue()

    def start(self, payload):
        doc, path = self.document(payload.get("document_id"))
        prompt, title = payload.get("prompt", ""), payload.get("target_title", "")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 20000:
            raise ValueError("请输入任务 prompt（最多 20,000 字符）。")
        if not isinstance(title, str) or len(title) > 500:
            raise ValueError("文章标题过长。")
        pages = parse_pages(payload.get("pages"), doc["pages"])
        limits = {"max_pages": payload.get("max_pages", 30), "max_calls": payload.get("max_calls", 80)}
        for key in limits:
            if type(limits[key]) is not int or not 1 <= limits[key] <= 200:
                raise ValueError(f"{key} 必须为 1–200 的整数。")
        if len(pages or doc["page_info"]) > limits["max_pages"]:
            raise ValueError("所选页数超过本次最大页数，请缩小页码范围或提高最大页数。")
        client = make_client(payload.get("config"))
        vision_client = make_client(payload["vision_config"]) if payload.get("vision_config") else None
        if vision_client and not vision_client.config.vision:
            raise ValueError("辅助视觉模型须开启视觉能力。")
        client._key()
        if vision_client:
            vision_client._key()
        with self._lock:
            if sum(j["status"] in {"queued", "running"} for j in self.jobs.values()) >= 4:
                raise ValueError("已有 4 个任务等待或执行，请稍后再试。")
            if len(self.jobs) >= 50:
                oldest = next((k for k, j in self.jobs.items() if j["status"] not in {"queued", "running"}), None)
                if oldest:
                    del self.jobs[oldest]; self._cancel.pop(oldest, None)
            jid = uuid4().hex
            self.jobs[jid] = {"job_id": jid, "status": "queued", "events": [], "result": None, "error": None}
            self._cancel[jid] = Event()
            self._executor.submit(self._run, jid, path, prompt, title, pages, limits, client, vision_client)
        return {"job_id": jid}

    def _run(self, jid, path, prompt, title, pages, limits, client, vision_client):
        cancel = self._cancel[jid]
        def event(value):
            item = {k: value[k] for k in ("stage", "message", "progress") if k in value}
            item["time"] = datetime.now(timezone.utc).isoformat()
            with self._lock:
                self.jobs[jid]["events"] = (self.jobs[jid]["events"] + [item])[-200:]
        with self._lock:
            self.jobs[jid]["status"] = "running"
        try:
            if cancel.is_set():
                return
            if self.runner is None:
                from .agent import run_pdf_agent
                runner = run_pdf_agent
            else:
                runner = self.runner
            result = runner(path, prompt, client, target_title=title, page_numbers=pages,
                            vision_client=vision_client, on_event=event, cancelled=cancel.is_set, **limits)
            with self._lock:
                self.jobs[jid]["result"] = result
                self.jobs[jid]["status"] = "needs_scope" if result.get("status") == "needs_scope" else "completed"
        except Exception as exc:
            message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else f"任务失败（{type(exc).__name__}），请检查 PDF 与模型配置。"
            for c in (client, vision_client):
                if c:
                    try:
                        key = c._key()
                    except ValueError:
                        key = ""
                    if key:
                        message = message.replace(key, "[REDACTED]")
            with self._lock:
                self.jobs[jid].update(status="failed", error=message[:1000])
        finally:
            if cancel.is_set():
                with self._lock:
                    self.jobs[jid]["status"] = "cancelled"

    def get_job(self, jid):
        with self._lock:
            if jid not in self.jobs:
                raise KeyError("任务不存在或已清理。")
            return deepcopy(self.jobs[jid])

    def cancel(self, jid):
        self.get_job(jid)
        self._cancel[jid].set()
        return {"ok": True, "message": "已请求取消；当前 API 请求返回后停止后续步骤。"}

    def close(self):
        for cancel in self._cancel.values():
            cancel.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._temp.cleanup()
