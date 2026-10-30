"""Offline PDF/job/HTTP integration checks; all model API calls are mocked."""
from http.client import HTTPResponse
from io import BytesIO
import json
import os
import socket
from threading import Event, Thread
import time
import unittest
from unittest.mock import patch

from pypdf import PdfWriter

from acuexact.classification import DIMENSIONS
from acuexact.providers import ProviderError
from acuexact.server import create_server
from acuexact.workspace import Workspace, available_providers, make_client, parse_pages


KEY = "workspace-secret-not-in-responses"
TEXT = "目标研究文章说明并引用古籍：‘普通内容’。“肩痛不可举，取肩髃。”又云“春生夏长”。"


def blank_pdf(pages=1, encrypted=False):
    output = BytesIO()
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=300, height=400)
    if encrypted:
        writer.encrypt("not-for-agent")
    writer.write(output)
    return output.getvalue()


def payload(doc, **overrides):
    value = {"document_id": doc["document_id"], "prompt": "提取所有古籍引文并分类", "config": {"provider": "openai", "model": "gpt-4.1", "api_key": KEY}}
    value.update(overrides)
    return value


def wait_job(workspace, jid, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        job = workspace.get_job(jid)
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(.01)
    raise AssertionError("background job did not finish")


def fake_model(self, system, value, schema, *, images=None, schema_name="agent_result"):
    self.calls.append({"provider": self.config.provider, "model": self.config.model, "request_id": "fake-" + schema_name, "usage": {}, "status": "completed"})
    if schema_name in {"page_transcription", "region_transcription"}:
        if not images or not images[0].startswith("data:image/png;base64,"):
            raise AssertionError("expected rendered PDF image")
        return {"text": TEXT, "uncertainties": [], "confidence": .9}
    if schema_name == "plan_article":
        return {"article_candidates": [{"id": "a1", "title": "目标研究文章", "pages": [1]}], "selected_article_id": "a1", "multiple_articles": False, "needs_scope": False, "confidence": .95, "reason": "一页单篇文章", "classification_dimensions": list(DIMENSIONS), "regions": [{"page": 1, "bbox": [0, 0, 300, 400], "article_id": "a1", "order": 1}]}
    if schema_name in {"review_quotes", "repair_quotes"}:
        return {"missing_quotes": [], "issues": [], "all_quotes_checked": True}
    if schema_name in {"classify_quotes", "repair_classification"}:
        return {"results": [{"id": q["id"], "label": "强相关" if "肩" in q["text"] else "弱相关", "dimensions": [DIMENSIONS[2]] if "肩" in q["text"] else [], "reason": "原文涉及肩痛取穴" if "肩" in q["text"] else "未直接涉及肩痛研究主题", "needs_review": True} for q in value["quotes"]]}
    if schema_name == "connection_test":
        return {"ok": True}
    raise AssertionError("unexpected model stage: " + schema_name)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Workspace(runner=lambda *args, **kwargs: {"status": "completed", "quotes": []})

    def tearDown(self):
        self.workspace.close()

    def test_upload_stays_temporary_sanitizes_name_and_copies_metadata(self):
        data = blank_pdf(2)
        doc = self.workspace.upload(data, r"C:\private\../../example.pdf")
        self.assertEqual(doc["filename"], "example.pdf")
        self.assertEqual(doc["pages"], 2)
        self.assertEqual(doc["page_info"][0]["char_count"], 0)
        self.assertEqual(len(doc["sha256"]), 64)
        doc["page_info"][0]["width"] = 99
        stored, path = self.workspace.document(doc["document_id"])
        self.assertEqual(stored["page_info"][0]["width"], 300)
        self.assertEqual(path.parent, self.workspace.root)
        self.assertEqual(path.read_bytes(), data)

    def test_bad_and_encrypted_pdf_do_not_leave_partial_files(self):
        for data in (b"text", b"%PDF- broken file", blank_pdf(encrypted=True), blank_pdf(0)):
            with self.subTest(data=data[:20]), self.assertRaises(ValueError):
                self.workspace.upload(data, "bad.pdf")
        self.assertEqual(self.workspace.documents, {})
        self.assertEqual(list(self.workspace.root.iterdir()), [])

    def test_document_and_preview_ranges_are_checked(self):
        doc = self.workspace.upload(blank_pdf(2), "test.pdf")
        with self.assertRaises(KeyError):
            self.workspace.document("../not-a-document")
        for page in (0, -1, 3):
            with self.subTest(page=page), self.assertRaises(ValueError):
                self.workspace.preview(doc["document_id"], page)
        data = self.workspace.preview(doc["document_id"], 2)
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_page_ranges_normalize_and_deduplicate(self):
        self.assertIsNone(parse_pages(None, 5))
        self.assertIsNone(parse_pages("", 5))
        self.assertEqual(parse_pages("1-3，2, 5", 5), [1, 2, 3, 5])
        for value in ("0", "4-2", "6", "1,", "one", "1.5", [1], "9" * 1001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_pages(value, 5)

    def test_start_checks_prompt_pages_and_numeric_budgets(self):
        doc = self.workspace.upload(blank_pdf(2), "test.pdf")
        for updates in ({"prompt": " "}, {"prompt": 42}, {"prompt": "x" * 20001}, {"target_title": "x" * 501}, {"pages": "3"}, {"max_pages": 1}, {"max_calls": True}, {"max_calls": 0}, {"max_calls": 201}, {"max_pages": 1.5}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.workspace.start(payload(doc, **updates))
        self.assertEqual(self.workspace.jobs, {})

    def test_missing_api_key_fails_before_queueing(self):
        doc = self.workspace.upload(blank_pdf(), "test.pdf")
        with patch.dict(os.environ, {}, clear=True), patch.object(self.workspace._executor, "submit") as submit:
            with self.assertRaisesRegex(ProviderError, "OPENAI_API_KEY"):
                self.workspace.start(payload(doc, config={"provider": "openai", "model": "gpt-4.1"}))
        submit.assert_not_called()
        self.assertEqual(self.workspace.jobs, {})

    def test_invalid_config_and_auxiliary_vision_flag_rejected(self):
        for config in (None, [], {}, {"provider": "openai", "model": "gpt-4.1", "unknown": KEY}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                make_client(config)
        doc = self.workspace.upload(blank_pdf(), "test.pdf")
        with self.assertRaisesRegex(ValueError, "视觉能力"):
            self.workspace.start(payload(doc, vision_config={"provider": "anthropic", "model": "claude-sonnet-4-6", "api_key": KEY, "vision": False}))

    def test_async_job_progress_result_and_copy_are_retained(self):
        seen = {}
        def runner(path, prompt, client, **kwargs):
            seen.update(path=path, prompt=prompt, pages=kwargs["page_numbers"])
            kwargs["on_event"]({"stage": "review", "message": "核对完成", "progress": .8, "private": "not published"})
            return {"status": "needs_review", "quotes": [{"text": "春生夏长"}]}
        self.workspace.runner = runner
        doc = self.workspace.upload(blank_pdf(2), "test.pdf")
        jid = self.workspace.start(payload(doc, pages="2"))["job_id"]
        job = wait_job(self.workspace, jid)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["result"]["status"], "needs_review")
        self.assertEqual(seen["pages"], [2])
        self.assertNotIn("private", job["events"][0])
        self.assertNotIn(KEY, json.dumps(job))
        job["result"]["quotes"].clear()
        self.assertEqual(len(self.workspace.get_job(jid)["result"]["quotes"]), 1)

    def test_scope_result_has_distinct_job_state(self):
        self.workspace.runner = lambda *args, **kwargs: {"status": "needs_scope", "article_candidates": [{"id": "a1"}]}
        doc = self.workspace.upload(blank_pdf(), "test.pdf")
        job = wait_job(self.workspace, self.workspace.start(payload(doc))["job_id"])
        self.assertEqual(job["status"], "needs_scope")
        self.assertEqual(job["result"]["article_candidates"], [{"id": "a1"}])

    def test_cancel_stops_next_steps_and_preserves_partial_result(self):
        entered, release = Event(), Event()
        def runner(*args, **kwargs):
            entered.set()
            release.wait(3)
            return {"status": "cancelled" if kwargs["cancelled"]() else "completed", "quotes": [{"text": "已提取原文"}]}
        self.workspace.runner = runner
        doc = self.workspace.upload(blank_pdf(), "test.pdf")
        jid = self.workspace.start(payload(doc))["job_id"]
        try:
            self.assertTrue(entered.wait(2))
            self.assertTrue(self.workspace.cancel(jid)["ok"])
        finally:
            release.set()
        job = wait_job(self.workspace, jid)
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(job["result"]["quotes"][0]["text"], "已提取原文")
        with self.assertRaises(KeyError):
            self.workspace.cancel("unknown")

    def test_job_error_redacts_primary_and_auxiliary_keys(self):
        second_key = "auxiliary-secret-key"
        def runner(*args, **kwargs):
            raise RuntimeError("failure " + KEY + " " + second_key)
        self.workspace.runner = runner
        doc = self.workspace.upload(blank_pdf(), "test.pdf")
        jid = self.workspace.start(payload(doc, vision_config={"provider": "anthropic", "model": "claude-sonnet-4-6", "api_key": second_key, "vision": True}))["job_id"]
        job = wait_job(self.workspace, jid)
        self.assertEqual(job["status"], "failed")
        self.assertNotIn(KEY, json.dumps(job))
        self.assertNotIn(second_key, json.dumps(job))

    def test_model_config_metadata_never_returns_environment_secret(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": KEY}, clear=True):
            options = available_providers()
        self.assertTrue(next(p for p in options["providers"] if p["id"] == "openai")["configured"])
        self.assertNotIn(KEY, json.dumps(options))

    def test_queue_capacity_is_bounded_and_queued_cancellation_is_retained(self):
        release = Event()
        def runner(*args, **kwargs):
            release.wait(3)
            return {"status": "completed", "quotes": []}
        self.workspace.runner = runner
        doc = self.workspace.upload(blank_pdf(), "test.pdf")
        try:
            ids = [self.workspace.start(payload(doc))["job_id"] for _ in range(4)]
            with self.assertRaisesRegex(ValueError, "4"):
                self.workspace.start(payload(doc))
            self.workspace.cancel(ids[-1])
        finally:
            release.set()
        self.assertEqual(wait_job(self.workspace, ids[-1])["status"], "cancelled")

    def test_close_removes_uploaded_files(self):
        doc = self.workspace.upload(blank_pdf(), "test.pdf")
        _, path = self.workspace.document(doc["document_id"])
        self.assertTrue(path.exists())
        self.workspace.close()
        self.assertFalse(path.exists())
        self.assertFalse(self.workspace.root.exists())

    def test_real_blank_pdf_agent_pipeline_with_fake_model(self):
        self.workspace.runner = None
        doc = self.workspace.upload(blank_pdf(), "scan.pdf")
        config = {"provider": "openai", "model": "gpt-4.1", "api_key": KEY, "vision": True}
        with patch("acuexact.providers.ModelClient.generate_json", new=fake_model):
            jid = self.workspace.start(payload(doc, config=config))["job_id"]
            job = wait_job(self.workspace, jid)
        self.assertEqual(job["status"], "completed", job.get("error"))
        result = job["result"]
        texts = [q["text"] for q in result["quotes"]]
        self.assertIn("肩痛不可举，取肩髃。", texts)
        self.assertIn("春生夏长", texts)
        self.assertTrue(all(q["source_explicit"]["book"] is None for q in result["quotes"]))
        shoulder = next(q for q in result["quotes"] if "肩" in q["text"])
        self.assertEqual(shoulder["relevance"]["label"], "强相关")
        self.assertTrue(shoulder["relevance"]["reason"])
        self.assertNotIn(KEY, json.dumps(job, ensure_ascii=False))
        self.assertGreaterEqual(len(result["audit"]["model_calls"]), 4)


class HTTPWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = Workspace(runner=lambda *args, **kwargs: {"status": "completed", "quotes": [{"text": "原文"}]})
        cls.server = create_server(0, workspace=cls.workspace)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(3)
        cls.server.server_close()
        cls.workspace.close()

    def request(self, method, path, body=None, headers=None):
        # Send headers/body in one TCP write. Separate writes can race a deliberate
        # early 403/415 close on Windows, obscuring the actual HTTP response.
        body = body or b""
        merged = {"Host": f"127.0.0.1:{self.server.server_port}", "Connection": "close", "Content-Length": str(len(body)), **(headers or {})}
        request = (f"{method} {path} HTTP/1.1\r\n" + "".join(f"{key}: {value}\r\n" for key, value in merged.items()) + "\r\n").encode("ascii") + body
        with socket.create_connection(("127.0.0.1", self.server.server_port), timeout=5) as connection:
            connection.sendall(request)
            response = HTTPResponse(connection)
            response.begin()
            return response.status, dict(response.getheaders()), response.read()

    def post_json(self, path, data, headers=None):
        return self.request("POST", path, json.dumps(data).encode(), {"Content-Type": "application/json", **(headers or {})})

    def test_health_security_headers_and_provider_metadata(self):
        status, headers, body = self.request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["mode"], "pdf_agent")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        with patch.dict(os.environ, {"OPENAI_API_KEY": KEY}):
            status, _, body = self.request("GET", "/api/providers")
        self.assertEqual(status, 200)
        self.assertNotIn(KEY.encode(), body)

    def test_foreign_host_and_origin_are_rejected(self):
        for headers in ({"Host": "attacker.example"}, {"Origin": "https://attacker.example"}, {"Origin": "null"}, {"Host": f"127.0.0.1.attacker.example:{self.server.server_port}"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request("GET", "/api/health", headers=headers)[0], 403)
                self.assertEqual(self.post_json("/api/jobs", {}, headers=headers)[0], 403)

    def test_local_origin_allowed(self):
        status, _, _ = self.request("GET", "/api/health", headers={"Origin": f"http://127.0.0.1:{self.server.server_port}"})
        self.assertEqual(status, 200)

    def test_content_type_and_size_rejected_before_parsing(self):
        self.assertEqual(self.request("POST", "/api/jobs", b"{}", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/uploads", blank_pdf(), {"Content-Type": "application/octet-stream"})[0], 415)
        self.assertEqual(self.request("POST", "/api/jobs", b"", {"Content-Type": "application/json"})[0], 413)
        self.assertEqual(self.request("POST", "/api/jobs", b"", {"Content-Type": "application/json", "Content-Length": "1048577"})[0], 413)

    def test_invalid_json_and_unknown_routes(self):
        self.assertEqual(self.request("POST", "/api/jobs", b"{", {"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.post_json("/api/jobs", [1, 2])[0], 400)
        self.assertEqual(self.request("GET", "/not-found")[0], 404)
        self.assertEqual(self.request("GET", "/api/jobs/" + "0" * 32)[0], 404)

    def test_upload_preview_and_background_job_over_http(self):
        status, _, raw = self.request("POST", "/api/uploads", blank_pdf(), {"Content-Type": "application/pdf", "X-Filename": "test%20scan.pdf"})
        self.assertEqual(status, 201)
        doc = json.loads(raw)
        self.assertEqual(doc["filename"], "test scan.pdf")
        status, headers, data = self.request("GET", doc["preview_url"])
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/png")
        self.assertTrue(data.startswith(b"\x89PNG"))
        status, _, raw = self.post_json("/api/jobs", payload(doc))
        self.assertEqual(status, 202)
        jid = json.loads(raw)["job_id"]
        wait_job(self.workspace, jid)
        status, _, raw = self.request("GET", "/api/jobs/" + jid)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["status"], "completed")
        self.assertNotIn(KEY.encode(), raw)
        self.assertEqual(self.request("GET", f"/api/documents/{doc['document_id']}/pages/0.png")[0], 400)

    def test_model_connection_route_uses_fake_adapter_and_redacts_key(self):
        config = {"provider": "openai", "model": "gpt-4.1", "api_key": KEY}
        with patch("acuexact.providers.ModelClient.generate_json", new=fake_model):
            status, _, body = self.post_json("/api/providers/test", {"config": config})
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["ok"])
        self.assertNotIn(KEY.encode(), body)


if __name__ == "__main__":
    unittest.main()
