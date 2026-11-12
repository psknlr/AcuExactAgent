import tempfile
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

from acuexact.agent_batch import CheckpointClient, _run_one, pipeline_complete, run_batch, select_jobs
from acuexact.providers import ModelClient, ProviderConfig
from test_agent import FakeClient, FakePage, plan


class AgentBatchTests(unittest.TestCase):
    def test_natural_selection_prefix_and_one_based_start(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "papers.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for name in ("A/10 paper.pdf", "A/2 paper.pdf", "A/1 paper.PDF", "B/1 paper.pdf", "A/note.txt"):
                    archive.writestr(name, b"fixture")
            self.assertEqual([r["member"] for r in select_jobs(path, 2, start=2, prefix="A/")],
                             ["A/2 paper.pdf", "A/10 paper.pdf"])
            self.assertFalse((Path(directory) / "A").exists())

    def test_directory_selection_ignores_non_pdf_and_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "folder.pdf").mkdir()
            for name in ("10 paper.pdf", "2 paper.pdf", "1 paper.txt"):
                (root / name).write_bytes(b"fixture")
            self.assertEqual([r["member"] for r in select_jobs(root, 1)], ["2 paper.pdf"])

    def test_review_flags_do_not_hide_incomplete_pipeline(self):
        result = {"status": "needs_review", "audit": {"stages": ["review_quotes"]},
                  "warnings": [{"code": "automatic_article_boundary_requires_review"}],
                  "quotes": [{"relevance": {"method": "llm_agent"}}]}
        self.assertTrue(pipeline_complete(result))
        for warning in ("agent_incomplete", "quote_review_failed", "vision_required",
                        "unclosed_quote", "unmatched_closing_quote"):
            result["warnings"] = [{"code": warning}]
            self.assertFalse(pipeline_complete(result))
        result["warnings"] = []
        result["quotes"][0]["relevance"]["method"] = "unclassified"
        self.assertFalse(pipeline_complete(result))

    def test_empty_results_require_the_review_stage(self):
        self.assertFalse(pipeline_complete({"status": "needs_review", "quotes": []}))
        self.assertTrue(pipeline_complete({"status": "needs_review", "quotes": [],
                                           "audit": {"stages": ["review_quotes"]}}))

    def test_checkpoint_reuses_success_and_invalidates_changed_evidence(self):
        schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
        with tempfile.TemporaryDirectory() as directory:
            client = CheckpointClient(ProviderConfig("minimax", "MiniMax-M3.1-Flash", "fixture-key"), directory)
            with patch.object(ModelClient, "generate_json", return_value={"text": "原字"}) as generated:
                first = client.generate_json("read", {"page": 1}, schema)
                self.assertEqual(first, client.generate_json("read", {"page": 1}, schema))
                self.assertEqual(generated.call_count, 1)
                self.assertEqual(client.calls[-1]["status"], "cache_hit")
                client.generate_json("read", {"page": 2}, schema)
                self.assertEqual(generated.call_count, 2)
            for file in Path(directory).glob("*.json"):
                self.assertNotIn("fixture-key", file.read_text(encoding="utf-8"))

    def test_stricter_transcript_schema_revalidates_prior_checkpoint(self):
        old = {"type": "object", "properties": {"text": {"type": "string"},
               "uncertainties": {"type": "array"}, "confidence": {"type": "number"}}}
        strict = json.loads(json.dumps(old))
        strict["properties"]["text"]["minLength"] = 1
        strict["properties"]["confidence"].update(minimum=0, maximum=1)
        with tempfile.TemporaryDirectory() as directory:
            client = CheckpointClient(ProviderConfig("minimax", "MiniMax-M3.1-Flash"), directory)
            with patch.object(ModelClient, "generate_json", return_value={"text": "原文", "uncertainties": [], "confidence": .9}) as generated:
                client.generate_json("read", {}, old)
                client.generate_json("read", {}, strict)
                self.assertEqual(generated.call_count, 1)
                self.assertEqual(client.calls[-1]["status"], "cache_hit")

    def test_switching_provider_does_not_relabel_or_reuse_old_checkpoint(self):
        schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
        with tempfile.TemporaryDirectory() as directory:
            old = CheckpointClient(ProviderConfig("minimax", "MiniMax-M3.1-Flash"), directory)
            new = CheckpointClient(ProviderConfig("openai", "gpt-4o-mini"), directory)
            with patch.object(ModelClient, "generate_json", side_effect=[{"text": "旧模型"}, {"text": "新模型"}]) as generated:
                self.assertEqual(old.generate_json("read", {}, schema), {"text": "旧模型"})
                self.assertEqual(new.generate_json("read", {}, schema), {"text": "新模型"})
                self.assertEqual(new.generate_json("read", {}, schema), {"text": "新模型"})
                self.assertEqual(generated.call_count, 2)
                self.assertEqual(new.calls[-1]["model"], "gpt-4o-mini")
                self.assertEqual(old.generate_json("read", {}, schema), {"text": "旧模型"})

    def test_gpt4o_mini_batch_uses_small_parts_and_preserves_all_original_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "source.pdf"; pdf.write_bytes(b"same-source")
            pages = [FakePage("肩痛研究" + "正常正文内容。" * 20) for _ in range(5)]

            @contextmanager
            def opened(_):
                yield SimpleNamespace(pages=pages)

            client = FakeClient({"plan_article": lambda payload: plan(pages=tuple(p["page"] for p in payload["pages"]))}, vision=True)
            client.config = ProviderConfig("openai", "gpt-4o-mini", vision=True)
            job = {"id": "article-1", "title": "肩痛研究", "pdf": str(pdf), "member": "folder/1 原始中文名.pdf"}
            settings = {"out": directory, "prompt": "extract", "max_pages": 20, "max_calls": 80,
                        "long_documents": True, "config": {"provider": "openai", "model": "gpt-4o-mini", "vision": True}}
            with patch("acuexact.pdf.open_pdf", opened), patch("acuexact.agent.open_pdf", opened), patch("acuexact.agent_batch.CheckpointClient", return_value=client):
                record = _run_one(job, settings)
            self.assertTrue(record["pipeline_complete"])
            result = json.loads((root / "article-1/result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["source_pdf_filename"], "1 原始中文名.pdf")
            self.assertEqual(result["audit"]["selected_pages"], [1, 2, 3, 4, 5])
            self.assertEqual([p["pages"] for p in result["audit"]["parts"]], [[1, 2], [3, 4], [5]])

    def test_resume_skips_completed_paper_and_refuses_changed_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, output = root / "papers", root / "results"
            source.mkdir(); output.mkdir()
            filename = "1 原始中文文件名.pdf"
            (source / filename).write_bytes(b"fixture")
            config = {"provider": "minimax", "model": "MiniMax-M3.1-Flash"}
            job = {"id": "article-1", "index": 1, "member": filename, "title": "不同的文章标题",
                   "pdf": str(output / "inputs/article-0001-hash.pdf")}
            fingerprint = {"source": str(source), "selection": select_jobs(source, 350),
                           "prompt": "extract", "config": config, "max_pages": 20, "max_calls": 80}
            (output / "manifest.json").write_text(json.dumps({"fingerprint": fingerprint, "jobs": [job]}))
            article = output / job["id"]; article.mkdir()
            (article / "status.json").write_text(json.dumps(dict(job, state="done", pipeline_complete=True)))
            (article / "result.json").write_text(json.dumps({"quotes": [{"id": "q1", "text": "原文"}]}))
            with patch.object(ModelClient, "_key", return_value="fixture"), patch.object(ModelClient, "generate_json") as remote:
                self.assertEqual(run_batch(source, output, "extract", config, workers=1), 0)
                remote.assert_not_called()
                progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
                self.assertEqual(progress["pipeline_completed"], 1)
                self.assertEqual(progress["state"], "finished")
                for tsv in (output / "quotes.tsv", article / "quotes.tsv"):
                    with tsv.open(encoding="utf-8-sig", newline="") as stream:
                        rows = list(csv.DictReader(stream, delimiter="\t"))
                    self.assertEqual(rows[0]["文件名（中文）"], filename)
                    self.assertEqual(rows[0]["古籍条文"], "原文")
                result = json.loads((article / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(result["quotes"][0]["source_pdf_filename"], filename)
                aggregate = json.loads((output / "results.json").read_text(encoding="utf-8"))
                self.assertEqual(aggregate["quotes"], result["quotes"])
                with self.assertRaisesRegex(ValueError, "differs"):
                    run_batch(source, output, "different research", config, workers=1)

    def test_short_batch_reuses_clear_text_and_still_transcribes_damaged_text(self):
        for damaged in (False, True):
            with self.subTest(damaged=damaged), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pdf = root / "source.pdf"; pdf.write_bytes(b"same-source")
                text = '(cid:123)' if damaged else '肩痛研究的完整正文引文“肩臂疼痛，气血不通。”'

                @contextmanager
                def opened(_):
                    yield SimpleNamespace(pages=[FakePage(text)])

                client = FakeClient({"region_transcription": {
                    "text": '肩痛研究的完整正文引文“肩臂疼痛，气血不通。”',
                    "uncertainties": [], "confidence": .9}}, vision=True)
                client.config = ProviderConfig("openai", "gpt-4o-mini", vision=True)
                job = {"id": "article-1", "title": "肩痛研究", "pdf": str(pdf), "member": "1 原始中文名.pdf"}
                settings = {"out": directory, "prompt": "extract", "max_pages": 20, "max_calls": 80,
                            "config": {"provider": "openai", "model": "gpt-4o-mini", "vision": True}}
                with patch("acuexact.agent.open_pdf", opened), patch("acuexact.agent_batch.CheckpointClient", return_value=client):
                    record = _run_one(job, settings)
                self.assertTrue(record["pipeline_complete"])
                self.assertEqual("region_transcription" in [r["stage"] for r in client.requests], damaged)
                self.assertTrue(client.requests[0]["images"])
                result = json.loads((root / "article-1/result.json").read_text())
                self.assertEqual(result["quotes"][0]["text"], "肩臂疼痛，气血不通。")
                self.assertEqual(result["segments"][0]["method"], "vision_transcription" if damaged else "pdf_text")

    def test_retry_archives_prior_evidence_before_replacing_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            article = root / "article-1"; article.mkdir()
            old = {"status": "needs_review", "quotes": [{"text": "先前的原文证据"}], "audit": {}}
            (article / "result.json").write_text(json.dumps(old))
            job = {"id": "article-1", "title": "paper", "pdf": "fixture.pdf", "member": "folder/1 中文原名.pdf"}
            settings = {"out": directory, "prompt": "extract", "max_pages": 20, "max_calls": 80,
                        "config": {"provider": "minimax", "model": "MiniMax-M3.1-Flash"}}
            new = {"status": "needs_review", "quotes": [], "warnings": [], "audit": {"stages": ["review_quotes"]}}
            with patch("acuexact.agent.run_pdf_agent", return_value=new):
                self.assertEqual(_run_one(job, settings)["state"], "done")
            prior = next(article.glob("attempts/*/result.json"))
            self.assertEqual(json.loads(prior.read_text(encoding="utf-8")), old)
            self.assertEqual(new["source_pdf_filename"], "1 中文原名.pdf")
            self.assertEqual(json.loads((article / "result.json").read_text(encoding="utf-8")), new)
