import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

from acuexact.agent_batch import CheckpointClient, pipeline_complete, run_batch, select_jobs
from acuexact.providers import ModelClient, ProviderConfig


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
        for warning in ("agent_incomplete", "quote_review_failed", "vision_required"):
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
                self.assertNotIn("fixture-key", file.read_text())

    def test_resume_skips_completed_paper_and_refuses_changed_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, output = root / "papers", root / "results"
            source.mkdir(); output.mkdir()
            (source / "1 paper.pdf").write_bytes(b"fixture")
            config = {"provider": "minimax", "model": "MiniMax-M3.1-Flash"}
            job = {"id": "article-1", "index": 1, "member": "1 paper.pdf", "title": "paper",
                   "pdf": str(source / "1 paper.pdf")}
            fingerprint = {"source": str(source), "selection": select_jobs(source, 350),
                           "prompt": "extract", "config": config, "max_pages": 20, "max_calls": 80}
            (output / "manifest.json").write_text(json.dumps({"fingerprint": fingerprint, "jobs": [job]}))
            article = output / job["id"]; article.mkdir()
            (article / "status.json").write_text(json.dumps(dict(job, state="done", pipeline_complete=True)))
            (article / "result.json").write_text(json.dumps({"quotes": []}))
            with patch.object(ModelClient, "_key", return_value="fixture"), patch.object(ModelClient, "generate_json") as remote:
                self.assertEqual(run_batch(source, output, "extract", config, workers=1), 0)
                remote.assert_not_called()
                progress = json.loads((output / "progress.json").read_text())
                self.assertEqual(progress["pipeline_completed"], 1)
                self.assertEqual(progress["state"], "finished")
                with self.assertRaisesRegex(ValueError, "differs"):
                    run_batch(source, output, "different research", config, workers=1)
