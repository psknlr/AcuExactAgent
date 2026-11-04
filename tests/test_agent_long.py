import copy
from contextlib import contextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from acuexact.agent import run_pdf_agent
from acuexact.agent_batch import pipeline_complete
from acuexact.agent_long import merge_parts, run_long_pdf_agent
from acuexact.extraction import extract_quotes
from test_agent import FakeClient, FakePage, plan


def part(text, page):
    segment = dict(page=page, bbox=[0, 0, 600, 800], region_id="r1", text=text, method="pdf_text")
    result = extract_quotes([segment], {"id": "article", "title": "肩痛研究"})
    result.update(segments=[segment], status="needs_review")
    result["audit"].update(selected_pages=[page], regions=[dict(id="r1", page=page)], stages=["review_quotes"],
                           calls_used=1, model_calls=[])
    for quote in result["quotes"]:
        quote["relevance"] = dict(label="弱相关", method="llm_agent")
    return result


class MergeLongTests(unittest.TestCase):
    def test_cross_part_quote_retained_with_original_offsets_and_pages(self):
        first, second = part('正文“肩臂', 1), part('疼痛。”另引“同文”“同文”', 2)
        original = copy.deepcopy([first, second])
        merged = merge_parts([first, second])
        self.assertEqual([q["text"] for q in merged["quotes"]], ["肩臂疼痛。", "同文", "同文"])
        self.assertEqual([e["page"] for e in merged["quotes"][0]["evidence"]], [1, 2])
        text = "\n".join(s["text"] for s in merged["segments"])
        for q in merged["quotes"]:
            self.assertEqual(text[q["char_start"] + 1:q["char_end"] - 1].replace("\n", ""), q["text"])
        self.assertEqual(len({q["id"] for q in merged["quotes"]}), 3)
        self.assertEqual(len({s["region_id"] for s in merged["segments"]}), 2)
        self.assertFalse(any(w["code"] in {"unclosed_quote", "unmatched_closing_quote"} for w in merged["warnings"]))
        self.assertEqual([first, second], original)

    def test_disease_name_split_across_parts_stays_excluded(self):
        merged = merge_parts([part('“肩', 1), part('痛”', 2)])
        self.assertEqual(merged["quotes"], [])
        self.assertEqual(merged["excluded"][0]["text"], "肩痛")


class ContinuationTests(unittest.TestCase):
    def test_long_document_reads_every_page_and_resumes_completed_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.pdf"
            path.write_bytes(b"same-source")
            pages = [FakePage("肩痛研究" + "正常正文内容。" * 20) for _ in range(21)]

            @contextmanager
            def opened(_):
                yield SimpleNamespace(pages=pages)

            def dynamic_plan(payload):
                return plan(pages=tuple(page["page"] for page in payload["pages"]))

            with patch("acuexact.agent.open_pdf", opened):
                client = FakeClient({"plan_article": dynamic_plan}, vision=True)
                result = run_long_pdf_agent(path, "提取引文", client, total_pages=21,
                    directory=directory, target_title="肩痛研究")
                self.assertTrue(pipeline_complete(result))
                self.assertEqual(result["audit"]["selected_pages"], list(range(1, 22)))
                self.assertEqual(len(result["audit"]["parts"]), 3)
                self.assertEqual(sum(r["stage"] == "plan_article" for r in client.requests), 3)
                self.assertTrue(all(r["images"] is None for r in client.requests if r["stage"] == "plan_article"))
                cached_client = FakeClient(vision=True)
                cached = run_long_pdf_agent(path, "提取引文", cached_client, total_pages=21,
                    directory=directory, target_title="肩痛研究")
                self.assertTrue(pipeline_complete(cached))
                self.assertEqual(cached_client.requests, [])
                path.write_bytes(b"different-source")
                changed_client = FakeClient({"plan_article": dynamic_plan}, vision=True)
                changed = run_long_pdf_agent(path, "提取引文", changed_client, total_pages=21,
                    directory=directory, target_title="肩痛研究")
                self.assertTrue(pipeline_complete(changed))
                self.assertEqual(sum(r["stage"] == "plan_article" for r in changed_client.requests), 3)

    def test_later_pages_use_verified_continuation_without_retranscribing_clean_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.pdf"
            path.write_bytes(b"same-source")
            pages = [FakePage("肩痛研究" + "正常正文内容。" * 20), FakePage("接续前页的正文。" * 20)]

            @contextmanager
            def opened(_):
                yield SimpleNamespace(pages=pages)

            with patch("acuexact.agent.open_pdf", opened):
                first = run_pdf_agent(path, "提取引文", FakeClient(vision=True), target_title="肩痛研究",
                    page_numbers=[1], transcribe_pages=False, transcribe_clean_regions=False)
                client = FakeClient({"plan_article": plan(pages=(2,))}, vision=True)
                second = run_pdf_agent(path, "提取引文", client, target_title="肩痛研究",
                    page_numbers=[2], continuation_of=first, transcribe_pages=False, transcribe_clean_regions=False)
                self.assertEqual(second["audit"]["selected_pages"], [2])
                self.assertEqual(client.requests[0]["payload"]["continuation_context"]["previous_pages"], [1])
                self.assertFalse(any(r["stage"] == "region_transcription" for r in client.requests))
                self.assertIn("review_quotes", second["audit"]["stages"])
                wrong = copy.deepcopy(first)
                wrong["audit"]["pdf_sha256"] = "different-source"
                rejected_client = FakeClient(vision=True)
                rejected = run_pdf_agent(path, "提取引文", rejected_client, target_title="肩痛研究",
                    page_numbers=[2], continuation_of=wrong, transcribe_pages=False)
                self.assertEqual(rejected_client.requests, [])
                self.assertTrue(any(w["code"] == "agent_incomplete" for w in rejected["warnings"]))
