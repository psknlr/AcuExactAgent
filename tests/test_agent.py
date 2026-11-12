"""No-network tests for the model/tool feedback loop and evidence invariants."""
from contextlib import contextmanager
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from acuexact.agent import run_pdf_agent, _image, _transcription_payload, _include_edge_glyphs
from acuexact.classification import DIMENSIONS
from acuexact.providers import ProviderError, ProviderRefusal


class FakeImage:
    def save(self, stream, format):
        stream.write(b"test-image-bytes")


class FakePage:
    width, height = 600, 800
    edges = []

    def __init__(self, text, crops=None):
        self.text = text
        self.crops = crops or {}

    def extract_text(self, **kwargs):
        return self.text

    def crop(self, bbox):
        return FakePage(self.crops.get(tuple(bbox), self.text))

    def to_image(self, **kwargs):
        return SimpleNamespace(original=FakeImage())


def plan(pages=(1,), title="肩痛研究", confidence=.96):
    return {"article_candidates": [{"id": "article-1", "title": title, "pages": list(pages)}],
            "selected_article_id": "article-1", "multiple_articles": False, "needs_scope": False,
            "confidence": confidence, "reason": "目标标题及正文清晰", "classification_dimensions": list(DIMENSIONS),
            "regions": [{"page": p, "bbox": [0, 0, 600, 800], "article_id": "article-1", "order": i}
                        for i, p in enumerate(pages)]}


class FakeClient:
    def __init__(self, responses=None, vision=False, hook=None):
        self.responses = responses or {}
        self.config = SimpleNamespace(model="fake-research-model", vision=vision)
        self.calls, self.requests = [], []
        self.hook = hook

    def generate_json(self, system, payload, schema, images=None, schema_name=None):
        self.calls.append({"model": self.config.model, "schema_name": schema_name,
                           "api_key": "must-never-appear", "status": "completed"})
        self.requests.append({"stage": schema_name, "payload": copy.deepcopy(payload), "images": images,
                              "system": system, "schema": schema})
        if self.hook:
            self.hook(schema_name)
        response = self.responses.get(schema_name)
        if isinstance(response, list):
            response = response.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(payload)
        if response is not None:
            return copy.deepcopy(response)
        if schema_name == "plan_article":
            return plan()
        if schema_name in {"review_quotes", "repair_quotes"}:
            return {"missing_quotes": [], "issues": [], "all_quotes_checked": True}
        if schema_name in {"classify_quotes", "repair_classification"}:
            return {"results": [{"id": q["id"], "label": "弱相关", "dimensions": [],
                                 "reason": "本条未直接涉及用户指定的研究维度，但仍完整保留。", "needs_review": False}
                                for q in payload["quotes"]]}
        raise AssertionError(f"Unexpected stage {schema_name}")


class AgentTests(unittest.TestCase):
    def test_crop_preserves_edge_punctuation_without_including_neighboring_column(self):
        page = FakePage("正文")
        page.chars = [
            {"text": "《", "x0": 8, "x1": 13.9, "top": 20, "bottom": 30},
            {"text": "肩", "x0": 15, "x1": 25, "top": 20, "bottom": 30},
            {"text": "末", "x0": 44, "x1": 50, "top": 20, "bottom": 30},
            {"text": "邻", "x0": 61, "x1": 71, "top": 20, "bottom": 30}]
        bbox = _include_edge_glyphs([14, 15, 45, 35], page)
        self.assertLess(bbox[0], 8)
        self.assertGreater(bbox[2], 50)
        self.assertLess(bbox[2], 61)

    def test_reading_images_use_more_detail_but_stay_bounded(self):
        page = FakePage("正文")
        with patch.object(page, "to_image", wraps=page.to_image) as render:
            _image(page)
            planning = render.call_args.kwargs["resolution"]
            _image(page, reading=True)
            reading = render.call_args.kwargs["resolution"]
        self.assertGreater(reading, planning)
        self.assertLessEqual(reading * page.height / 72, 2600)

    def test_legacy_punctuation_reference_is_advisory_and_never_replaces_vision(self):
        original = "肩痛研究" * 20 + "5内经6/有因痰湿留注0"
        transcript = {"text": "肩痛研究“原页可见文字。”", "uncertainties": [], "confidence": .9}
        client = FakeClient({"region_transcription": transcript}, vision=True)
        result = self.run_agent(original, client, transcribe_pages=False)
        request = next(r for r in client.requests if r["stage"] == "region_transcription")
        self.assertEqual(request["payload"]["pdf_text_reference"], original)
        self.assertTrue(request["payload"]["reference_is_unverified"])
        self.assertEqual(result["quotes"][0]["text"], "原页可见文字。")
        self.assertEqual(result["segments"][0]["pdf_text_original"], original)

    def test_garbled_or_excessive_native_text_is_not_sent_as_reading_reference(self):
        for text in ["(cid:123)正文", "坏字�正文", "私用字\ue001", "控制字\x82", "中" * 16_001]:
            with self.subTest(text=text[:20]):
                payload = _transcription_payload(1, [0, 0, 600, 800], "题名", text)
                self.assertNotIn("pdf_text_reference", payload)

    def test_provider_refusal_stops_further_review_and_classification_requests(self):
        for stage in ("review_quotes", "classify_quotes"):
            client = FakeClient({stage: ProviderRefusal("模型拒绝了该请求。")})
            result = self.run_agent('正文' + '“肩臂疼痛。”' * 21, client)
            requests = [r["stage"] for r in client.requests]
            self.assertEqual(requests[-1], stage)
            self.assertEqual(requests.count(stage), 1)
            self.assertTrue(any(w["code"] == "agent_incomplete" for w in result["warnings"]))
            self.assertTrue(any(q.get("relevance", {}).get("method") == "unclassified" for q in result["quotes"]))

    def test_classification_uses_short_ids_without_changing_occurrence_ids(self):
        client = FakeClient()
        result = self.run_agent('正文“肩臂疼痛。”另一次“肩臂疼痛。”', client)
        request = next(r for r in client.requests if r["stage"] == "classify_quotes")
        self.assertEqual([q["id"] for q in request["payload"]["quotes"]], ["q1", "q2"])
        self.assertEqual(request["schema"]["properties"]["results"]["items"]["properties"]["id"]["enum"], ["q1", "q2"])
        self.assertEqual(result["audit"]["classification_id_aliases"],
                         [{"q1": result["quotes"][0]["id"], "q2": result["quotes"][1]["id"]}])
        self.assertTrue(all(q["id"].startswith(result["article"]["id"] + ":q") for q in result["quotes"]))
        self.assertEqual(len({q["id"] for q in result["quotes"]}), 2)
        self.assertTrue(all(q["relevance"]["method"] == "llm_agent" for q in result["quotes"]))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "article.pdf"
        self.path.write_bytes(b"%PDF-test-document")

    def run_agent(self, text='肩痛研究 正文引“肩臂疼痛，气血不通。”', client=None, pages=None, **kwargs):
        document = SimpleNamespace(pages=pages or [FakePage(text)])

        @contextmanager
        def opened(path):
            yield document

        with patch("acuexact.agent.open_pdf", opened):
            return run_pdf_agent(self.path, kwargs.pop("prompt", "按针刺治疗肩痛的四个维度分类"),
                                 client or FakeClient(), **kwargs)

    def test_all_quotes_retained_without_inventing_source(self):
        client = FakeClient()
        result = self.run_agent('肩痛研究《书甲·篇乙》曰：“肩痛应取穴。”另引“陌生言论。”“肩痛”“肩不举”', client)
        self.assertEqual([q["text"] for q in result["quotes"]], ["肩痛应取穴。", "陌生言论。"])
        self.assertEqual(len(result["excluded"]), 2)
        self.assertEqual(result["quotes"][0]["source_explicit"]["book"], "书甲")
        self.assertIsNone(result["quotes"][1]["source_explicit"]["book"])
        self.assertEqual(result["quotes"][1]["relevance"]["label"], "弱相关")
        self.assertEqual([r["stage"] for r in client.requests], ["plan_article", "review_quotes", "classify_quotes"])
        self.assertNotIn("must-never-appear", str(result))

    def test_multiple_articles_require_target_even_with_manual_page_selection(self):
        multi = plan()
        multi["article_candidates"].append({"id": "article-2", "title": "其他研究", "pages": [1]})
        multi["multiple_articles"] = True
        client = FakeClient({"plan_article": multi})
        result = self.run_agent(client=client, page_numbers=[1])
        self.assertEqual(result["status"], "needs_scope")
        self.assertEqual(result["quotes"], [])
        self.assertEqual(len(result["article_candidates"]), 2)
        self.assertEqual(len(client.calls), 1)

    def test_target_region_excludes_second_article_on_same_page(self):
        scoped = plan()
        scoped["regions"][0]["bbox"] = [0, 0, 600, 300]
        page = FakePage('肩痛研究“应该提取。”其他文章“不能混入。”',
                        {(0., 0., 600., 300.): '肩痛研究引文“应该提取。”'})
        result = self.run_agent(client=FakeClient({"plan_article": scoped}), pages=[page], target_title="肩痛研究")
        self.assertEqual([q["text"] for q in result["quotes"]], ["应该提取。"])
        self.assertEqual(result["quotes"][0]["evidence"][0]["bbox"], [0., 0., 600., 300.])

    def test_wrong_target_title_is_rejected(self):
        result = self.run_agent(client=FakeClient({"plan_article": plan(title="错误的另一篇文章")}), target_title="肩痛研究")
        self.assertEqual(result["status"], "needs_scope")
        self.assertFalse(result["quotes"])

    def test_bibliographic_title_variants_do_not_rewrite_quote_evidence(self):
        for target, visible in (("针刺阳蹻脉治疗肩关节周围炎", "针刺阳跷脉治疗肩关节周围炎"),
                                ("针剌条口穴治疗肩周炎", "针刺条口穴治疗肩周炎")):
            with self.subTest(target=target):
                result = self.run_agent(text='原文“針剌陽蹻脈。”',
                                        client=FakeClient({"plan_article": plan(title=visible)}),
                                        target_title=target)
                self.assertNotEqual(result["status"], "needs_scope")
                self.assertEqual(result["quotes"][0]["text"], "針剌陽蹻脈。")

    def test_visual_title_mismatch_rechecks_image_once(self):
        transcript = {"text": '肩痛研究“引文保留。”', "uncertainties": [], "confidence": .9}
        client = FakeClient({"plan_article": [plan(title="误读标题"), plan(title="肩痛研究")],
                             "region_transcription": transcript}, vision=True)
        result = self.run_agent(client=client, target_title="肩痛研究")
        planning = [r for r in client.requests if r["stage"] == "plan_article"]
        self.assertEqual(len(planning), 2)
        self.assertIn("validation_error", planning[1]["payload"])
        self.assertTrue(planning[1]["images"])
        self.assertNotEqual(result["status"], "needs_scope")

    def test_visual_recheck_still_rejects_another_article(self):
        client = FakeClient({"plan_article": plan(title="另一篇文章")}, vision=True)
        result = self.run_agent(client=client, target_title="肩痛研究")
        self.assertEqual(result["status"], "needs_scope")
        self.assertFalse(result["quotes"])

    def test_visual_region_repair_receives_exact_bounds_and_previous_plan(self):
        bad = plan()
        bad["regions"][0]["bbox"][2] = 600.0263
        client = FakeClient({"plan_article": [bad, plan()]}, vision=True)
        result = self.run_agent(client=client, target_title="肩痛研究", transcribe_clean_regions=False)
        request = [r for r in client.requests if r["stage"] == "plan_article"][1]
        self.assertEqual(request["payload"]["previous_plan"], bad)
        self.assertIn("600.0263", request["payload"]["validation_error"])
        self.assertIn("width=600.0", request["payload"]["validation_error"])
        self.assertTrue(result["quotes"])

    def test_sparse_line_only_plan_is_rejected_and_can_be_replanned(self):
        sparse = plan()
        sparse["regions"] = [dict(page=1, bbox=[0, y, 600, y + 12], article_id="article-1", order=i)
                             for i, y in enumerate((60, 300, 700))]
        result = self.run_agent(client=FakeClient({"plan_article": sparse}))
        self.assertFalse(result["quotes"])
        self.assertTrue(any(w["code"] == "agent_incomplete" for w in result["warnings"]))
        client = FakeClient({"plan_article": [sparse, plan()]}, vision=True)
        result = self.run_agent(client=client, target_title="肩痛研究", transcribe_clean_regions=False)
        repair = [r for r in client.requests if r["stage"] == "plan_article"][1]
        self.assertIn("零散行", repair["payload"]["validation_error"])
        self.assertTrue(result["quotes"])

    def test_visual_repair_still_rejects_overlapping_regions(self):
        bad = plan()
        bad["regions"].append(dict(bad["regions"][0], order=1))
        client = FakeClient({"plan_article": [bad, bad]}, vision=True)
        result = self.run_agent(client=client, target_title="肩痛研究")
        request = [r for r in client.requests if r["stage"] == "plan_article"][1]
        self.assertIn("order=1", request["payload"]["validation_error"])
        self.assertIn("order=0", request["payload"]["validation_error"])
        self.assertEqual(result["quotes"], [])
        self.assertTrue(any(w["code"] == "agent_incomplete" for w in result["warnings"]))

    def test_batch_can_plan_scan_from_image_then_transcribe_selected_regions(self):
        transcript = {"text": '肩痛研究“引文保留。”', "uncertainties": [], "confidence": .9}
        client = FakeClient({"region_transcription": transcript}, vision=True)
        result = self.run_agent(text="(cid:123)", client=client, transcribe_pages=False)
        stages = [r["stage"] for r in client.requests]
        self.assertNotIn("page_transcription", stages)
        self.assertIn("region_transcription", stages)
        planning = client.requests[0]
        self.assertTrue(planning["images"])
        self.assertEqual(planning["payload"]["pages"][0]["text"], "")
        self.assertEqual([q["text"] for q in result["quotes"]], ["引文保留。"])

    def test_articles_in_same_pdf_have_distinct_stable_ids(self):
        first = self.run_agent(client=FakeClient({"plan_article": plan(title="肩痛研究")}), target_title="肩痛研究")
        second = self.run_agent(client=FakeClient({"plan_article": plan(title="另一文章")}), target_title="另一文章")
        repeat = self.run_agent(client=FakeClient({"plan_article": plan(title="肩痛 研究")}), target_title="肩痛研究")
        self.assertNotEqual(first["article"]["id"], second["article"]["id"])
        self.assertEqual(first["article"]["id"], repeat["article"]["id"])
        self.assertNotEqual(first["quotes"][0]["id"], second["quotes"][0]["id"])

    def test_overlap_and_out_of_bounds_fail_closed(self):
        for boxes in ([[0, 0, 600, 500], [0, 400, 600, 800]], [[0, 0, 650, 800]]):
            with self.subTest(boxes=boxes):
                bad = plan()
                bad["regions"] = [{"page": 1, "bbox": box, "article_id": "article-1", "order": i} for i, box in enumerate(boxes)]
                result = self.run_agent(client=FakeClient({"plan_article": bad}))
                self.assertEqual(result["status"], "needs_review")
                self.assertFalse(result["quotes"])

    def test_scanned_pdf_requires_vision_without_false_success(self):
        client = FakeClient()
        result = self.run_agent(text="", client=client)
        self.assertEqual(result["status"], "needs_review")
        self.assertFalse(result["quotes"])
        self.assertEqual(client.calls, [])
        self.assertIn("vision_required", [w["code"] for w in result["warnings"]])

    def test_vision_transcript_is_never_claimed_as_pdf_text(self):
        transcript = {"text": '肩痛研究 引文“肩臂痠痛；氣血不通。”', "uncertainties": [], "confidence": .99}
        client = FakeClient({"page_transcription": transcript, "region_transcription": transcript}, vision=True)
        result = self.run_agent(text="", client=client)
        quote = result["quotes"][0]
        self.assertEqual(quote["text"], "肩臂痠痛；氣血不通。")
        self.assertIn("vision_transcription_requires_review", quote["review_flags"])
        self.assertEqual(quote["evidence"][0]["method"], "vision_transcription")
        self.assertEqual(result["segments"][0]["pdf_text_original"], "")
        self.assertEqual(result["status"], "needs_review")
        self.assertTrue(any(r["images"] for r in client.requests))
        self.assertNotIn("data:image", str(result))

    def test_separate_vision_client_is_used_for_scan(self):
        transcript = {"text": '肩痛研究引文“肩痛久留其针。”', "uncertainties": [], "confidence": .9}
        vision = FakeClient({"page_transcription": transcript, "region_transcription": transcript}, vision=True)
        main = FakeClient()
        result = self.run_agent(text="", client=main, vision_client=vision)
        self.assertEqual(len(result["quotes"]), 1)
        self.assertEqual([r["stage"] for r in vision.requests], ["page_transcription", "plan_article", "region_transcription"])

    def test_hallucinated_review_addition_is_rejected_then_repaired(self):
        bad = {"missing_quotes": [{"region_id": "r1", "text": "不存在的古文", "evidence": '“不存在的古文”'}],
               "issues": [], "all_quotes_checked": True}
        client = FakeClient({"review_quotes": bad})
        result = self.run_agent(client=client)
        self.assertEqual(len(result["quotes"]), 1)
        self.assertNotIn("不存在", str(result["quotes"]))
        self.assertIn("repair_quotes", [r["stage"] for r in client.requests])

    def test_exact_straight_quote_can_be_recovered_and_classified(self):
        review = {"missing_quotes": [{"region_id": "r1", "text": "肩臂痠痛。", "evidence": '"肩臂\n痠痛。"'}],
                  "issues": [], "all_quotes_checked": True}
        result = self.run_agent('肩痛研究："肩臂\n痠痛。"', FakeClient({"review_quotes": review}))
        self.assertEqual(result["quotes"][0]["text"], "肩臂痠痛。")
        self.assertEqual(result["quotes"][0]["text_raw"], "肩臂\n痠痛。")
        self.assertEqual(result["quotes"][0]["relevance"]["method"], "llm_agent")

    def test_review_cannot_correct_character_or_forge_explicit_source(self):
        review = {"missing_quotes": [{"region_id": "r1", "text": "肩臂酸痛。", "evidence": '“肩臂痠痛。”',
                                      "source_explicit": {"book": "黄帝内经"}}], "issues": [], "all_quotes_checked": True}
        client = FakeClient({"review_quotes": review, "repair_quotes": review})
        result = self.run_agent('肩痛研究 原文引“肩臂痠痛。”', client)
        self.assertEqual([q["text"] for q in result["quotes"]], ["肩臂痠痛。"])
        self.assertIsNone(result["quotes"][0]["source_explicit"]["book"])
        self.assertIn("quote_review_incomplete", [w["code"] for w in result["warnings"]])

    def test_classification_missing_ids_get_one_repair(self):
        client = FakeClient({"classify_quotes": {"results": []}})
        result = self.run_agent('肩痛研究“甲乙丙丁。”“戊己庚辛。”', client)
        self.assertEqual(len(result["quotes"]), 2)
        self.assertTrue(all(q["relevance"]["method"] == "llm_agent" for q in result["quotes"]))
        self.assertEqual([r["stage"] for r in client.requests].count("repair_classification"), 1)

    def test_duplicate_and_extra_ids_cannot_drop_rows(self):
        def invalid(payload):
            row = {"id": payload["quotes"][0]["id"], "label": "弱相关", "dimensions": [], "reason": "理由", "needs_review": False}
            return {"results": [row, copy.deepcopy(row), dict(row, id="forged-id")]}
        client = FakeClient({"classify_quotes": invalid, "repair_classification": invalid})
        result = self.run_agent('肩痛研究“甲乙丙丁。”“戊己庚辛。”', client)
        self.assertEqual(len(result["quotes"]), 2)
        self.assertTrue(all(q["relevance"]["label"] == "待分类" for q in result["quotes"]))
        self.assertEqual(result["status"], "needs_review")

    def test_strong_classification_requires_dimension(self):
        def invalid(payload):
            return {"results": [{"id": q["id"], "label": "强相关", "dimensions": [], "reason": "理由", "needs_review": False} for q in payload["quotes"]]}
        client = FakeClient({"classify_quotes": invalid, "repair_classification": invalid})
        result = self.run_agent(client=client)
        self.assertEqual(result["quotes"][0]["relevance"]["label"], "待分类")

    def test_custom_prompt_and_dimensions_are_forwarded(self):
        custom = plan()
        custom["classification_dimensions"] = ["腰痛取穴", "腰痛辨证"]
        client = FakeClient({"plan_article": custom})
        prompt = "研究改为针刺治疗腰痛，维度为腰痛取穴、腰痛辨证"
        result = self.run_agent(client=client, prompt=prompt)
        classification = next(r for r in client.requests if r["stage"] == "classify_quotes")
        self.assertEqual(classification["payload"]["user_prompt"], prompt)
        self.assertEqual(classification["payload"]["dimensions"], custom["classification_dimensions"])
        self.assertEqual(result["audit"]["research_prompt"], prompt)

    def test_batches_never_exceed_twenty_and_keep_duplicates(self):
        client = FakeClient()
        result = self.run_agent('肩痛研究' + '“肩臂痠痛。”' * 43, client)
        batches = [r for r in client.requests if r["stage"] == "classify_quotes"]
        self.assertEqual([len(r["payload"]["quotes"]) for r in batches], [20, 20, 3])
        self.assertEqual(len(result["quotes"]), 43)
        self.assertEqual(len({q["id"] for q in result["quotes"]}), 43)

    def test_page_limit_never_silently_truncates(self):
        client = FakeClient()
        pages = [FakePage('肩痛研究 引“肩痛取穴。”')] * 3
        result = self.run_agent(client=client, pages=pages, max_pages=2)
        self.assertEqual(result["status"], "needs_scope")
        self.assertEqual(client.calls, [])
        self.assertIn("page_limit_exceeded", [w["code"] for w in result["warnings"]])

    def test_call_budget_preserves_quotes_with_pending_classification(self):
        client = FakeClient()
        result = self.run_agent(client=client, max_calls=2)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(result["quotes"]), 1)
        self.assertEqual(result["quotes"][0]["relevance"]["label"], "待分类")
        self.assertEqual(result["audit"]["calls_used"], 2)

    def test_cancellation_after_model_call_is_observed_and_evidence_kept(self):
        state = {"cancelled": False}
        client = FakeClient(hook=lambda stage: state.update(cancelled=True) if stage == "review_quotes" else None)
        result = self.run_agent(client=client, cancelled=lambda: state["cancelled"])
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(len(result["quotes"]), 1)
        self.assertEqual(result["quotes"][0]["relevance"]["label"], "待分类")

    def test_low_confidence_boundaries_flag_every_quote(self):
        result = self.run_agent('肩痛研究“甲乙丙丁。”“戊己庚辛。”', FakeClient({"plan_article": plan(confidence=.5)}))
        self.assertTrue(all("automatic_article_boundary_requires_review" in q["review_flags"] for q in result["quotes"]))

    def test_provider_failure_preserves_evidence_and_redacts_errors(self):
        client = FakeClient({"classify_quotes": RuntimeError("secret API token goes here")})
        events = []
        result = self.run_agent(client=client, on_event=events.append)
        self.assertEqual(len(result["quotes"]), 1)
        self.assertEqual(result["quotes"][0]["relevance"]["label"], "待分类")
        self.assertNotIn("secret API", str(result))
        self.assertEqual([e["progress"] for e in events], sorted(e["progress"] for e in events))

    def test_vision_planning_receives_page_images(self):
        transcript = {"text": '肩痛研究 引文“肩痛取穴。”', "uncertainties": [], "confidence": .9}
        client = FakeClient({"region_transcription": transcript}, vision=True)
        result = self.run_agent('肩痛研究 引文“肩痛取穴。”', client)
        planning = next(r for r in client.requests if r["stage"] == "plan_article")
        self.assertEqual(len(planning["images"]), 1)
        self.assertEqual(planning["payload"]["image_page_order"], [1])
        self.assertIn("automatic_article_boundary_requires_review", result["quotes"][0]["review_flags"])

    def test_vision_limit_requires_explicit_smaller_scope(self):
        client = FakeClient(vision=True)
        result = self.run_agent(client=client, pages=[FakePage("page content") for _ in range(21)])
        self.assertEqual(result["status"], "needs_scope")
        self.assertEqual(client.requests, [])
        self.assertIn("vision_page_limit_exceeded", [w["code"] for w in result["warnings"]])

    def test_unbalanced_text_quotes_trigger_visual_transcription(self):
        transcript = {"text": '肩痛研究 古文“肩痛取穴。”', "uncertainties": [], "confidence": .9}
        client = FakeClient({"page_transcription": transcript, "region_transcription": transcript}, vision=True)
        result = self.run_agent('肩痛研究 古文肩痛取穴。”', client)
        self.assertEqual(result["quotes"][0]["text"], "肩痛取穴。")
        self.assertEqual(result["quotes"][0]["evidence"][0]["method"], "vision_transcription")

    def test_visual_quote_with_both_delimiters_missing_from_layer_is_retained(self):
        transcript = {"text": '肩痛研究 古文“肩痛取穴。”', "uncertainties": [], "confidence": .9}
        client = FakeClient({"region_transcription": transcript}, vision=True)
        result = self.run_agent('肩痛研究 古文肩痛取穴。', client)
        self.assertEqual(result["quotes"][0]["text"], "肩痛取穴。")
        self.assertEqual(result["quotes"][0]["evidence"][0]["method"], "vision_transcription")
        self.assertIn("vision_transcription_requires_review", result["quotes"][0]["review_flags"])
        self.assertEqual(result["segments"][0]["method"], "pdf_text")

    def test_visual_disagreement_does_not_rewrite_original(self):
        transcript = {"text": '肩痛研究 古文“肩臂酸痛。”', "uncertainties": [], "confidence": .9}
        client = FakeClient({"region_transcription": transcript}, vision=True)
        result = self.run_agent('肩痛研究 古文“肩臂痠痛。”', client)
        original = next(q for q in result["quotes"] if q["evidence"][0]["method"] == "pdf_text")
        visual = next(q for q in result["quotes"] if q["evidence"][0]["method"] == "vision_transcription")
        self.assertEqual(original["text"], "肩臂痠痛。")
        self.assertEqual(visual["text"], "肩臂酸痛。")
        self.assertIn("visual_text_layer_disagreement", visual["review_flags"])

    def test_visual_offsets_are_local_and_reading_order_spans_regions(self):
        first = "肩痛研究正文" * 30 + '“先出现原文。”'
        second = "第二选区正文丢失引号的古文。"
        page = FakePage(first + second, {(0., 0., 600., 400.): first, (0., 400., 600., 800.): second})
        selected = plan()
        selected["regions"] = [
            {"page": 1, "bbox": [0, 0, 600, 400], "article_id": "article-1", "order": 0},
            {"page": 1, "bbox": [0, 400, 600, 800], "article_id": "article-1", "order": 1}]
        transcripts = [{"text": first, "uncertainties": [], "confidence": .9},
                       {"text": '第二选区正文“后出现古文。”', "uncertainties": [], "confidence": .9}]
        client = FakeClient({"plan_article": selected, "region_transcription": transcripts}, vision=True)
        result = self.run_agent(client=client, pages=[page])
        self.assertEqual([q["text"] for q in result["quotes"]], ["先出现原文。", "后出现古文。"])
        visual = result["quotes"][1]
        self.assertEqual(visual["offset_space"], "vision_transcription:r2")
        self.assertEqual(visual["char_start"], visual["evidence"][0]["char_start"])
        self.assertEqual(visual["char_start"], 6)

    def test_provider_safe_error_is_visible_without_extra_agent_retry(self):
        client = FakeClient({"classify_quotes": ProviderError("API HTTP 401: 请检查 API key。")})
        result = self.run_agent(client=client)
        self.assertIn("HTTP 401", result["quotes"][0]["relevance"]["reason"])
        self.assertNotIn("repair_classification", [r["stage"] for r in client.requests])

    def test_http_retry_budget_and_hook_are_restored(self):
        class RetryClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.before_request = None
                self.http_attempts = 0

            def generate_json(self, *args, **kwargs):
                for _ in range(2):
                    self.before_request()
                    self.http_attempts += 1
                return super().generate_json(*args, **kwargs)

        client = RetryClient()
        result = self.run_agent(client=client, max_calls=2)
        self.assertEqual(client.http_attempts, 2)
        self.assertEqual(result["audit"]["calls_used"], 2)
        self.assertEqual(result["audit"]["call_budget_unit"], "http_requests_including_retries")
        self.assertIsNone(client.before_request)
        self.assertEqual(result["quotes"][0]["relevance"]["label"], "待分类")


if __name__ == "__main__":
    unittest.main()
