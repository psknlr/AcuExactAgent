import unittest
from acuexact.extraction import extract_quotes
from acuexact.pdf import text_quality_flags


def run(text, segments=None):
    return extract_quotes(segments or [{"text": text, "page": 1}], {"id": "a", "title": "target"})


class ExtractionTests(unittest.TestCase):
    def test_all_unknown_and_weak_quotes_remain(self):
        r = run('“肩痛”“肩不举”“无关词”“肩痛不可忍。”')
        self.assertEqual([q["text"] for q in r["quotes"]], ["无关词", "肩痛不可忍。"])
        self.assertEqual(len(r["excluded"]), 2)

    def test_no_source_invention(self):
        r = run('《素问》是一本书。另引：“未注明来源。”')
        self.assertIsNone(r["quotes"][0]["source_explicit"]["book"])

    def test_explicit_before_and_after(self):
        r = run('《书甲·篇乙》曰：“甲乙。”另引“丙丁。”（《书丙·篇丁》）')
        self.assertEqual(r["quotes"][0]["source_explicit"]["book"], "书甲")
        self.assertEqual(r["quotes"][0]["source_explicit"]["chapter"], "篇乙")
        self.assertEqual(r["quotes"][1]["source_explicit"]["book"], "书丙")

    def test_next_attribution_not_attached_to_previous_quote(self):
        r = run('“甲”《乙书》曰：“乙”')
        self.assertIsNone(r["quotes"][0]["source_explicit"]["book"])
        self.assertEqual(r["quotes"][1]["source_explicit"]["book"], "乙书")

    def test_attribution_can_cross_layout_newline(self):
        r = run('《素问·痹论》曰：\n“风寒湿三气杂至。”')
        self.assertEqual(r["quotes"][0]["source_explicit"]["chapter"], "痹论")

    def test_valid_quote_after_unclosed_quote_is_recovered_for_review(self):
        r = run('前文“残缺。后文曰：“完整古文。”')
        self.assertEqual(r["quotes"][0]["text"], "完整古文。")
        self.assertIn("recovered_with_unclosed_quote_context", r["quotes"][0]["review_flags"])

    def test_duplicate_occurrences_preserved(self):
        r = run('“甲乙。”“甲乙。”')
        self.assertEqual(len(r["quotes"]), 2)
        self.assertNotEqual(r["quotes"][0]["id"], r["quotes"][1]["id"])

    def test_across_region_page_boundary(self):
        r = run('', [{"text": '引“肩背', "page": 1}, {"text": '疼痛。”', "page": 2}])
        q = r["quotes"][0]
        self.assertEqual(q["text"], "肩背疼痛。")
        self.assertEqual(q["text_raw"], "肩背\n疼痛。")
        self.assertEqual([e["page"] for e in q["evidence"]], [1, 2])

    def test_traditional_and_punctuation_unchanged(self):
        self.assertEqual(run('“肩臂痠痛；氣血不通。”')["quotes"][0]["text"], "肩臂痠痛；氣血不通。")

    def test_nested_quotes_preserved_in_outer_quote(self):
        self.assertEqual(run('“曰「内文」，外文。”')["quotes"][0]["text"], '曰「内文」，外文。')

    def test_unclosed_is_flagged_not_hallucinated(self):
        r = run('“缺右引号。')
        self.assertEqual(r["quotes"], [])
        self.assertIn("unclosed_quote", [w["code"] for w in r["warnings"]])
        self.assertEqual(r["status"], "needs_review")

    def test_scope_only_selected_segments(self):
        selected = [{"page": 1, "bbox": [0, 0, 200, 200], "text": '目标：“应提取。”'}]
        r = run('', selected)
        self.assertEqual([q["text"] for q in r["quotes"]], ["应提取。"])

    def test_corrupt_text_flags(self):
        flags = text_quality_flags('缺左引号”\u0082\ue001(cid:1)')
        self.assertEqual(set(flags), {"unmapped_glyphs", "private_use_characters", "control_characters", "unbalanced_curly_quotes"})


if __name__ == "__main__":
    unittest.main()
