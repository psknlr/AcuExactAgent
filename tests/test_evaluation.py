import unittest
from acuexact.evaluation import evaluate
from acuexact.export import to_tsv


class EvaluationTests(unittest.TestCase):
    def test_empty_gold_not_perfect_score(self):
        self.assertIsNone(evaluate([], [])["f1"])
        self.assertFalse(evaluate([], [])["metrics_available"])

    def test_unlabelled_chapter_not_scored_as_wrong(self):
        p = {"article_id": "a", "text": "甲", "source_explicit": {"book": "某书", "chapter": "某篇"}}
        g = {"article_id": "a", "text": "甲", "source_explicit": {"book": "某书"}}
        self.assertEqual(evaluate([p], [g])["source_mismatches"], [])

    def test_duplicates_article_identity_and_character_errors(self):
        gold = [{"article_id": "a", "text": "肩痛。"}, {"article_id": "a", "text": "肩痛。"},
                {"article_id": "b", "text": "肩痛。"}]
        pred = [{"article_id": "a", "text": "肩痛。"}, {"article_id": "b", "text": "肩疼。"}]
        r = evaluate(pred, gold)
        self.assertEqual((r["true_positive"], r["false_positive"], r["false_negative"]), (1, 1, 2))

    def test_extra_source_is_error(self):
        p = {"id": "q1", "article_id": "a", "text": "甲", "source_explicit": {"book": "猜的书", "chapter": None}}
        g = {"article_id": "a", "text": "甲", "source_explicit": {"book": None, "chapter": None}}
        self.assertEqual(evaluate([p], [g])["source_false_additions"], ["q1"])

    def test_layout_metric_not_same_as_exact(self):
        p, g = [{"article_id": "a", "text": "甲 乙"}], [{"article_id": "a", "text": "甲乙"}]
        self.assertEqual(evaluate(p, g)["true_positive"], 0)
        self.assertEqual(evaluate(p, g, layout_only=True)["true_positive"], 1)

    def test_tsv_formula_injection_guard(self):
        self.assertIn("'=HYPERLINK", to_tsv([{"text": '=HYPERLINK("x")', "id": "1"}]))


if __name__ == "__main__":
    unittest.main()
