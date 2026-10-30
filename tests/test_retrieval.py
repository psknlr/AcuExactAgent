import unittest
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

from acuexact.retrieval import _allowed_url, lookup_source


class LookupTests(unittest.TestCase):
    def setUp(self):
        self.quote = {"text": "肩似拔，臑似折。", "source_explicit": {"book": None, "chapter": None},
                      "source_enriched": {"status": "unverified"}}

    def test_no_network_by_default_and_deep_copy(self):
        with patch("acuexact.retrieval._fetch_json", side_effect=AssertionError("network")):
            result = lookup_source(self.quote)
        self.assertEqual(result["source_lookup"]["status"], "disabled")
        result["source_explicit"]["book"] = "changed"
        self.assertIsNone(self.quote["source_explicit"]["book"])

    def test_match_is_candidate_and_never_overwrites_attribution(self):
        urls = []
        def fetch(url, timeout):
            urls.append(url)
            params = parse_qs(urlsplit(url).query)
            if params["action"] == ["query"]:
                return {"query": {"search": [{"pageid": 7, "title": "靈樞經/經脈"}]}}
            return {"parse": {"title": "靈樞經/經脈", "revid": 42,
                              "text": "<p>前文：肩似拔，<span>臑似折。</span>后文</p>"}}
        result = lookup_source(self.quote, enable_network=True, fetch=fetch)
        self.assertEqual(len(urls), 2)
        self.assertEqual(result["source_explicit"], self.quote["source_explicit"])
        self.assertEqual(result["source_enriched"], self.quote["source_enriched"])
        candidate = result["source_candidates"][0]
        self.assertEqual(candidate["status"], "candidate_requires_review")
        self.assertEqual(candidate["book"], "靈樞經")
        self.assertEqual(candidate["chapter"], "經脈")
        self.assertIn(self.quote["text"], candidate["evidence"])
        self.assertEqual(candidate["revision_url"], "https://zh.wikisource.org/w/index.php?oldid=42")

    def test_whitespace_can_differ_but_punctuation_cannot(self):
        def fetch(url, timeout):
            if "action=query" in url:
                return {"query": {"search": [{"pageid": 1, "title": "甲/一"}, {"pageid": 2, "title": "乙/二"}]}}
            pageid = parse_qs(urlsplit(url).query)["pageid"][0]
            html = "<p>肩似拔，\n臑似折。</p>" if pageid == "1" else "<p>肩似拔、臑似折。</p>"
            return {"parse": {"title": "甲/一" if pageid == "1" else "乙/二", "text": html}}
        result = lookup_source(self.quote, enable_network=True, fetch=fetch)
        self.assertEqual(len(result["source_candidates"]), 1)
        self.assertEqual(result["source_lookup"]["requests"], 3)

    def test_snippet_or_script_does_not_verify_quote(self):
        def fetch(url, timeout):
            if "action=query" in url:
                return {"query": {"search": [{"pageid": 1, "title": "甲", "snippet": self.quote["text"]}]}}
            return {"parse": {"title": "甲", "text": "<script>肩似拔，臑似折。</script><p>没有匹配条文</p>"}}
        result = lookup_source(self.quote, enable_network=True, fetch=fetch)
        self.assertEqual(result["source_candidates"], [])
        self.assertEqual(result["source_lookup"]["status"], "pending_review")

    def test_timeout_is_a_review_status(self):
        def fetch(url, timeout):
            raise TimeoutError("timed out")
        result = lookup_source(self.quote, enable_network=True, fetch=fetch)
        self.assertEqual(result["source_lookup"]["status"], "lookup_error_requires_review")
        self.assertEqual(result["source_explicit"], self.quote["source_explicit"])

    def test_empty_and_malformed_results_remain_unverified(self):
        for response, status in [({"query": {"search": []}}, "pending_review"),
                                 ({"query": None}, "lookup_error_requires_review"),
                                 ({"error": {"code": "ratelimited"}}, "lookup_error_requires_review")]:
            result = lookup_source(self.quote, enable_network=True, fetch=lambda url, timeout: response)
            self.assertEqual(result["source_candidates"], [])
            self.assertEqual(result["source_lookup"]["status"], status)

    def test_budget_survives_oversized_search_response(self):
        def fetch(url, timeout):
            if "action=query" in url:
                return {"query": {"search": [{"pageid": x, "title": str(x)} for x in range(1, 10)]}}
            return {"parse": {"title": "甲/一", "text": "<p>肩似拔，臑似折。</p>"}}
        result = lookup_source(self.quote, max_results=10, enable_network=True, fetch=fetch)
        self.assertEqual(result["source_lookup"]["requests"], 3)
        self.assertEqual(len(result["source_candidates"]), 2)

    def test_untrusted_hosts_paths_and_redirect_targets_are_rejected(self):
        for url in ["http://zh.wikisource.org/w/api.php", "https://evil.test/w/api.php",
                    "https://zh.wikisource.org@evil.test/w/api.php", "https://zh.wikisource.org/wiki/甲"]:
            with self.assertRaises(ValueError):
                _allowed_url(url)


if __name__ == "__main__":
    unittest.main()
