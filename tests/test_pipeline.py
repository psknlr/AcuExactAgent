import json
from pathlib import Path
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

from acuexact.pipeline import demo
from acuexact.evaluation import evaluate
from acuexact.server import Handler


class PipelineTests(unittest.TestCase):
    def test_demo_gold_is_independent_and_exact(self):
        result = demo()
        gold = json.loads((Path(__file__).parent.parent / "acuexact/data/demo-gold.json").read_text(encoding="utf-8"))
        score = evaluate(result["quotes"], gold)
        self.assertEqual((score["true_positive"], score["false_positive"], score["false_negative"]), (8, 0, 0))
        self.assertFalse(result["audit"]["gold_used_in_inference"])
        self.assertFalse(any("邻文引用" in q["text"] for q in result["quotes"]))

    def test_local_server_roundtrip_and_cross_origin_rejection(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/api/demo", timeout=5) as response:
                self.assertEqual(len(json.load(response)["quotes"]), 8)
            data = json.dumps({"text": '“肩痛”“必须保留。”', "article_id": "http-test", "article_title": "HTTP测试"}).encode()
            request = Request(base + "/api/analyze", data=data, headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=5) as response:
                result = json.load(response)
                self.assertEqual([q["text"] for q in result["quotes"]], ["必须保留。"])
            request.add_header("Origin", "https://unrelated.example")
            with self.assertRaises(HTTPError) as context:
                urlopen(request, timeout=5)
            self.assertEqual(context.exception.code, 403)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
