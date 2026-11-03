import importlib.util
import os
import unittest
import json
import threading
import urllib.request
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "redai_search_bridge", os.path.join(os.path.dirname(__file__), "redai_search_bridge.py")
)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


class BridgeTests(unittest.TestCase):
    def test_empty_query_rejected(self):
        with self.assertRaises(ValueError):
            bridge.search("   ", "duckduckgo")

    def test_safe_url_requires_public_https(self):
        self.assertTrue(bridge.safe_public_url("https://example.com/page"))
        self.assertFalse(bridge.safe_public_url("http://example.com"))
        self.assertFalse(bridge.safe_public_url("https://127.0.0.1/admin"))
        self.assertFalse(bridge.safe_public_url("https://localhost/"))

    def test_result_parser_extracts_title_url_and_snippet(self):
        parser = bridge.ResultParser()
        parser.feed(
            '<a class="result__a" href="https://example.com/story">Example Story</a>'
            '<a class="result__snippet">A useful excerpt from the page.</a>'
        )
        self.assertEqual(len(parser.results), 1)
        self.assertEqual(parser.results[0]["title"], "Example Story")
        self.assertEqual(parser.results[0]["url"], "https://example.com/story")
        self.assertIn("useful excerpt", parser.results[0]["excerpt"])

    def test_duckduckgo_falls_back_to_wikipedia(self):
        with patch.object(bridge, "duckduckgo", side_effect=RuntimeError("blocked")), patch.object(
            bridge, "wikipedia", return_value=[{"title": "Test", "url": "https://en.wikipedia.org/wiki/Test", "excerpt": "text"}]
        ), patch.object(bridge, "enrich_results", side_effect=lambda items: items):
            result = bridge.search("test topic", "duckduckgo")
        self.assertTrue(result["ok"])
        self.assertEqual(result["results"][0]["title"], "Test")

    def test_tavily_requires_user_supplied_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "TAVILY_API_KEY"):
                bridge.tavily("test")

    def test_http_search_endpoint_returns_tool_result(self):
        server = bridge.ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = "http://127.0.0.1:%d/v1/search" % server.server_address[1]
            payload = json.dumps({"query": "current test", "provider": "wikipedia"}).encode()
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            expected = {"ok": True, "query": "current test", "results": [{"title": "A source"}]}
            with patch.object(bridge, "search", return_value=expected):
                with urllib.request.urlopen(req, timeout=3) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(json.loads(response.read()), expected)
            with urllib.request.urlopen("http://127.0.0.1:%d/" % server.server_address[1], timeout=3) as response:
                self.assertIn(b"RedAI", response.read())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_wikipedia_provider_is_no_key(self):
        with patch.object(bridge, "wikipedia", return_value=[{"title": "Article", "url": "https://en.wikipedia.org/wiki/Article"}]), patch.object(
            bridge, "enrich_results", side_effect=lambda items: items
        ):
            result = bridge.search("article", "wikipedia")
        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "wikipedia")


if __name__ == "__main__":
    unittest.main()
