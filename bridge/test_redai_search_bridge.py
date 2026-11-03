import importlib.util
import os
import unittest
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
        ):
            result = bridge.search("test topic", "duckduckgo")
        self.assertTrue(result["ok"])
        self.assertEqual(result["results"][0]["title"], "Test")

    def test_tavily_requires_user_supplied_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "TAVILY_API_KEY"):
                bridge.tavily("test")

    def test_wikipedia_provider_is_no_key(self):
        with patch.object(bridge, "wikipedia", return_value=[{"title": "Article", "url": "https://en.wikipedia.org/wiki/Article"}]):
            result = bridge.search("article", "wikipedia")
        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "wikipedia")


if __name__ == "__main__":
    unittest.main()
