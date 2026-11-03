# RedAI live web search

## Architecture

The GGUF remains a local language model. It does not contain a search engine, API key, browser, or internet connection. RedAI Android v3.2 calls a separate, lightweight search service only when a message clearly asks for research/verification or needs fresh information. Ordinary conversation and creative writing stay local.

The Android implementation is in:
- `android-ui/app/src/main/java/com/giggleloot/redai/WebSearchService.kt`
- `android-ui/app/src/main/java/com/giggleloot/redai/MainActivity.kt`

The service returns structured source titles, URLs, excerpts and publication dates where supplied. It extracts additional text from the top accessible result page with bounded time/size limits. Webpage content is always treated as untrusted evidence, not instructions. Search context is bounded for mobile memory. Only the newest user message is sent to the provider; recent chat history and attachments are not sent.

## Providers and cost

| Provider | Key required | Cost / limits | Role |
|---|---:|---|---|
| DuckDuckGo HTML results | No | No API charge; unofficial HTML interface, best-effort and subject to markup changes or blocking | Default broad-web option |
| Wikipedia REST search | No | Free public API; rate limits and usage policies apply | No-key fallback; encyclopedia coverage only |
| Tavily Search API | Yes | Current free Researcher tier: 1,000 API credits/month, no credit card; requests stop when the free allowance is exhausted | Optional structured search and excerpts |
| Brave Search API | Yes | Not selected as the zero-cost default: current $5 monthly credits require account/card verification; paid prepaid usage thereafter | Optional alternative, not required |

Tavily key setup is optional. In the Android app it is encrypted with an Android Keystore AES-GCM key and is not written into prompts, logs, or the repository. The app does not require Tavily or any paid service. DuckDuckGo and Wikipedia work without a key, though neither provides a guaranteed uptime or service-level agreement.

Provider terms and limits can change. Check the provider's current dashboard before relying on a free allowance.

## Android setup (RedAI app)

1. Install the RedAI Android APK from the repository's `redai-android` release after the Android workflow passes.
2. Load the existing RedAI GGUF as usual. Search does not replace or rebuild the model.
3. Open **CONTROL**. Live search is enabled by default; switch it off any time.
4. Cycle the provider setting between DuckDuckGo, Wikipedia and Tavily.
5. For Tavily, create a free key at Tavily, paste it into the optional key field, and save. Keep DuckDuckGo or Wikipedia selected if you do not want an account or key.
6. Ask for current information, or explicitly say “search the web”, “research”, “verify”, or “investigate”. RedAI reports when it is offline, disabled, blocked, or has no usable results.

The Android app makes no search requests for ordinary chat or creative writing. There is no background polling.

## Standalone bridge for other GGUF frontends

A dependency-free Python bridge is provided at `bridge/redai_search_bridge.py`. It does not load a model and does not need a second LLM. It exposes:
- `GET /` — local browser search page
- `GET /health`
- `POST /v1/search` with JSON `{"query":"...","provider":"duckduckgo"}`

By default it binds only to `127.0.0.1:8765`. It does not log queries or credentials. Non-local binding is refused unless `RED_AI_BRIDGE_TOKEN` is configured.

### Android / Termux

Install Termux from its official F-Droid or GitHub distribution, then run:

```sh
pkg update -y
pkg install -y python git
git clone https://github.com/GiggleLootCoin/RedAI.git
cd RedAI/bridge
python redai_search_bridge.py
```

The bridge starts on `http://127.0.0.1:8765`. Open that address in Brave to use its small local search page, choose a provider, search, open source links, and copy the results into a chat. This is the practical fallback for a frontend that cannot execute tools. Keep Termux open while the page is in use; stop it with Ctrl+C. This is optional; the RedAI and RedUI apps have built-in search and do not need Termux.

Optional Tavily provider:

```sh
read -s TAVILY_API_KEY
export TAVILY_API_KEY
python redai_search_bridge.py
```

Do not paste the key into a public repository or share it. DuckDuckGo and Wikipedia require no key.

### Generic function/tool schema

For a frontend that supports application-side function calling, register a function named `search_web`:

```json
{
  "type": "function",
  "function": {
    "name": "search_web",
    "description": "Search live public web sources when current information, research or verification is needed. Send only the user's search query.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string"},
        "provider": {"type": "string", "enum": ["duckduckgo", "wikipedia", "tavily"]}
      },
      "required": ["query"],
      "additionalProperties": false
    }
  }
}
```

The exact schema is also supplied as [search_web_tool.json](bridge/search_web_tool.json). When the model requests `search_web`, the host application—not the GGUF—must POST the function arguments to `/v1/search`, then return the JSON result as the tool output and let the model answer. The bridge endpoint is a search tool, not an OpenAI-compatible chat-completions server.

## ChatterUI compatibility

ChatterUI can load GGUF files locally, but its published documentation describes local inference and API backends without documenting a local-mode tool/function-call executor. A model cannot make HTTP requests by itself. Therefore:
- RedAI's own Android app and RedUI can search automatically because the app code performs the lookup.
- ChatterUI Local Mode continues to run the GGUF offline; it cannot automatically invoke this bridge. ChatterUI's public tool-calling/MCP feature request (#535) was closed as “not planned” by its maintainer on 8 July 2026. This is an app limitation, not a GGUF limitation.
- A frontend that supports external tool/function execution can call the standalone bridge using the schema above.
- ChatterUI Remote Mode can connect to chat-completion backends, but the search bridge is not itself a chat-completion backend. Do not point ChatterUI's model endpoint at `/v1/search`.

No universal GGUF-interface compatibility is claimed.

## Verification and remaining limits

The Android build verifies compilation, native llama.cpp libraries, app identity and presence of the search integration. The bridge has isolated unit tests for routing/parsing and safe URL handling. A successful APK build does not prove live-provider uptime, real device connectivity, or compatibility with a third-party app's private tool implementation. Those require on-device testing.

DuckDuckGo's HTML page is not a stable developer API and may block automation. Wikipedia is not a substitute for full-web or breaking-news search. Tavily is the more structured option when its free monthly credits are available. Search results can be wrong, incomplete or conflicting; important claims should be checked against the linked pages.
