# RedAI

Standalone local AI model for RedUI.

## Current build target

- Foundation: **MiniCPM5-1B Q4_K_M**
- Personalization: **Red's Ways Of Thinking**
- Personalization training: **RedAI MiniCPM5-1B LoRA, merged and quantized to GGUF Q4_K_M**
- Quantization: **GGUF Q4_K_M**
- Runtime: **SmolChatAndroidLib V2.1 / llama.cpp, ARM64 CPU feature detection, streaming**
- Runtime target: **Android 12 / Galaxy A12-class hardware**
- Design goal: **free, offline-first local inference**, with optional external tools only when explicitly invoked

## What was improved

The training pipeline is deliberately conservative because the personalization corpus is small. RedAI now:

- trains only on assistant completion tokens instead of wasting capacity learning to reproduce system/user prompts;
- uses the Qwen chat template consistently for training and llama.cpp verification;
- uses the separately trained RedAI MiniCPM5-1B checkpoint, with Red's identity and framework material included in the fine-tuning corpus and runtime prompt;
- preserves Red's Ways Of Thinking as personalized material while teaching the model to distinguish framework/perspective from independently verified fact;
- includes identity, correction, honesty, tool-use, privacy, planning, math, ambiguity, and failure-reporting anchors;
- verifies non-empty output, identity, creator, normal chat, math, capabilities, app/project naming, current-information honesty, and hidden response-planning leakage before packaging.

## Identity

RedAI is created and personalized by Red, also known as LittleRedBigSmile. The user-facing identity is RedAI; the human creator is Red. The model does not claim ownership of the underlying foundation weights.

## Recommended local runtime

- Context: **1536 tokens** for the Galaxy A12 preset
- Temperature: **0.70**
- Top-p: **0.80–0.90**
- Top-k: **20–40**
- Repeat penalty: **1.10**
- Output: **104 pieces for ordinary replies; up to 176 for explicitly detailed replies**
- CPU threads: **4**, with actual throughput displayed in the app
- Memory mapping: **enabled**
- Flash attention: **when supported**
- GPU layers: **0 by default for stability**

The bundled MiniCPM5-1B Q4_K_M checkpoint is approximately 688 MB and was selected as the stronger personalized 1B-class candidate. Its no-think template avoids spending the phone's limited CPU budget on hidden reasoning for ordinary chat. Short replies are capped to reduce unnecessary decoding; longer limits are used only when the request calls for detail. The app streams generated text and reports measured tokens per second. Actual Galaxy A12 speed still requires on-device measurement.

## Free build

RedAI is built from the repository using GitHub Actions and open-source tooling. No paid API, paid model host, subscription, or proprietary inference service is required to build or run the GGUF locally.

## Android v5 runtime

The Android app uses the Apache-2.0 SmolChatAndroidLib V2.1 runtime, which selects an ARM64 native build based on CPU features and supports streamed token output. The bundled GGUF is the fine-tuned RedAI MiniCPM5-1B candidate, not the unpersonalized Qwen3.5-0.8B test model. The app keeps the model bundled and local, restores a short recent conversation, and uses a smaller relevant subset of Red's Ways Of Thinking instead of inserting all 38 statements into every related prompt. The app displays measured generation speed. A successful CI build is not a substitute for testing on the physical Galaxy A12.

## Optional live web search

The GGUF remains fully local and independent of web access. The RedAI Android app includes an optional search integration with no-key DuckDuckGo and Wikipedia providers, plus an optional Tavily provider (current free tier: 1,000 credits/month, no credit card). Search runs only for explicit research/verification or current-information requests. Only the newest user question is sent to the provider; ordinary local chat remains local.

See [WEB_SEARCH.md](WEB_SEARCH.md) for provider costs, privacy behavior, Android setup, the standalone bridge for compatible tool-calling frontends, and the ChatterUI limitation. The independent bridge is in [bridge/](bridge/).

## Release

The redai-latest GitHub Release contains the verified RedAI.gguf package and SHA-256 checksum. The release is rebuilt whenever the main training/build pipeline is deliberately updated. Web search is provided by the app/bridge, not baked into or required by the GGUF.
