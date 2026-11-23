# RedAI

RedAI is an Android AI app created and personalized by **Red (LittleRedBigSmile)**. It combines a bundled local language model, an Android interface, private on-device chat history, Red's Ways Of Thinking, and optional online tools.

## Current development build: Android 5.5

- Foundation: MiniCPM5-1B Q4_K_M GGUF, bundled by the GitHub Actions build.
- Personalization: RedAI fine-tuned candidate plus Red's Ways Of Thinking framework.
- Runtime: SmolChatAndroidLib V2.1 / llama.cpp, ARM64 CPU runtime, streamed output.
- Runtime profile: 512-token context, 2 CPU threads, temperature 0.70, top-p 0.95, top-k 40, repeat penalty 1.08.
- Model load is user-initiated and now requires at least 1,408 MB of available system RAM before native loading; inference context is capped at 512 tokens to reduce memory pressure.
- Up to 30 local conversation sessions, with switching and deletion.
- Optional DuckDuckGo/Wikipedia search and optional Tavily search.
- Local text-file reading, PDF text extraction (up to five pages), and image OCR.
- Android system speech recognition (tap-to-talk) and installed Android TTS voices.
- Optional image generation through external AI Horde volunteer workers.

## What works and what does not

| Capability | Current state |
|---|---|
| Bundled local GGUF and ARM64 inference | Included; CI verifies model checksum and runtime smoke test |
| Local chat and saved sessions | Implemented |
| Red's Ways Of Thinking context | Included in runtime prompts |
| Optional web search | Implemented; requires internet |
| Text/PDF/image OCR | Implemented on-device |
| Camera capture | Captures a photo; analysis is OCR-only |
| Microphone | Tap-to-talk through Android speech recognition |
| Speech output | Android TTS voices already installed on the phone |
| Custom cloned Red voice | Not integrated |
| Continuous hands-free conversation | Not implemented |
| Full visual scene understanding | Not implemented; OCR only |
| Local/offline image generation | Not implemented; AI Horde sends the prompt to external volunteer workers |
| Galaxy A12 performance validation | Requires testing on the physical handset |

Do not interpret a successful GitHub Actions build as proof of physical-device stability, speed, microphone compatibility, thermal performance, or low-memory reliability. Those require a real Android 12 handset test.

## Privacy and network behavior

- Normal local model inference and saved conversations remain on the device.
- Web lookup is optional and only used for requests that need current information or explicitly ask for research. The latest question is sent to the selected search provider.
- Image generation is a separate online feature. Its prompt is sent to AI Horde; it is not private on-device generation.
- Microphone input uses the Android speech-recognition service. Offline recognition depends on the installed service and its language packs.
- Speech output uses the Android TTS engine and its installed voices. No custom Red voice pack is bundled.

## Build and verification

GitHub Actions downloads the exact GGUF candidate, verifies its SHA-256 and GGUF header, runs a llama.cpp smoke test, builds the ARM64 Android APK, checks the APK contents and source-level regression conditions, and verifies the resulting APK checksum.

The current workflow uses Gradle's **debug build variant**. The APK is debug-signed for installation/testing; this is not a production release-signing configuration. The v5.5 artifact is available from the Actions run that produced it.

## Creator and foundation-model attribution

RedAI's product identity, personalization, application code and Red's Ways Of Thinking integration are created by Red aka LittleRedBigSmile. The underlying MiniCPM5 foundation model is third-party software/model content and is not claimed as authored by Red.

## Next release gates

1. Complete feature implementations that are currently listed as unavailable.
2. Add deterministic tests for each new capability and preserve model/chat regression coverage.
3. Test cold start, offline chat, long sessions, memory pressure, audio input/output, attachments, and recovery on the target Galaxy A12.
4. Configure and validate a production signing/update path before publishing a production release.

See [WEB_SEARCH.md](WEB_SEARCH.md) for search-provider behavior and [RED_AI_COMPLETE_SPEC.md](RED_AI_COMPLETE_SPEC.md) for the product acceptance requirements.
