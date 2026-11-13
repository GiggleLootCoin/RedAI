# RedAI v4 — Product Redesign Blueprint

## Product definition

RedAI is one complete Android application: the interface, RedAI identity, selected foundation model, local inference runtime, voice, conversation storage, file understanding and optional web research belong to one product. It is not a separate RedUI shell that asks its owner to find and install a brain.

Creator identity: **Created and developed by Red (LittleRedBigSmile)**. RedAI is its own assistant identity, personalised using Red's Ways Of Thinking and the stated foundation of Love, Truth, care and respect. It must never claim that Red created third-party foundation weights.

## Non-negotiable product behaviour

- First launch opens into a ready-to-chat experience. No GGUF picker, model upload, model-store UI, empty-model state or disabled Send button caused by a missing user-supplied model.
- The chosen, verified GGUF foundation is packaged with the product and selected automatically. The model is not considered bundled until the build verifies the exact model asset, SHA-256, GGUF magic, architecture/tokenizer metadata and load test.
- Keep the existing local llama.cpp ARM64 runtime and offline-first behaviour. No mandatory accounts, cloud inference, subscriptions, paid APIs, Colab, Kaggle or external model-host dependency at runtime.
- Optional web research is a clearly separate, user-controlled capability. Chat remains local when offline. Search sources and uncertainty are shown honestly.
- Keep premium black and vivid-red visual identity, readable cards, refined typography, voice/live conversation, chat history, file/image understanding, settings, theme controls, creator/channel link and Red's Ways Of Thinking. Remove technical model-management controls from ordinary user-facing screens.
- Keep Android 12 / ARM64 compatibility as the minimum device target, with specific Galaxy A12 memory, heat and latency testing before calling it device-ready.
- No changes to Little Red's Big Studio or RED-LIVE.

## Proposed app architecture

1. **PackagedBrain** — immutable model manifest identifying exact GGUF filename, architecture, tokenizer, context limit, quantisation, checksum, licence and provenance. Build fails if the asset or checksum is missing.
2. **BrainBootstrap** — verifies the packaged asset on first launch, copies it atomically into app-private storage only if the runtime requires a filesystem path, loads it, and reports actionable failures. It never silently substitutes a different model.
3. **InferenceService** — one lifecycle-managed llama.cpp session, bounded context and thread settings for low-memory ARM64 devices, cancellation, OOM handling and deterministic release.
4. **ConversationStore** — private on-device chat history, per-chat deletion and explicit clear-all.
5. **VoiceService** — microphone capture, speech recognition availability checks, interruption-safe TTS and a hands-free state machine. Do not label a typed response as spoken.
6. **AttachmentService** — text/PDF/image handling with size limits and clear unsupported-format feedback.
7. **ResearchService** — optional web lookup, citations and offline fallback; never sends chat history when only the latest query is needed.
8. **RedAI UI** — Chat, Talk Live, History and Settings. No GGUF upload or model picker in the customer-facing product.

## Packaging decision and release gate

The exact GGUF is not currently present in the RedAI repository tree. Therefore the existing APK is a runtime shell, not the complete product requested. Do not remove the picker until the verified model has been added to the build inputs. Bundle the chosen model into the installable distribution (APK or a signed, checksum-verified companion package delivered with the release); the user must not have to locate a random model online. The release pipeline must verify the model is inside the published distribution and that a clean install can load it without network access.

Before final release, run CI checks for asset presence, checksum, model metadata, load smoke test, UI state and APK contents; then test cold start, offline chat, long conversation, voice, file attachment, history deletion and memory pressure on a physical Galaxy A12.

## Acceptance criteria

- Install once; open RedAI; the packaged brain is already selected and ready.
- Send works immediately after successful first-load verification.
- Airplane-mode chat works.
- No GGUF/index upload or model-management workflow is visible.
- No unsupported feature is represented as working.
- Model identity and creator identity are accurately distinguished.
- A failed model load gives a clear recovery message and does not leave a dead chat screen.