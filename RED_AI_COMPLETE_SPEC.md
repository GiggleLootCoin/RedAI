# RedAI Complete Build Specification

## Goal

RedAI is a complete personal AI platform, not a demo chat screen.

## Core identity

- Created by LittleRedBigSmile.
- Created with love; love is part of the foundational identity and purpose.
- Consistent core identity for every user.
- Each user gets an independent relationship, memory, preferences, and project context.
- Never claim capabilities, memories, files, searches, or actions that did not actually occur.

## Local-first architecture

- Qwen3-1.7B Q4_K_M local GGUF remains the Galaxy A12 baseline.
- Local inference works without an account or paid API.
- Persistent per-user memory is stored separately from model weights.
- Online tools are optional enhancements, never silently required for basic chat.

## Multimodal capabilities

- Attach and inspect supported files.
- Camera capture and vision workflow.
- On-device OCR for images and camera frames.
- PDF/document handling.
- Audio transcription and speech output.
- Image generation through a provider abstraction when online generation is available.
- Clear capability states so the UI never presents an unavailable tool as working.

## Live conversation

- Hands-free voice mode.
- Fast speech recognition.
- Streaming/early response behavior where supported.
- Speech output begins as soon as a usable response is available.
- User can interrupt speech and continue listening.
- Offline recognition is preferred when the device supports it.

## Memory

- Separate memory namespace per user.
- Remember only information appropriate for long-term use.
- User controls: inspect, add, correct, forget, clear all.
- Project-scoped memory is separate from global user memory.
- No cross-user memory leakage.

## UI

- RedAI branding.
- No meaningless timing/status panel.
- Main conversation screen.
- Attach, camera, microphone, send, stop, copy/share.
- Voice mode control.
- Web control.
- Memory control.
- Projects/workspaces.
- Settings and capability status.
- Clear error states.
- Responsive design for Galaxy A12-class screens.

## Build/test gates

Every feature must pass these gates before being called complete:

1. Source reviewed for the intended behavior.
2. Implementation added without fake controls.
3. GitHub Actions build succeeds.
4. APK contains expected classes/resources.
5. Feature-specific automated or deterministic checks pass where possible.
6. Regression checks cover model loading and normal chat.
7. Release artifact is verified before it is described as finished.

## Important constraint

No feature is considered complete merely because its button exists. A feature is complete only when the underlying action works or the UI explicitly reports that the required capability is unavailable.
