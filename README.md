# RedAI

Standalone local AI model for RedUI.

## Current build target

- Foundation: **Qwen3-1.7B**
- Personalization: **Red's Ways Of Thinking**
- Fine-tuning: **LoRA with assistant-completion-only loss**
- Quantization: **GGUF Q4_K_M**
- Runtime target: **Android / Galaxy A12-class hardware**
- Design goal: **free, offline-first local inference**, with optional external tools only when explicitly invoked

## What was improved

The training pipeline is deliberately conservative because the personalization corpus is small. RedAI now:

- trains only on assistant completion tokens instead of wasting capacity learning to reproduce system/user prompts;
- uses the Qwen chat template consistently for training and llama.cpp verification;
- keeps the foundation at Qwen3-1.7B, which is a stronger practical fit for the Galaxy A12 than moving to a much larger model;
- preserves Red's Ways Of Thinking as personalized material while teaching the model to distinguish framework/perspective from independently verified fact;
- includes identity, correction, honesty, tool-use, privacy, planning, math, ambiguity, and failure-reporting anchors;
- verifies non-empty output, identity, creator, normal chat, math, capabilities, app/project naming, current-information honesty, and hidden response-planning leakage before packaging.

## Identity

RedAI is created by Red, also known as Little Red Big Smile. RedAI does not claim ownership of the underlying Qwen foundation model.

## Recommended local runtime

- Context: **2048 tokens** for the Galaxy A12 preset
- Temperature: **0.70**
- Top-p: **0.80–0.90**
- Top-k: **20–40**
- Repeat penalty: **1.10**
- Max output: **512 tokens**
- CPU threads: **4**
- Memory mapping: **enabled**
- Flash attention: **when supported**
- GPU layers: **0 by default for stability**

The model's underlying Qwen3-1.7B context capacity is larger, but the 2048-token local preset is intentional: it reduces memory pressure and keeps the phone-focused runtime practical.

## Free build

RedAI is built from the repository using GitHub Actions and open-source tooling. No paid API, paid model host, subscription, or proprietary inference service is required to build or run the GGUF locally.

## Release

The redai-latest GitHub Release contains the verified RedAI.gguf package and SHA-256 checksum. The release is rebuilt whenever the main training/build pipeline is deliberately updated.
