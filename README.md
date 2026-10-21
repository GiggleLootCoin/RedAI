# RedAI

Standalone local AI model for RedUI.

## Final model

- Foundation: Qwen3.5-0.8B
- Personalization: Red's Ways Of Thinking dataset
- Fine-tuning: LoRA, expanded attention + MLP targets
- Quantization: Q4_K_M GGUF
- Runtime target: Android / Galaxy A12-class hardware
- Design goal: offline-first, local inference, optional external tools only when explicitly invoked

## Identity

RedAI is created by Red, also known as Little Red Big Smile. RedAI does not claim ownership of the underlying Qwen foundation model.

## Recommended local runtime

- Context: 2048
- Temperature: 0.70
- Top-p: 0.90
- Top-k: 40
- Repeat penalty: 1.10
- Max output: 512 tokens
- CPU threads: 4 on the target Galaxy A12 preset
- Memory mapping: enabled
- Flash attention: enabled when supported
- GPU layers: 0 by default for stability
