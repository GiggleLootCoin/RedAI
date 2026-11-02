# RedAI multi-teacher distillation input

RedAI does not merge the private weights of GPT or Claude into a GGUF. The hybrid approach is **teacher-response distillation**: several teacher models answer the same carefully chosen prompts, a human approves the strongest answer (or writes an adjudicated answer), and the student model learns from the approved response.

The MiniCPM5 candidate training script optionally reads `data/teacher_responses.jsonl`.

Each line must be a JSON object:

```json
{"teacher":"Gemma","prompt":"How should an assistant evaluate a controversial claim?","response":"A clear, evidence-aware answer...","quality_score":4.7,"approved":true,"adjudicated_response":""}
```

Accepted teacher labels include Qwen, Gemma, DeepSeek, GPT and Claude. These labels are provenance only; they do not imply access to proprietary model weights.

Rules:
- `approved` must be `true`.
- `quality_score` must be at least 4.0 on a 1–5 scale.
- The training script keeps one highest-scored approved answer per prompt.
- If `adjudicated_response` is non-empty, that becomes the student target.
- Do not paste private conversations, personal data, or copyrighted material that you do not have permission to use.
- The current candidate build has no teacher-response file yet, so it is trained on the RedAI dataset and the canonical 38-statement Red's Ways Of Thinking source. It must not be described as a completed multi-teacher hybrid until approved teacher examples are added and the model is rebuilt.

GPT and Claude are proprietary services: their model weights cannot be downloaded and merged. Their outputs can only be used as text supervision when the applicable service terms permit it.
