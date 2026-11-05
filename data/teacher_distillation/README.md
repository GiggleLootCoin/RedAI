# RedAI multi-teacher distillation corpus

This folder contains reviewed teacher responses used to transfer useful capabilities into
one smaller RedAI student model. Teacher weights are **not** merged into one another.

## Required JSONL record

Each line must be a JSON object:

```json
{
  "teacher": {
    "model_id": "publisher/exact-checkpoint",
    "license": "apache-2.0",
    "model_card_url": "https://huggingface.co/publisher/exact-checkpoint"
  },
  "review_status": "approved",
  "quality_score": 0.95,
  "epistemic_label": "verified_fact",
  "messages": [
    {"role": "user", "content": "A useful user question"},
    {"role": "assistant", "content": "A concise, accurate answer"}
  ]
}
```

Allowed licence metadata for this initial pipeline: Apache-2.0 and MIT. Verify the exact
checkpoint's model card and licence before adding data. Do not assume every checkpoint in
a model family has the same terms.

Allowed epistemic labels: `verified_fact`, `inference`, `red_framework`, `creative`,
and `uncertain`. Red's Ways Of Thinking must be labelled `red_framework` where it
expresses Red's personal framework rather than independently established facts.

## Teacher shortlist to evaluate

These are candidates, not claims that their answers have already been generated or used:

- OpenAI `gpt-oss-120b` — broad reasoning and instruction-following teacher.
- DeepSeek `DeepSeek-R1` — reasoning and problem-solving teacher.
- Google Gemma 4 — general reasoning, language and multimodal capability teacher.
- Qwen3.5 — coding, multilingual and general-purpose teacher.

The exact model card, checkpoint version, licence and any usage terms must be recorded for
each corpus row. Proprietary GPT/Claude API outputs are not included by default: their
service terms and costs need separate review before using them as training data.

## Quality and safety gates

- Only explicitly human-reviewed `approved` records are used.
- Default minimum quality score is 0.85.
- No hidden chain-of-thought or private reasoning traces; train concise answers and useful
  explanations instead.
- No invented citations, tool calls, memory, or capabilities.
- Separate verified facts, inference, uncertainty, and Red's framework.
- Health, legal, financial, and safety-critical examples need domain-appropriate review.
- Teacher provenance stays in training metadata and technical documentation. RedAI's
  conversational identity remains RedAI; it must not falsely claim that it created its
  own foundation weights.

The preparation script fails closed when no approved multi-teacher examples exist. This
prevents a single-foundation run from being mislabelled as a multi-teacher build.
