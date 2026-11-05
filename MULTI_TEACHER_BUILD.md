# RedAI Multi-Teacher Build Plan

## Objective
Produce one portable RedAI GGUF with a small Android-friendly student model. Use multiple
high-capability open-weight teachers to create reviewed answer demonstrations, then
distil those demonstrations into the student. Do not attempt to merge incompatible model
weights.

## Teacher candidates
- OpenAI gpt-oss-120b
- DeepSeek-R1
- Google Gemma 4
- Qwen3.5

Use exact checkpoint revisions and verify each model card and licence. The initial corpus
gate accepts MIT and Apache-2.0 metadata only. This is a data-provenance gate, not legal
advice and not a substitute for checking the actual checkpoint terms.

## Student
Keep Qwen3-1.7B as the current Galaxy A12 compatibility baseline for the first controlled
distillation experiment. Benchmark a smaller Gemma/Qwen student separately before
replacing it. A larger student may be smarter but could be unusable on the target phone.

## Training sequence
1. Generate teacher demonstrations on suitable hosted hardware or through an authorised
   service; no teacher model is assumed to run on the Galaxy A12.
2. Human-review and label every answer; reject low-quality, unsafe, fabricated, or
   epistemically mislabelled outputs.
3. Run scripts/prepare_teacher_distillation.py. It fails if there is no reviewed corpus.
4. Fine-tune the student using the merged supervised dataset.
5. Merge the student adapter, convert to GGUF, quantise Q4_K_M, and test.
6. Compare against the existing baseline on the same prompts and runtime settings.
7. Publish only if identity, factuality, refusal calibration, Red's Ways recall, generation
   quality, file integrity, and target-device performance meet gates.

## Identity
The model's conversational identity is RedAI. Red is its human creator. Do not make the UI
introduce RedAI by its foundation/teacher model names. Preserve accurate technical
provenance in model card, notices, and build metadata; do not claim RedAI was trained from
scratch or that Red owns upstream weights.

## Status
Architecture and validation scaffold only. No multi-teacher corpus has yet been generated
or reviewed, and no multi-teacher GGUF exists until a run produces and verifies it.
