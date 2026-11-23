#!/usr/bin/env python3
"""Validate and merge human-reviewed multi-teacher answers into RedAI's SFT corpus.

Teacher generations are never trusted automatically. Every row must carry explicit
model/license provenance, a human approval flag, a quality score, and an epistemic label.
This script intentionally does not generate teacher answers or expose private chain-of-thought.
"""
import glob
import hashlib
import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_GLOB = os.path.join(ROOT, "data", "teacher_distillation", "*.jsonl")
BASE = os.path.join(ROOT, "data", "red_ai_dataset.jsonl")
MIN_SCORE = float(os.getenv("REDAI_TEACHER_MIN_SCORE", "0.85"))
ALLOWED_LICENSES = {"apache-2.0", "mit"}
ALLOWED_LABELS = {"verified_fact", "inference", "red_framework", "creative", "uncertain"}
BAD_OUTPUT = re.compile(
    r"(let me think step by step|here is my hidden chain|as an ai language model|"
    r"the user asked me to|i need to respond|let me craft)",
    re.I,
)

def fail(message):
    print("TEACHER_DATA_ERROR: " + message, file=sys.stderr)
    raise SystemExit(2)

def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append((line_no, json.loads(line)))
            except json.JSONDecodeError as exc:
                fail(f"{path}:{line_no}: invalid JSON: {exc}")
    return rows

def normalized(text):
    return re.sub(r"\s+", " ", text).strip().casefold()

def main():
    sources = sorted(glob.glob(INPUT_GLOB))
    if not sources:
        fail("No teacher corpus found. Add reviewed *.jsonl files under data/teacher_distillation/; refusing to label a single-teacher run as multi-teacher.")

    accepted = []
    seen = set()
    for path in sources:
        for line_no, row in read_jsonl(path):
            where = f"{path}:{line_no}"
            if row.get("review_status") != "approved":
                continue
            try:
                score = float(row.get("quality_score", 0))
            except (TypeError, ValueError):
                score = 0
            if score < MIN_SCORE:
                continue
            teacher = row.get("teacher") or {}
            model_id = teacher.get("model_id")
            license_id = str(teacher.get("license", "")).lower()
            model_card = teacher.get("model_card_url", "")
            if not model_id or license_id not in ALLOWED_LICENSES or not model_card.startswith("https://"):
                fail(f"{where}: missing model ID, HTTPS model card, or permissive MIT/Apache-2.0 licence metadata")
            label = row.get("epistemic_label")
            if label not in ALLOWED_LABELS:
                fail(f"{where}: epistemic_label must be one of {sorted(ALLOWED_LABELS)}")
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                fail(f"{where}: messages must contain at least user and assistant turns")
            clean = []
            for message in messages:
                if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
                    fail(f"{where}: only user/assistant messages are accepted")
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    fail(f"{where}: empty or invalid message content")
                if message["role"] == "assistant" and BAD_OUTPUT.search(content):
                    fail(f"{where}: response contains chain-of-thought/planning leakage or boilerplate")
                clean.append({"role": message["role"], "content": content.strip()})
            if clean[-1]["role"] != "assistant" or not any(m["role"] == "user" for m in clean):
                fail(f"{where}: conversation must end with an assistant answer and contain a user turn")
            key = hashlib.sha256(
                (normalized(clean[-2]["content"]) + "\n" + normalized(clean[-1]["content"])).encode("utf-8")
            ).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            accepted.append({
                "messages": clean,
                "source_tag": "multi_teacher_distillation",
                "teacher_provenance": {
                    "model_id": model_id,
                    "license": license_id,
                    "model_card_url": model_card,
                },
                "epistemic_label": label,
            })

    distinct_teachers = {r["teacher_provenance"]["model_id"] for r in accepted}
    if len(distinct_teachers) < 2:
        fail(f"At least two distinct reviewed teacher models are required; found {len(distinct_teachers)}.")
    if not accepted:
        fail(f"No approved rows met the quality threshold of {MIN_SCORE:.2f}; no training data was changed.")

    base_rows = read_jsonl(BASE)
    existing = [row for _, row in base_rows]
    existing_keys = set()
    for row in existing:
        msgs = row.get("messages", [])
        if len(msgs) >= 2:
            existing_keys.add(hashlib.sha256(
                (normalized(msgs[-2].get("content", "")) + "\n" + normalized(msgs[-1].get("content", ""))).encode("utf-8")
            ).hexdigest())
    fresh = [row for row in accepted if hashlib.sha256(
        (normalized(row["messages"][-2]["content"]) + "\n" + normalized(row["messages"][-1]["content"])).encode("utf-8")
    ).hexdigest() not in existing_keys]

    if not fresh:
        fail("All approved teacher rows already exist in the base corpus; refusing a misleading no-op build.")
    fd, temp_path = tempfile.mkstemp(prefix="redai-dataset-", suffix=".jsonl", dir=os.path.dirname(BASE))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for row in existing + fresh:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temp_path, BASE)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    print(f"RED_AI_MULTI_TEACHER_ROWS_ACCEPTED {len(fresh)}")
    print(f"RED_AI_MULTI_TEACHER_MODELS {len({r['teacher_provenance']['model_id'] for r in fresh})}")
    print(f"RED_AI_TOTAL_SFT_ROWS {len(existing) + len(fresh)}")

if __name__ == "__main__":
    main()
