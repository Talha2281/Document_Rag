import re
import logging

logger = logging.getLogger("documind.guardrails")

# Simple, foolproof patterns for testing
JAILBREAK_PATTERNS = [
    r"ignore.*previous.*instructions",
    r"system prompt",
    r"you are now",
    r"dan mode",
    r"reveal.*prompt",
]

BLOCKED_MESSAGE = "Security Alert: Prompt Injection / Jailbreak attempt detected. Request blocked by DocuMind Guardrails."

async def validate_query_safety(question: str) -> dict:
    print(f"\n[GUARDRAILS CHECK] Received query: {question}")
    
    normalized = question.lower().strip()

    # 1. Immediate Pattern Match
    for pattern in JAILBREAK_PATTERNS:
        if re.search(pattern, normalized):
            print(f"[GUARDRAILS BLOCKED] Matched pattern: {pattern}")
            return {
                "allowed": False,
                "blocked_reason": BLOCKED_MESSAGE
            }

    print("[GUARDRAILS ALLOWED] Passed safety check.")
    return {"allowed": True}