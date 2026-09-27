"""Interactive CLI. Run with: python -m triageagent"""

from __future__ import annotations

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # avoid mojibake on Windows consoles

from . import red_flags
from .agent import TriageAgent
from .ollama_client import OllamaClient, OllamaUnavailable

DISCLAIMER = """\
============================================================
 triageagent - educational demo, NOT a medical device.
 This does not diagnose, prescribe, or replace a clinician.
 In any emergency, call 911 (or your local emergency number).
============================================================
"""


def run(model: str = "llama3.2") -> None:
    print(DISCLAIMER)
    client = OllamaClient(model=model)
    agent = TriageAgent(client=client)

    print("Describe what's going on. Type 'exit' to quit.\n")
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        # Hard safety backstop: checked directly, independent of whether the
        # LLM decides to call the check_red_flags tool this turn.
        backstop = red_flags.check_red_flags(user_text)
        if backstop.is_emergency:
            print(f"\n[SAFETY CHECK] matched: {', '.join(backstop.matched)}")
            print(backstop.notice())
            print()

        try:
            result = agent.send(user_text)
        except OllamaUnavailable as exc:
            print(f"\n[error] {exc}")
            sys.exit(1)

        print(f"\nagent> {result.reply}\n")


if __name__ == "__main__":
    run()
