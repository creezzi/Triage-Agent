"""Runs two scripted conversations through the real agent against a live
local Ollama model, and prints a transcript. Requires `ollama serve` running
and the model pulled (`ollama pull llama3.2`). This is what generated
samples/demo_transcript.md.

    python demo.py
"""

from __future__ import annotations

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # avoid mojibake on Windows consoles

from triageagent.agent import TriageAgent
from triageagent.ollama_client import OllamaClient
from triageagent import red_flags


def run_scenario(title: str, turns: list[str]) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    agent = TriageAgent(client=OllamaClient(model="llama3.2"))
    for turn in turns:
        print(f"\nyou> {turn}")
        backstop = red_flags.check_red_flags(turn)
        if backstop.is_emergency:
            print(f"[SAFETY CHECK] matched: {', '.join(backstop.matched)}")
            print(backstop.notice())
        result = agent.send(turn)
        for step in result.tool_trace:
            print(f"  [tool call] {step['tool']}({step['args']}) -> {step['result']}")
        print(f"agent> {result.reply}")


if __name__ == "__main__":
    run_scenario(
        "Scenario 1: mild, non-urgent symptom",
        [
            "I've had a mild headache since yesterday, maybe a 3 out of 10.",
            "No fever, and it's not getting worse. I'm 24.",
        ],
    )
    run_scenario(
        "Scenario 2: emergency red flag",
        ["I have crushing chest pain and I can't catch my breath."],
    )
