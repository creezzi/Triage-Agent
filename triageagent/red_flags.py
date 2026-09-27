"""Deterministic emergency red-flag detection.

This module is the agent's hard safety backstop. It does not depend on the
LLM in any way, and the CLI runs it directly on raw patient input *before*
the LLM ever sees the conversation, and again as a tool the LLM can call
mid-conversation. A model can be wrong, distracted by a long conversation,
or simply not call the tool it was told to call - this module can't forget,
get talked out of it, or hallucinate a miss. If the LLM and this module ever
disagree, this module wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Each entry: (label, all phrases that must co-occur for this flag to fire).
# Multi-phrase entries reduce false positives on single ambiguous words
# ("pain" alone is common and not urgent; "crushing chest pain" is).
_RED_FLAGS: list[tuple[str, list[str]]] = [
    ("possible heart attack", ["chest pain"]),
    ("possible heart attack", ["chest pressure"]),
    ("possible heart attack", ["crushing", "chest"]),
    ("possible stroke", ["face drooping"]),
    ("possible stroke", ["facial droop"]),
    ("possible stroke", ["slurred speech"]),
    ("possible stroke", ["one side", "weak"]),
    ("possible stroke", ["sudden", "confusion"]),
    ("possible stroke", ["worst headache"]),
    ("severe breathing difficulty", ["can't breathe"]),
    ("severe breathing difficulty", ["cannot breathe"]),
    ("severe breathing difficulty", ["can't catch my breath"]),
    ("severe breathing difficulty", ["blue lips"]),
    ("severe breathing difficulty", ["gasping"]),
    ("anaphylaxis", ["throat closing"]),
    ("anaphylaxis", ["throat swelling"]),
    ("anaphylaxis", ["allergic reaction", "swelling"]),
    ("uncontrolled bleeding", ["bleeding", "won't stop"]),
    ("uncontrolled bleeding", ["bleeding", "wont stop"]),
    ("uncontrolled bleeding", ["spurting blood"]),
    ("loss of consciousness", ["passed out"]),
    ("loss of consciousness", ["unresponsive"]),
    ("loss of consciousness", ["unconscious"]),
    ("suicidal ideation", ["suicidal"]),
    ("suicidal ideation", ["want to die"]),
    ("suicidal ideation", ["kill myself"]),
    ("suicidal ideation", ["end my life"]),
    ("seizure", ["seizure"]),
    ("seizure", ["convulsing"]),
    ("severe allergic swelling", ["swollen tongue"]),
    ("poisoning / overdose", ["overdose"]),
    ("poisoning / overdose", ["swallowed", "poison"]),
]

CRISIS_LINE_NOTICE = (
    "If you are thinking about suicide or self-harm, please call or text 988 "
    "(Suicide & Crisis Lifeline, US) right now, or call 911. You don't have "
    "to be in crisis to call - you can call to talk to someone."
)

EMERGENCY_NOTICE = (
    "This describes a potential medical emergency. Call 911 (or your local "
    "emergency number) or go to the nearest emergency room now. Do not wait "
    "for a chatbot's opinion."
)


@dataclass
class RedFlagResult:
    matched: list[str] = field(default_factory=list)

    @property
    def is_emergency(self) -> bool:
        return len(self.matched) > 0

    @property
    def is_crisis(self) -> bool:
        return "suicidal ideation" in self.matched

    def notice(self) -> str:
        if self.is_crisis:
            return CRISIS_LINE_NOTICE + "\n" + EMERGENCY_NOTICE
        return EMERGENCY_NOTICE


def check_red_flags(text: str) -> RedFlagResult:
    """Scan free text for emergency red-flag phrase combinations.

    Case-insensitive substring matching on purpose (see README "Design"):
    recall matters far more than precision here, since a false positive
    just means the agent double-checks with the user, but a false
    negative means a missed emergency.
    """
    lowered = text.lower()
    matched: list[str] = []
    for label, phrases in _RED_FLAGS:
        if all(phrase in lowered for phrase in phrases) and label not in matched:
            matched.append(label)
    return RedFlagResult(matched=matched)
