"""The agentic loop: lets the local LLM reason across multiple turns and
decide, on its own, when to call which tool - this is what makes it an
*agent* rather than a scripted decision tree with an LLM bolted on for
wording. The LLM never sees the two most important tools' opinions as
things it can override: check_red_flags is also run independently by the
CLI as a hard backstop (see red_flags.py docstring), and score_urgency is
pure deterministic math the model can call but not talk its way around.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from . import knowledge_base as kb
from . import red_flags
from .ollama_client import ChatResponse, OllamaClient

SYSTEM_PROMPT = """\
You are a clinical triage assistant. You are NOT a doctor and you do not \
diagnose. Your job is to have a short, focused conversation with a patient \
about their symptoms, gather the information the tools need, and recommend \
a *level of care* (self-care, routine appointment, urgent care, or \
emergency) - never a specific diagnosis.

Rules you must follow:
1. Call check_red_flags on the patient's own words as soon as you have a \
   description of their main symptom, before anything else.
2. If check_red_flags reports any match, immediately tell the patient this \
   may be a medical emergency and to call 911 / go to the ER. Do not \
   continue gathering routine information first.
3. Otherwise, ask brief clarifying questions ONE AT A TIME (not a list) to \
   learn: severity (1-10), how many days it has lasted, whether it's \
   getting worse, whether they have a fever, and roughly their age - only \
   what you don't already know.
4. Once you have enough to fill in the inputs, call score_urgency, then \
   call recommend_care_setting with the level it returns, then give the \
   patient a short final recommendation that includes that guidance text.
5. Always end a final recommendation with: "This is not a medical \
   diagnosis - if you're unsure, contact a healthcare provider."
6. Never invent lab results, vitals, or a diagnosis. If asked something \
   outside triage scope, say so plainly.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "check_red_flags",
            "description": "Scan the patient's own description for emergency red-flag symptom combinations (e.g. possible heart attack, stroke, anaphylaxis, suicidal ideation). Call this first, on the patient's raw words.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The patient's own description of their symptoms, verbatim."}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_symptom",
            "description": "Look up general background info (common causes, self-care, when to seek care) for one named symptom from the curated knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {"symptom": {"type": "string", "description": "A single symptom name, e.g. 'headache', 'sore throat', 'fever'."}},
                "required": ["symptom"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_urgency",
            "description": "Compute a deterministic urgency level from structured inputs gathered during the conversation. Call once you have enough information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "red_flag_count": {"type": "integer", "description": "Number of red flags check_red_flags returned. 0 if none or not yet checked."},
                    "severity_1_to_10": {"type": "integer", "description": "Patient's self-rated severity, 1 (mild) to 10 (worst possible)."},
                    "duration_days": {"type": "number", "description": "How many days the symptom has lasted."},
                    "is_worsening": {"type": "boolean"},
                    "has_fever": {"type": "boolean"},
                    "age_years": {"type": "integer", "description": "Patient's age in years, if known."},
                },
                "required": ["severity_1_to_10", "duration_days", "is_worsening", "has_fever"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_care_setting",
            "description": "Given an urgency level from score_urgency, return the guidance text to relay to the patient.",
            "parameters": {
                "type": "object",
                "properties": {"urgency_level": {"type": "string", "enum": ["self_care", "routine_appointment", "urgent_care", "emergency"]}},
                "required": ["urgency_level"],
            },
        },
    },
]


def _tool_check_red_flags(text: str) -> dict[str, Any]:
    result = red_flags.check_red_flags(text)
    return {"matched": result.matched, "is_emergency": result.is_emergency}


def _tool_lookup_symptom(symptom: str) -> dict[str, Any]:
    entry = kb.lookup_symptom(symptom)
    return entry if entry is not None else {"error": f"no knowledge base entry for '{symptom}'"}


def _tool_score_urgency(
    severity_1_to_10: int,
    duration_days: float,
    is_worsening: bool,
    has_fever: bool,
    red_flag_count: int = 0,
    age_years: int | None = None,
) -> dict[str, Any]:
    inputs = kb.UrgencyInputs(
        red_flag_count=red_flag_count,
        severity_1_to_10=severity_1_to_10,
        duration_days=duration_days,
        is_worsening=is_worsening,
        has_fever=has_fever,
        age_years=age_years,
    )
    level, score, reasons = kb.score_urgency(inputs)
    return {"urgency_level": level, "score": score, "reasons": reasons}


def _tool_recommend_care_setting(urgency_level: str) -> dict[str, Any]:
    if urgency_level == kb.EMERGENCY:
        return {"guidance": red_flags.EMERGENCY_NOTICE}
    guidance = kb.CARE_SETTING_GUIDANCE.get(urgency_level)
    if guidance is None:
        return {"error": f"unknown urgency_level '{urgency_level}'"}
    return {"guidance": guidance}


_TOOL_PARAM_TYPES: dict[str, dict[str, str]] = {
    tool["function"]["name"]: {
        pname: pschema.get("type", "string")
        for pname, pschema in tool["function"]["parameters"]["properties"].items()
    }
    for tool in TOOLS
}


def _coerce_value(value: Any, json_type: str) -> Any:
    """Local models (llama3.2 via Ollama included) frequently stringify tool
    arguments regardless of the declared schema type ("false" instead of
    false, "24" instead of 24). Coerce against the declared type so tool
    implementations can assume real Python types, without silently masking
    genuinely malformed input (falls back to the original value on failure,
    so the TypeError still surfaces from the implementation call).
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if json_type == "boolean":
        return stripped.lower() in {"true", "1", "yes"}
    if json_type == "integer":
        try:
            return int(float(stripped))
        except ValueError:
            return value
    if json_type == "number":
        try:
            return float(stripped)
        except ValueError:
            return value
    return value


def _coerce_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    declared_types = _TOOL_PARAM_TYPES.get(name, {})
    return {key: _coerce_value(value, declared_types.get(key, "string")) for key, value in args.items()}


TOOL_IMPLEMENTATIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "check_red_flags": _tool_check_red_flags,
    "lookup_symptom": _tool_lookup_symptom,
    "score_urgency": _tool_score_urgency,
    "recommend_care_setting": _tool_recommend_care_setting,
}


@dataclass
class TurnResult:
    reply: str
    tool_trace: list[dict[str, Any]] = field(default_factory=list)


class TriageAgent:
    def __init__(self, client: OllamaClient | None = None, max_tool_iterations: int = 6):
        self.client = client or OllamaClient()
        self.max_tool_iterations = max_tool_iterations
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _dispatch_tool_call(self, call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        function = call.get("function", {})
        name = function.get("name", "")
        raw_args = function.get("arguments", {})
        args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")

        impl = TOOL_IMPLEMENTATIONS.get(name)
        if impl is None:
            return name, {"error": f"unknown tool '{name}'"}
        try:
            return name, impl(**_coerce_args(name, args))
        except TypeError as exc:
            return name, {"error": f"bad arguments for {name}: {exc}"}

    def send(self, user_text: str) -> TurnResult:
        self.messages.append({"role": "user", "content": user_text})
        trace: list[dict[str, Any]] = []

        for _ in range(self.max_tool_iterations):
            response: ChatResponse = self.client.chat(self.messages, tools=TOOLS)

            if not response.tool_calls:
                self.messages.append({"role": "assistant", "content": response.content})
                return TurnResult(reply=response.content, tool_trace=trace)

            self.messages.append(
                {"role": "assistant", "content": response.content, "tool_calls": response.tool_calls}
            )
            for call in response.tool_calls:
                name, result = self._dispatch_tool_call(call)
                trace.append({"tool": name, "args": call.get("function", {}).get("arguments"), "result": result})
                self.messages.append({"role": "tool", "name": name, "content": json.dumps(result)})

        return TurnResult(
            reply="(agent reached its tool-call limit for this turn without a final answer)",
            tool_trace=trace,
        )
