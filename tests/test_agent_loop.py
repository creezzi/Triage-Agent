"""Tests the tool-calling loop with a scripted fake Ollama client, so the
loop's control flow is verified without needing a live model. See
samples/demo_transcript.md for a transcript captured against a real local
Ollama run.
"""

import json

from triageagent.agent import TriageAgent
from triageagent.ollama_client import ChatResponse


class FakeClient:
    """Returns pre-scripted responses in order, one per .chat() call."""

    def __init__(self, scripted_responses: list[ChatResponse]):
        self._responses = list(scripted_responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        return self._responses.pop(0)


def _tool_call(name: str, arguments: dict) -> dict:
    return {"function": {"name": name, "arguments": arguments}}


def test_single_tool_call_then_final_answer():
    fake = FakeClient(
        [
            ChatResponse(
                role="assistant",
                content="",
                tool_calls=[_tool_call("check_red_flags", {"text": "I have a mild headache"})],
            ),
            ChatResponse(role="assistant", content="That sounds mild. Can you rate the severity 1-10?"),
        ]
    )
    agent = TriageAgent(client=fake)
    result = agent.send("I have a mild headache")

    assert result.reply == "That sounds mild. Can you rate the severity 1-10?"
    assert len(result.tool_trace) == 1
    assert result.tool_trace[0]["tool"] == "check_red_flags"
    assert result.tool_trace[0]["result"]["is_emergency"] is False

    # the tool result was fed back into the conversation as a 'tool' message
    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert json.loads(tool_messages[0]["content"])["is_emergency"] is False


def test_multiple_sequential_tool_calls():
    fake = FakeClient(
        [
            ChatResponse(role="assistant", content="", tool_calls=[_tool_call("check_red_flags", {"text": "chest pain"})]),
            ChatResponse(
                role="assistant",
                content="",
                tool_calls=[
                    _tool_call(
                        "score_urgency",
                        {"severity_1_to_10": 9, "duration_days": 0.1, "is_worsening": True, "has_fever": False, "red_flag_count": 1},
                    )
                ],
            ),
            ChatResponse(role="assistant", content="This may be an emergency. Call 911."),
        ]
    )
    agent = TriageAgent(client=fake)
    result = agent.send("chest pain, getting worse")

    assert "911" in result.reply
    assert [t["tool"] for t in result.tool_trace] == ["check_red_flags", "score_urgency"]
    assert result.tool_trace[1]["result"]["urgency_level"] == "emergency"


def test_unknown_tool_name_does_not_crash_the_loop():
    fake = FakeClient(
        [
            ChatResponse(role="assistant", content="", tool_calls=[_tool_call("teleport_patient", {})]),
            ChatResponse(role="assistant", content="ok, ignoring that."),
        ]
    )
    agent = TriageAgent(client=fake)
    result = agent.send("hello")
    assert result.tool_trace[0]["result"] == {"error": "unknown tool 'teleport_patient'"}
    assert result.reply == "ok, ignoring that."


def test_stringified_tool_arguments_are_coerced_to_declared_types():
    # Regression test: llama3.2 via Ollama sends tool arguments as strings
    # ("false", "24") even though the schema declares boolean/integer.
    fake = FakeClient(
        [
            ChatResponse(
                role="assistant",
                content="",
                tool_calls=[
                    _tool_call(
                        "score_urgency",
                        {
                            "severity_1_to_10": "3",
                            "duration_days": "1",
                            "is_worsening": "false",
                            "has_fever": "false",
                            "age_years": "24",
                            "red_flag_count": "0",
                        },
                    )
                ],
            ),
            ChatResponse(role="assistant", content="Self-care sounds appropriate."),
        ]
    )
    agent = TriageAgent(client=fake)
    result = agent.send("mild headache")
    assert "error" not in result.tool_trace[0]["result"]
    assert result.tool_trace[0]["result"]["urgency_level"] == "self_care"


def test_hits_max_tool_iterations_without_crashing():
    loop_response = ChatResponse(role="assistant", content="", tool_calls=[_tool_call("check_red_flags", {"text": "x"})])
    fake = FakeClient([loop_response] * 10)
    agent = TriageAgent(client=fake, max_tool_iterations=3)
    result = agent.send("x")
    assert "tool-call limit" in result.reply
    assert len(result.tool_trace) == 3
