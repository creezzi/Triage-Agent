# triageagent

A local, LLM-backed clinical triage agent: given a plain-English description of symptoms, it
holds a short conversation, decides for itself which tools to call and when, and recommends a
level of care (self-care / routine appointment / urgent care / emergency) - with a deterministic
safety net underneath it that the LLM cannot override.

**Educational software only. Not a medical device, not FDA-cleared, not a diagnostic tool, and
not a substitute for professional medical judgment.** See "Safety design" below.

## The problem

Most "AI agent" demos either (a) call a paid hosted API, which costs money and requires sharing
data with a third party, or (b) are a chatbot with a system prompt and no real tool use, which
isn't actually agentic - it's just a scripted conversation with nicer wording. And in a healthcare
context specifically, letting an LLM be the *only* thing standing between a user and a missed
emergency is a genuine safety risk, not a hypothetical one.

## What it does

Runs entirely against a local [Ollama](https://ollama.com) model (`llama3.2` by default) - no API
key, no per-request cost, nothing leaves the machine. The agent decides on its own, turn by turn,
whether to call `check_red_flags`, `lookup_symptom`, `score_urgency`, or `recommend_care_setting`,
and the conversation continues until it has enough information to give a recommendation. That
tool-selection loop is what makes it an agent rather than a decision tree with LLM-generated text.

## Design

```
triageagent/
  red_flags.py       deterministic emergency detection (the safety backstop)
  knowledge_base.py   curated symptom info + deterministic urgency scoring
  ollama_client.py    stdlib-only client for Ollama's local /api/chat
  agent.py            the tool-calling loop, tool schemas, system prompt
  cli.py               interactive entrypoint (python -m triageagent)
```

**Why the LLM doesn't get the final say on emergencies:** `red_flags.py` is plain string matching
with zero LLM involvement, and the CLI runs it *directly on the user's raw input* before the LLM
sees the conversation at all - in addition to the LLM also having it as a callable tool. A model
can forget to call a tool it was told to call, get distracted mid-conversation, or paraphrase text
in a way that breaks a keyword match (see the demo transcript below for a real example of exactly
that happening). The independent backstop can't do any of those things. If the two ever disagree,
the deterministic check wins and fires regardless of what the LLM says next.

**Why urgency scoring is a fixed formula, not an LLM judgment call:** same reasoning as
[`08-resumematch-python`](../08-resumematch-python)'s curated skill vocabulary - "why did the
agent recommend urgent care" needs a reproducible, auditable answer (`score_urgency` returns its
`reasons` list alongside the score), not "the model felt like it that time." The LLM's job is
conversation and deciding *when* it has enough information to call the tool; turning that
information into a level of care is deterministic math.

**The one real bug I hit building this:** Ollama's `llama3.2` tool-calling sends every argument
as a string regardless of the JSON schema type - `"false"` instead of `false`, `"24"` instead of
`24`. The first live run crashed `score_urgency` with `'>' not supported between instances of
'str' and 'int'`. `agent.py`'s `_coerce_args` reads each tool's declared parameter types from its
own schema and coerces string arguments against them before the implementation ever runs -
documented and regression-tested in `tests/test_agent_loop.py`, and visible directly in
`samples/demo_transcript.md`.

## Running it

Requires [Ollama](https://ollama.com) installed and running locally, with a tool-calling-capable
model pulled:

```bash
ollama pull llama3.2
```

```bash
pip install -r requirements-dev.txt
python -m pytest                              # 19 tests, no Ollama required

python demo.py                                 # two scripted scenarios against the real model
python -m triageagent                          # interactive chat
```

See `samples/demo_transcript.md` for real output from `python demo.py`, including the emergency
scenario where the safety backstop matters.

## Tests

19 pytest tests, all fast and offline (the agent-loop tests use a scripted fake Ollama client, not
a live model, so they're deterministic and don't need Ollama running):

- `test_red_flags.py` - multi-phrase matching (won't false-positive on "confusion" alone),
  case-insensitivity, no duplicate labels, crisis-line notice for suicidal ideation.
- `test_knowledge_base.py` - a red flag always forces `EMERGENCY` regardless of other inputs,
  scoring thresholds, age-band risk bump, reasons are always populated.
- `test_agent_loop.py` - single and multi-step tool-call sequences, unknown-tool-name handling,
  the string-argument coercion regression, and the max-iterations fallback.

## What I'd change for production

- Replace the curated symptom knowledge base and red-flag phrase list with a clinically reviewed
  one (built by/with a clinician, not a CS student), and add real ICD-10/SNOMED coding.
- Log every conversation and tool call to an auditable, access-controlled store for after-the-fact
  clinical review - this repo does not persist anything.
- Run behind authentication with per-patient data isolation, and treat all input as PHI: encrypt
  at rest and in transit, add a BAA-covered hosting story, and go through an actual HIPAA
  compliance review before any real patient data touches it.
- Swap the local model for one validated on medical-triage benchmarks, and add a human-in-the-loop
  review step before any recommendation reaches a patient, not just an LLM's own confidence.
- Structured intake instead of free text for anything safety-critical (e.g. severity as a
  required numeric field, not something the LLM has to remember to ask for).
