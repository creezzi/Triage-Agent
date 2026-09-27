# Sample transcript

Captured with `python demo.py` against a real local Ollama server running
`llama3.2:latest` (no cloud API, no cost). Reproducible by anyone who has
Ollama installed and running `ollama pull llama3.2` first.

## Scenario 1: mild, non-urgent symptom

```
you> I've had a mild headache since yesterday, maybe a 3 out of 10.
  [tool call] check_red_flags({'text': 'mild headache since yesterday, maybe a 3 out of 10'})
    -> {'matched': [], 'is_emergency': False}
agent> I'm going to ask a few questions to help me better understand your symptoms.
       Can you tell me how long the headache has been getting worse?

you> No fever, and it's not getting worse. I'm 24.
  [tool call] score_urgency({'has_fever': 'false', 'age_years': '24', 'red_flag_count': '0',
                             'severity_1_to_10': '3', 'duration_days': '1', 'is_worsening': 'false'})
    -> {'urgency_level': 'self_care', 'score': 0, 'reasons': ['no significant risk factors reported']}
agent> Since your symptoms are mild and not worsening, I recommend taking some over-the-counter
       pain relief medication and staying hydrated. You can also try applying a cold or warm
       compress to your forehead or neck to help alleviate the headache. If you experience any
       worsening of symptoms or if the headache persists, consider scheduling an appointment
       with your primary care physician. This is not a medical diagnosis - if you're unsure,
       contact a healthcare provider.
```

Note the tool call arguments arrive as strings (`'false'`, `'24'`) even though the schema
declares `boolean`/`integer` - see "The one real bug I hit" in the README. `agent.py`'s
`_coerce_args` converts these before the tool implementation ever sees them.

## Scenario 2: emergency red flag

```
you> I have crushing chest pain and I can't catch my breath.
[SAFETY CHECK] matched: possible heart attack, severe breathing difficulty
This describes a potential medical emergency. Call 911 (or your local emergency number) or go
to the nearest emergency room now. Do not wait for a chatbot's opinion.
  [tool call] check_red_flags({'text': "crushing chest pain and can’t catch my breath"})
    -> {'matched': ['possible heart attack'], 'is_emergency': True}
agent> This may be a medical emergency. Please call 911 or go to the emergency room immediately.
       Do not continue with our conversation.
```

This run is exactly why the CLI runs `check_red_flags` on the patient's raw input itself,
independent of the LLM: the model re-typed the apostrophe in "can't" as a curly `’`
before calling the tool, which alone would have missed the "severe breathing difficulty"
match (it only fired on "possible heart attack"). The CLI's own backstop check, run on the
unmodified user text before the LLM ever sees it, caught both.
