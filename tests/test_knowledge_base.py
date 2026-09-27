from triageagent.knowledge_base import (
    EMERGENCY,
    ROUTINE_APPOINTMENT,
    SELF_CARE,
    URGENT_CARE,
    UrgencyInputs,
    lookup_symptom,
    score_urgency,
)


def test_lookup_exact_match():
    entry = lookup_symptom("headache")
    assert entry is not None
    assert "tension" in entry["common_causes"]


def test_lookup_case_insensitive_and_unknown():
    assert lookup_symptom("Sore Throat") is not None
    assert lookup_symptom("bubonic plague") is None


def test_red_flag_always_wins_regardless_of_other_inputs():
    inputs = UrgencyInputs(red_flag_count=1, severity_1_to_10=1, duration_days=0.1)
    level, score, reasons = score_urgency(inputs)
    assert level == EMERGENCY
    assert score == 100


def test_mild_short_symptom_is_self_care():
    inputs = UrgencyInputs(severity_1_to_10=2, duration_days=1, is_worsening=False, has_fever=False)
    level, _, _ = score_urgency(inputs)
    assert level == SELF_CARE


def test_moderate_persistent_symptom_is_routine():
    inputs = UrgencyInputs(severity_1_to_10=5, duration_days=8, is_worsening=False, has_fever=False)
    level, score, reasons = score_urgency(inputs)
    assert level == ROUTINE_APPOINTMENT
    assert any("8" in r for r in reasons)


def test_severe_worsening_with_fever_is_urgent():
    inputs = UrgencyInputs(severity_1_to_10=9, duration_days=3, is_worsening=True, has_fever=True)
    level, score, _ = score_urgency(inputs)
    assert level == URGENT_CARE
    assert score >= 7


def test_extreme_age_bumps_score():
    baseline = score_urgency(UrgencyInputs(severity_1_to_10=4, duration_days=1))[1]
    with_age = score_urgency(UrgencyInputs(severity_1_to_10=4, duration_days=1, age_years=1))[1]
    assert with_age > baseline


def test_reasons_are_never_empty():
    _, _, reasons = score_urgency(UrgencyInputs())
    assert len(reasons) >= 1
