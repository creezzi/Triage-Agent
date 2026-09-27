"""Curated symptom knowledge base and deterministic urgency scoring.

Same design choice as ResumeMatch's skill vocabulary (see
../08-resumematch-python): a small curated table instead of free-text
generation from the LLM. Urgency scoring in particular must be
reproducible and auditable - "why did the agent say urgent care" needs a
traceable answer, not "the model felt like it that time." The LLM's job is
conversation and judgment about *which* symptoms and inputs apply; this
module's job is to turn structured inputs into a consistent number.
"""

from __future__ import annotations

from dataclasses import dataclass

SYMPTOM_KB: dict[str, dict[str, object]] = {
    "headache": {
        "common_causes": ["tension", "dehydration", "eye strain", "migraine", "poor sleep"],
        "self_care": ["rest in a dark quiet room", "hydrate", "over-the-counter pain reliever"],
        "seek_care_if": ["worst headache of your life", "sudden onset", "with fever and stiff neck", "after a head injury"],
    },
    "sore throat": {
        "common_causes": ["viral cold", "strep throat", "allergies", "dry air"],
        "self_care": ["warm salt water gargle", "fluids", "throat lozenges"],
        "seek_care_if": ["difficulty swallowing", "difficulty breathing", "lasts more than a week", "high fever"],
    },
    "cough": {
        "common_causes": ["viral upper respiratory infection", "allergies", "post-nasal drip", "asthma"],
        "self_care": ["fluids", "honey (if over age 1)", "humidifier"],
        "seek_care_if": ["coughing up blood", "lasts more than 3 weeks", "with high fever or shortness of breath"],
    },
    "abdominal pain": {
        "common_causes": ["indigestion", "gas", "constipation", "gastroenteritis"],
        "self_care": ["bland diet", "fluids", "rest"],
        "seek_care_if": ["severe or localized to lower right side", "with fever and vomiting", "rigid/hard abdomen"],
    },
    "fever": {
        "common_causes": ["viral infection", "bacterial infection", "inflammation"],
        "self_care": ["fluids", "rest", "fever reducer per label instructions"],
        "seek_care_if": ["over 103F/39.4C", "lasts more than 3 days", "with rash, stiff neck, or confusion"],
    },
    "rash": {
        "common_causes": ["contact dermatitis", "allergic reaction", "viral exanthem", "heat rash"],
        "self_care": ["avoid known irritants", "cool compress", "over-the-counter antihistamine"],
        "seek_care_if": ["spreading rapidly", "with facial/throat swelling", "with fever", "looks like bruising/petechiae"],
    },
    "back pain": {
        "common_causes": ["muscle strain", "poor posture", "overuse"],
        "self_care": ["rest briefly then gentle movement", "over-the-counter pain reliever", "heat/ice"],
        "seek_care_if": ["numbness/weakness in legs", "loss of bladder/bowel control", "after significant trauma"],
    },
    "fatigue": {
        "common_causes": ["poor sleep", "stress", "viral illness", "anemia"],
        "self_care": ["prioritize sleep", "hydrate", "balanced meals"],
        "seek_care_if": ["sudden and severe", "with chest pain or shortness of breath", "unexplained weight loss"],
    },
}


def lookup_symptom(symptom: str) -> dict[str, object] | None:
    """Case-insensitive exact/substring lookup against the curated KB."""
    key = symptom.strip().lower()
    if key in SYMPTOM_KB:
        return SYMPTOM_KB[key]
    for kb_key, entry in SYMPTOM_KB.items():
        if kb_key in key or key in kb_key:
            return entry
    return None


UrgencyLevel = str  # one of the four strings below, kept as str for simple JSON tool results

SELF_CARE = "self_care"
ROUTINE_APPOINTMENT = "routine_appointment"
URGENT_CARE = "urgent_care"
EMERGENCY = "emergency"


@dataclass
class UrgencyInputs:
    red_flag_count: int = 0
    severity_1_to_10: int = 3
    duration_days: float = 1.0
    is_worsening: bool = False
    has_fever: bool = False
    age_years: int | None = None


def score_urgency(inputs: UrgencyInputs) -> tuple[UrgencyLevel, int, list[str]]:
    """Weighted, deterministic urgency score.

    Returns (level, score, reasons) so the "why" is always inspectable -
    every point added is tied to a specific input, not a black box.
    Any red flag short-circuits straight to EMERGENCY regardless of the
    rest of the score, since a single red flag (e.g. chest pain) matters
    more than the sum of everything else.
    """
    if inputs.red_flag_count > 0:
        return EMERGENCY, 100, ["one or more emergency red flags were present"]

    score = 0
    reasons: list[str] = []

    if inputs.severity_1_to_10 >= 8:
        score += 4
        reasons.append(f"severity rated {inputs.severity_1_to_10}/10 (high)")
    elif inputs.severity_1_to_10 >= 5:
        score += 2
        reasons.append(f"severity rated {inputs.severity_1_to_10}/10 (moderate)")

    if inputs.is_worsening:
        score += 2
        reasons.append("symptom is actively worsening")

    if inputs.has_fever:
        score += 1
        reasons.append("fever present")

    if inputs.duration_days >= 14:
        score += 2
        reasons.append(f"has lasted {inputs.duration_days:g} days (2+ weeks)")
    elif inputs.duration_days >= 7:
        score += 1
        reasons.append(f"has lasted {inputs.duration_days:g} days (1+ week)")

    if inputs.age_years is not None and (inputs.age_years < 2 or inputs.age_years >= 75):
        score += 2
        reasons.append(f"age {inputs.age_years} is in a higher-risk band")

    if not reasons:
        reasons.append("no significant risk factors reported")

    if score >= 7:
        level = URGENT_CARE
    elif score >= 3:
        level = ROUTINE_APPOINTMENT
    else:
        level = SELF_CARE

    return level, score, reasons


CARE_SETTING_GUIDANCE: dict[UrgencyLevel, str] = {
    SELF_CARE: (
        "Self-care is reasonable for now: rest, hydration, and over-the-counter "
        "remedies as appropriate. Reassess if symptoms change or worsen."
    ),
    ROUTINE_APPOINTMENT: (
        "Schedule a routine appointment with a primary care provider in the "
        "next few days, or use a telehealth visit if one isn't available soon."
    ),
    URGENT_CARE: (
        "Seek care today - an urgent care clinic or same-day appointment is "
        "appropriate. Go sooner if anything worsens."
    ),
    EMERGENCY: EMERGENCY,  # overridden by red_flags.EMERGENCY_NOTICE at the call site
}
