from triageagent.red_flags import check_red_flags


def test_no_match_on_ordinary_text():
    result = check_red_flags("I have a mild headache and a runny nose.")
    assert result.matched == []
    assert not result.is_emergency


def test_matches_chest_pain():
    result = check_red_flags("I've had crushing chest pain for the last hour.")
    assert "possible heart attack" in result.matched
    assert result.is_emergency


def test_matches_stroke_combo_requires_both_phrases():
    assert check_red_flags("noticing face drooping on one side").matched == ["possible stroke"]
    # "confusion" alone should not fire the "sudden confusion" stroke flag
    assert check_red_flags("I feel a bit confused today").matched == []


def test_matches_suicidal_ideation_and_flags_crisis():
    result = check_red_flags("I want to kill myself")
    assert "suicidal ideation" in result.matched
    assert result.is_crisis
    assert "988" in result.notice()


def test_case_insensitive():
    result = check_red_flags("I CANNOT BREATHE")
    assert "severe breathing difficulty" in result.matched


def test_no_duplicate_labels():
    # both "chest pain" and "chest pressure" phrasing present
    result = check_red_flags("chest pain and chest pressure that won't go away")
    assert result.matched.count("possible heart attack") == 1
