"""Smoke tests for preprocessing and the trained model.

Run from the project root:  pytest
"""
import pytest

import config
from src.preprocess import clean_text, strip_dateline


def test_strip_dateline_removes_agency_tag():
    assert strip_dateline("WASHINGTON (Reuters) - The Senate voted.") == "The Senate voted."
    assert strip_dateline("PARIS (AFP) - Talks resumed.") == "Talks resumed."


def test_long_title_does_not_leak_reuters():
    title = "A very long headline about a Senate vote on the budget that runs well past eighty characters"
    body = "WASHINGTON (Reuters) - The Senate voted on Tuesday."
    cleaned = clean_text(title + " " + strip_dateline(body))
    assert "washington" not in cleaned and "reuters" not in cleaned
    # Without stripping the body first, the city survives (the regex window is 80 chars).
    assert "washington" in clean_text(title + " " + body)


def test_clean_text_basics():
    out = clean_text("Visit https://example.com <b>NOW</b>!!! This is NOT true.")
    assert "http" not in out and "<b>" not in out and "!" not in out
    assert "not" in out  # negations are kept
    assert clean_text("") == "" and clean_text(None) == ""


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="trained model not present")
def test_model_predicts():
    from src.predict import predict

    fake = predict("BREAKING: Scientists CONFIRM 5G towers drain vitamin D from your body!!! "
                   "The media is hiding it. Share this before it gets deleted!")
    assert fake["label"] == "FAKE"
    assert 0.0 <= fake["p_fake"] <= 1.0
    with pytest.raises(ValueError):
        predict("!!! ???")
