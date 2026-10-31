"""
Automated checks. Run from the project folder:
    python -m pytest

They train a tiny model on six messages, so they run in about a second and
don't need the real datasets.
"""

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.analyzer import TextAnalyzer
from src.clean_text import clean_text

TINY_TEXTS = [
    "win a free prize now", "free cash click here", "claim your free reward",
    "team meeting on friday", "lunch with the team", "project review friday",
]
TINY_LABELS = ["spam", "spam", "spam", "ham", "ham", "ham"]


# ---------------------------------------------------------------- clean_text
def test_clean_text_lowercases_and_removes_punctuation():
    assert clean_text("FREE Prize!!!") == "free prize"


def test_clean_text_replaces_links_and_usernames():
    assert clean_text("Visit http://win.com @john") == "visit url user"


def test_clean_text_keeps_apostrophes_and_not():
    assert clean_text("I don't like it, not good") == "i don't like it not good"


def test_clean_text_handles_numbers_only():
    assert clean_text(12345) == ""


# ---------------------------------------------------------------- analyzer
def save_tiny_model(folder, algorithm):
    vectorizer = TfidfVectorizer(preprocessor=clean_text, token_pattern=r"[a-z][a-z']+")
    pipeline = Pipeline([("tfidf", vectorizer), ("model", algorithm)])
    pipeline.fit(TINY_TEXTS, TINY_LABELS)
    path = folder / "tiny.joblib"
    joblib.dump(pipeline, path)
    return path


@pytest.mark.parametrize("algorithm", [MultinomialNB(), LogisticRegression()])
def test_analyzer_predicts_and_explains(tmp_path, algorithm):
    analyzer = TextAnalyzer(save_tiny_model(tmp_path, algorithm))

    spam = analyzer.analyze("free prize, click now")
    assert spam["label"] == "spam"
    assert 0.5 < spam["confidence"] <= 1
    assert "free" in [w["word"] for w in spam["top_words"]]

    ham = analyzer.analyze("team meeting friday")
    assert ham["label"] == "ham"
    # every listed word must have pushed toward the predicted label
    assert all(w["weight"] > 0 for w in ham["top_words"])


def test_analyzer_handles_unknown_words(tmp_path):
    analyzer = TextAnalyzer(save_tiny_model(tmp_path, MultinomialNB()))
    result = analyzer.analyze("hello mom")
    assert result["top_words"] == []          # no known words, nothing to explain
    assert result["label"] in ("spam", "ham")
