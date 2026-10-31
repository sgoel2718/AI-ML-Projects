"""
Export the trained models and test messages so the dashboard can run
entirely in the browser (the free, always-on Hugging Face static Space).

    python -m scripts.export_for_browser

Why this works: both models are Logistic Regression on TF-IDF numbers. Making
a prediction only needs, for every word, its IDF value and its weight, plus
one bias number:

    score       = sum(tfidf value of each word x its weight) + bias
    probability = 1 / (1 + e^-score)          (the sigmoid from Day 3)

So we save those numbers as JSON, and app/static/engine.js repeats the same
maths in JavaScript. The script checks that its own copy of that maths agrees
with scikit-learn before saving anything.

Creates deploy/browser/data/:
    spam_model.json.gz, sentiment_model.json.gz     words, IDF values, weights, bias
    stream_spam.json.gz, stream_sentiment.json.gz   unseen test messages for the stream
    metrics.json                                    test scores + bulk benchmark

The big files are gzip-compressed (17 MB -> 5 MB), because the hosting serves
them as they are; engine.js unzips them in the browser.
"""

import gzip
import json
import math
import re

import joblib
import numpy as np
import pandas as pd

from src.clean_text import clean_text
from src.config import (
    BENCHMARK_FILE, EXPLICIT_WORD_STARTS, EXPLICIT_WORDS, METRICS_FILE, PROJECT_DIR,
    RANDOM_SEED, SENTIMENT_MODEL_FILE, SENTIMENT_STREAM_FILE, SPAM_MODEL_FILE, SPAM_STREAM_FILE,
)

OUTPUT_DIR = PROJECT_DIR / "deploy" / "browser" / "data"

# The browser downloads these files, so we keep them small:
SENTIMENT_STREAM_SIZE = 20_000   # a random sample of the 320,000 test tweets
TEXT_LIMIT = 1_000               # characters kept per message (same as the server)
DIGITS = 5                       # significant digits kept for each number

TASKS = [
    ("spam", SPAM_MODEL_FILE, SPAM_STREAM_FILE, None),
    ("sentiment", SENTIMENT_MODEL_FILE, SENTIMENT_STREAM_FILE, SENTIMENT_STREAM_SIZE),
]


def short(number):
    """0.123456789 -> 0.12346, to keep the JSON small."""
    return float(f"{number:.{DIGITS}g}")


def export_model(model_file):
    pipeline = joblib.load(model_file)
    vectorizer = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["model"]
    words = vectorizer.get_feature_names_out()         # in column order
    return {
        "labels": [str(label) for label in model.classes_],
        "bias": short(model.intercept_[0]),
        "words": [str(word) for word in words],
        "idf": [short(value) for value in vectorizer.idf_],
        "weights": [short(value) for value in model.coef_[0]],
    }


TOKEN = re.compile(r"[a-z][a-z']+")   # same token_pattern as training


def browser_probability(model, index, text):
    """The exact steps engine.js performs, written in Python to test them."""
    tokens = TOKEN.findall(clean_text(text))
    terms = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    counts = {}
    for term in terms:
        if term in index:
            counts[term] = counts.get(term, 0) + 1
    values = {t: (1 + math.log(c)) * model["idf"][index[t]] for t, c in counts.items()}
    length = math.sqrt(sum(v * v for v in values.values())) or 1
    score = model["bias"] + sum(v / length * model["weights"][index[t]] for t, v in values.items())
    return 1 / (1 + math.exp(-score))


def check_against_sklearn(model_file, model, texts):
    pipeline = joblib.load(model_file)
    expected = pipeline.predict_proba(texts)[:, 1]
    index = {word: i for i, word in enumerate(model["words"])}
    ours = np.array([browser_probability(model, index, text) for text in texts])
    same_label = np.mean((ours > 0.5) == (expected > 0.5))
    biggest_gap = np.max(np.abs(ours - expected))
    print(f"  checked {len(texts):,} messages: same label {same_label:.2%}, largest probability gap {biggest_gap:.5f}")
    if same_label < 0.999:
        raise SystemExit("The browser maths does not match scikit-learn. Nothing was saved.")


def export_stream(stream_file, sample_size):
    messages = pd.read_csv(stream_file)
    if sample_size:
        messages = messages.sample(n=sample_size, random_state=RANDOM_SEED)
    # Short lists instead of objects: [label, text, source] saves a lot of space
    return [[row.label, str(row.text)[:TEXT_LIMIT], row.source] for row in messages.itertuples(index=False)]


def save(name, data):
    path = OUTPUT_DIR / name
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if name.endswith(".gz"):
        text = gzip.compress(text, compresslevel=9, mtime=0)   # mtime=0: same input, same file
    path.write_bytes(text)
    print(f"  saved {path.relative_to(PROJECT_DIR)} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for task, model_file, stream_file, sample_size in TASKS:
        print(f"{task}:")
        model = export_model(model_file)
        stream = export_stream(stream_file, sample_size)
        check_against_sklearn(model_file, model, [text for _, text, _ in stream[:3000]])
        save(f"{task}_model.json.gz", model)
        save(f"stream_{task}.json.gz", stream)

    metrics = json.loads(METRICS_FILE.read_text())
    if BENCHMARK_FILE.exists():
        metrics["benchmark"] = json.loads(BENCHMARK_FILE.read_text())
    metrics["safe_display"] = {"words": EXPLICIT_WORDS, "word_starts": EXPLICIT_WORD_STARTS}
    save("metrics.json", metrics)


if __name__ == "__main__":
    main()
