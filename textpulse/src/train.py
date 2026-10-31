"""
Train both models: the spam detector and the sentiment analyser.

Run from the project folder:
    python -m src.train

For each task this script:
  1. loads the data
  2. splits it into training data (80%) and test data (20%)
  3. turns text into numbers with TF-IDF
  4. trains two algorithms, Naive Bayes and Logistic Regression
  5. measures both on the test data and keeps the better one
  6. saves the model, its scores, and the test messages for the live stream

If a dataset hasn't been downloaded, that task is skipped with a message.
"""

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.clean_text import clean_text
from src.config import (
    MAX_WORDS, METRICS_FILE, MIN_WORD_COUNT, MODELS_DIR, RANDOM_SEED,
    SENTIMENT_MODEL_FILE, SENTIMENT_STREAM_FILE, SPAM_MODEL_FILE,
    SPAM_STREAM_FILE, STREAM_TEXT_LIMIT, TEST_SIZE, VOCAB_SAMPLE_SIZE,
)
from src.load_data import load_sentiment_data, load_spam_data


def make_vectorizer():
    """
    TF-IDF turns each message into a long list of numbers, one per word.

    - TF  (term frequency): words used more in a message get a higher number.
    - IDF (inverse document frequency): words that appear in almost every
      message ("the", "and") get a lower number, because they tell us little.
    """
    return TfidfVectorizer(
        preprocessor=clean_text,          # clean every message first
        token_pattern=r"[a-z][a-z']+",    # a word = letters (and ') of length 2+
        ngram_range=(1, 2),               # single words AND word pairs ("not good")
        min_df=MIN_WORD_COUNT,            # skip very rare words (often typos)
        max_features=MAX_WORDS,           # keep the most common words only
        sublinear_tf=True,                # 10 x "free" counts less than 10 times 1
        dtype=np.float32,                 # half the memory of the default float64
    )


def score_model(model, X_test, y_test, positive_label):
    """Measure a trained model on test data it has never seen."""
    predictions = model.predict(X_test)
    labels = [label for label in model.classes_ if label != positive_label] + [positive_label]
    return {
        # Of all messages, how many did we get right?
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        # Of the messages we flagged as spam, how many really were spam?
        "precision": round(precision_score(y_test, predictions, pos_label=positive_label), 4),
        # Of the real spam messages, how many did we catch?
        "recall": round(recall_score(y_test, predictions, pos_label=positive_label), 4),
        # One number that balances precision and recall
        "f1": round(f1_score(y_test, predictions, pos_label=positive_label), 4),
        # Rows = true label, columns = predicted label, in the order of "labels"
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "confusion_labels": labels,
    }


def train_task(task_name, data, positive_label, model_file, stream_file):
    """Run the full training process for one task and return its results."""
    print(f"\n=== {task_name.upper()} ===")
    print(f"  {len(data):,} messages | label counts: {data['label'].value_counts().to_dict()}")

    # 1. Split. "stratify" keeps the same spam/ham ratio in both parts.
    train_data, test_data = train_test_split(
        data, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=data["label"]
    )
    print(f"  train: {len(train_data):,} | test: {len(test_data):,}")

    # 2. Learn the vocabulary from a random sample of the TRAINING data only.
    #    Using test data here would leak information about the test.
    vectorizer = make_vectorizer()
    sample_size = min(len(train_data), VOCAB_SAMPLE_SIZE)
    vocab_sample = train_data["text"].sample(sample_size, random_state=RANDOM_SEED)
    start = time.time()
    vectorizer.fit(vocab_sample)
    print(f"  vocabulary: {len(vectorizer.vocabulary_):,} words and word pairs "
          f"(learned from {sample_size:,} messages in {time.time() - start:.1f}s)")

    # 3. Convert every message into numbers.
    start = time.time()
    X_train = vectorizer.transform(train_data["text"])
    X_test = vectorizer.transform(test_data["text"])
    print(f"  converted text to numbers in {time.time() - start:.1f}s")

    # 4. Train two algorithms and compare them.
    candidates = {
        # The Day 3 algorithm: counts how often each word appears per class.
        "Naive Bayes": MultinomialNB(alpha=0.1),
        # Learns one weight per word; usually a bit more accurate.
        "Logistic Regression": LogisticRegression(C=4.0, max_iter=1000, solver="liblinear"),
    }
    comparison = []
    trained = {}
    for name, model in candidates.items():
        start = time.time()
        model.fit(X_train, train_data["label"])
        seconds = round(time.time() - start, 1)
        scores = score_model(model, X_test, test_data["label"], positive_label)
        scores.update({"algorithm": name, "train_seconds": seconds})
        comparison.append(scores)
        trained[name] = model
        print(f"  {name:<20} accuracy {scores['accuracy']:.4f} | F1 {scores['f1']:.4f} | {seconds}s")

    # 5. Keep the algorithm with the higher F1 score.
    best = max(comparison, key=lambda result: result["f1"])
    print(f"  best: {best['algorithm']}")

    # One overall score can hide a weak spot, so also score each source
    # (e.g. SMS vs Email) separately with the winning model.
    best_model = trained[best["algorithm"]]
    test_predictions = best_model.predict(X_test)
    best["accuracy_by_source"] = {}
    for source in sorted(test_data["source"].unique()):
        rows = (test_data["source"] == source).to_numpy()
        accuracy = accuracy_score(test_data["label"][rows], test_predictions[rows])
        best["accuracy_by_source"][source] = round(accuracy, 4)
        print(f"    {source:<6} accuracy {accuracy:.4f} on {rows.sum():,} test messages")

    # 6. Save the cleaner + vectorizer + model together as one "pipeline", so
    #    the web app can pass raw text in and get a prediction out.
    pipeline = Pipeline([("tfidf", vectorizer), ("model", best_model)])
    joblib.dump(pipeline, model_file, compress=3)

    # Save the unseen test messages for the live stream (shortened to keep
    # the file small).
    stream = test_data.copy()
    stream["text"] = stream["text"].astype(str).str.slice(0, STREAM_TEXT_LIMIT)
    stream.to_csv(stream_file, index=False)
    print(f"  saved {model_file.name} and {len(stream):,} stream messages")

    return {
        "task": task_name,
        "rows_total": len(data),
        "rows_train": len(train_data),
        "rows_test": len(test_data),
        "label_counts": data["label"].value_counts().to_dict(),
        "sources": data["source"].value_counts().to_dict(),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "best": best,
        "comparison": comparison,
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    tasks = [
        ("spam", load_spam_data, "spam", SPAM_MODEL_FILE, SPAM_STREAM_FILE),
        ("sentiment", load_sentiment_data, "positive", SENTIMENT_MODEL_FILE, SENTIMENT_STREAM_FILE),
    ]

    # Keep results from an earlier run, so training one task doesn't erase the other.
    results = json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else {}

    for task_name, load_function, positive_label, model_file, stream_file in tasks:
        try:
            data = load_function()
        except FileNotFoundError as error:
            print(f"\n=== {task_name.upper()} === skipped{error}")
            continue
        results[task_name] = train_task(task_name, data, positive_label, model_file, stream_file)
        results[task_name]["trained_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    METRICS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {METRICS_FILE}")


if __name__ == "__main__":
    main()
