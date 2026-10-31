"""
Bulk scoring benchmark: run the trained models over EVERY record in the
datasets (about 1.64 million messages) and measure how fast that is.

Run from the project folder, after training:
    python -m scripts.benchmark

The live stream shows one message at a time so you can watch it. This script
shows the other side of a real system: processing a large backlog in batches.
"""

import json
import time

from src.analyzer import TextAnalyzer
from src.config import BENCHMARK_FILE, SENTIMENT_MODEL_FILE, SPAM_MODEL_FILE
from src.load_data import load_sentiment_data, load_spam_data

BATCH_SIZE = 50_000   # score 50,000 messages per call; faster than one by one


def score_in_batches(analyzer, texts):
    """Predict every text, one batch at a time. Returns the number scored."""
    scored = 0
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        analyzer.predict_many(batch)
        scored += len(batch)
        print(f"    {scored:,} / {len(texts):,}", end="\r")
    print()
    return scored


def main():
    jobs = [
        ("spam", SPAM_MODEL_FILE, load_spam_data),
        ("sentiment", SENTIMENT_MODEL_FILE, load_sentiment_data),
    ]
    total_records, total_seconds, details = 0, 0.0, {}

    for task, model_file, load_function in jobs:
        if not model_file.exists():
            print(f"Skipping {task}: no trained model. Run  python -m src.train  first.")
            continue
        texts = load_function()["text"].astype(str).tolist()
        analyzer = TextAnalyzer(model_file)
        print(f"Scoring {len(texts):,} {task} records...")
        started = time.time()
        scored = score_in_batches(analyzer, texts)
        seconds = time.time() - started
        print(f"  {scored:,} records in {seconds:.1f}s ({scored / seconds:,.0f} per second)")
        details[task] = {"records": scored, "seconds": round(seconds, 1)}
        total_records += scored
        total_seconds += seconds

    if total_records:
        result = {
            "records": total_records,
            "seconds": round(total_seconds, 1),
            "records_per_second": round(total_records / total_seconds),
            "by_task": details,
        }
        BENCHMARK_FILE.write_text(json.dumps(result, indent=2))
        print(f"\nTotal: {total_records:,} records in {total_seconds:.1f}s. Saved to {BENCHMARK_FILE}")


if __name__ == "__main__":
    main()
