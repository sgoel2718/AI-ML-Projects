"""
Read the downloaded Kaggle files into one simple, standard table.

Every dataset comes in a different shape. These functions turn each one into
the same three columns, so the rest of the project never has to care where
the data came from:

    text    the message itself
    label   the answer we want to predict ("spam"/"ham" or "positive"/"negative")
    source  where the message came from ("SMS", "Email" or "Tweet")
"""

import pandas as pd

from src.config import ENRON_SPAM_FILE, SENTIMENT_FILE, SMS_SPAM_FILE


def check_file_exists(path):
    """Stop with a clear message if a dataset hasn't been downloaded yet."""
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing dataset: {path.name}\n"
            f"Run  python scripts/download_data.py  or see data/README.md.\n"
        )


def load_sms_spam():
    """SMS Spam Collection (Kaggle): 5,572 text messages."""
    check_file_exists(SMS_SPAM_FILE)
    # The file has two useful columns, v1 (label) and v2 (text), plus three
    # mostly empty ones. usecols=[0, 1] keeps only the first two.
    # encoding="latin-1" is needed because the file isn't saved as UTF-8.
    sms = pd.read_csv(SMS_SPAM_FILE, encoding="latin-1", usecols=[0, 1])
    sms.columns = ["label", "text"]
    sms["source"] = "SMS"
    return sms


def load_enron_spam():
    """Enron-Spam (preprocessed by M. Wiechmann): 33,716 emails."""
    check_file_exists(ENRON_SPAM_FILE)
    enron = pd.read_csv(ENRON_SPAM_FILE)
    # An email's subject line is useful evidence too ("You WON!!!"), so we join
    # subject and body into one text. fillna("") avoids errors on empty parts.
    enron["text"] = enron["Subject"].fillna("") + " " + enron["Message"].fillna("")
    enron["label"] = enron["Spam/Ham"]          # already "spam" or "ham"
    enron["source"] = "Email"
    return enron[["label", "text", "source"]]


def load_spam_data():
    """Both spam datasets combined: about 39,000 SMS and emails."""
    combined = pd.concat([load_sms_spam(), load_enron_spam()], ignore_index=True)

    # The same message can appear more than once (spammers resend). If one copy
    # lands in training and another in testing, the test is too easy and the
    # results look better than they really are. That's a form of data leakage.
    before = len(combined)
    combined = combined.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"  removed {before - len(combined):,} duplicate messages")
    return combined


def load_sentiment_data():
    """Sentiment140 (Kaggle): 1,600,000 tweets labelled negative or positive."""
    check_file_exists(SENTIMENT_FILE)
    # The file has no header row, so we name the six columns ourselves and
    # read only the two we need (this halves the memory used).
    column_names = ["target", "id", "date", "query", "user", "text"]
    tweets = pd.read_csv(
        SENTIMENT_FILE,
        encoding="latin-1",
        header=None,
        names=column_names,
        usecols=["target", "text"],
    )
    # In this dataset 0 means negative and 4 means positive.
    tweets["label"] = tweets["target"].map({0: "negative", 4: "positive"})
    tweets = tweets.dropna(subset=["label"])     # drop any unexpected codes
    tweets["source"] = "Tweet"
    return tweets[["label", "text", "source"]]
