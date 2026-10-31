"""
Download the three datasets into the data/ folder.

Run from the project folder:
    python -m scripts.download_data

Sentiment140 and the SMS Spam Collection come from Kaggle through the
`kagglehub` library. If Kaggle asks you to sign in, either:
  * follow the "Kaggle API token" steps in data/README.md, or
  * download the files in your browser using the links in data/README.md.
The Enron-Spam file is downloaded from its GitHub home.
"""

import shutil
import urllib.request
import zipfile
from pathlib import Path

from src.config import DATA_DIR, ENRON_SPAM_FILE, SENTIMENT_FILE, SMS_SPAM_FILE

KAGGLE_DATASETS = [
    # (Kaggle dataset id, file inside it, where we save it)
    ("kazanova/sentiment140", "training.1600000.processed.noemoticon.csv", SENTIMENT_FILE),
    ("uciml/sms-spam-collection-dataset", "spam.csv", SMS_SPAM_FILE),
]
ENRON_ZIP_URL = "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip"


def download_from_kaggle(dataset_id, file_name, destination):
    import kagglehub  # imported here so the other downloads work even without it

    print(f"Downloading {dataset_id} from Kaggle...")
    folder = Path(kagglehub.dataset_download(dataset_id))
    matches = list(folder.rglob(file_name))
    if not matches:
        raise FileNotFoundError(f"{file_name} not found in the downloaded dataset at {folder}")
    shutil.copy(matches[0], destination)
    print(f"  saved {destination.name}")


def download_enron():
    print("Downloading Enron-Spam from GitHub...")
    zip_path = DATA_DIR / "enron_spam_data.zip"
    urllib.request.urlretrieve(ENRON_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extract("enron_spam_data.csv", DATA_DIR)
    zip_path.unlink()
    print(f"  saved {ENRON_SPAM_FILE.name}")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for dataset_id, file_name, destination in KAGGLE_DATASETS:
        if destination.exists():
            print(f"Already have {destination.name}")
            continue
        try:
            download_from_kaggle(dataset_id, file_name, destination)
        except Exception as error:
            print(f"  Could not download {dataset_id}: {error}")
            print("  Download it in your browser instead; see data/README.md.")

    if ENRON_SPAM_FILE.exists():
        print(f"Already have {ENRON_SPAM_FILE.name}")
    else:
        download_enron()


if __name__ == "__main__":
    main()
