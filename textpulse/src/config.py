"""
All project settings in one place.

Every other file imports its paths and numbers from here, so if you want to
change something (for example the stream speed), you only change it once.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
# __file__ is this file (src/config.py). .parent is "src", .parent again is
# the project folder. Building paths this way means the project works no
# matter which folder you run it from.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"      # raw datasets you download
MODELS_DIR = PROJECT_DIR / "models"  # trained models and results we create

# ---------------------------------------------------------------------------
# Raw dataset files (see data/README.md for where each one comes from)
# ---------------------------------------------------------------------------
SENTIMENT_FILE = DATA_DIR / "training.1600000.processed.noemoticon.csv"  # 1.6M tweets
SMS_SPAM_FILE = DATA_DIR / "spam.csv"                                    # 5,572 SMS
ENRON_SPAM_FILE = DATA_DIR / "enron_spam_data.csv"                       # 33,716 emails

# ---------------------------------------------------------------------------
# Files that training creates
# ---------------------------------------------------------------------------
SPAM_MODEL_FILE = MODELS_DIR / "spam_model.joblib"
SENTIMENT_MODEL_FILE = MODELS_DIR / "sentiment_model.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"        # accuracy, F1, etc.
BENCHMARK_FILE = MODELS_DIR / "benchmark.json"    # bulk scoring speed

# The test messages the model never saw while training. The live dashboard
# replays these, so every prediction you see is on genuinely unseen data.
SPAM_STREAM_FILE = MODELS_DIR / "stream_spam.csv"
SENTIMENT_STREAM_FILE = MODELS_DIR / "stream_sentiment.csv"

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2          # keep 20% of the data aside for testing
RANDOM_SEED = 42         # same seed = same split = same results every run

# Building the word list (vocabulary) from 1.28 million tweets with word pairs
# would need many gigabytes of memory. Learning the vocabulary from a large
# random sample gives almost the same word list with a fraction of the memory.
VOCAB_SAMPLE_SIZE = 300_000
MAX_WORDS = 200_000      # keep at most this many words and word pairs
MIN_WORD_COUNT = 3       # ignore words seen in fewer than 3 messages

# ---------------------------------------------------------------------------
# Live stream settings
# ---------------------------------------------------------------------------
STREAM_START_SPEED = 20      # messages per second when the server starts
STREAM_MAX_SPEED = 200       # upper limit for the speed control
STREAM_TEXT_LIMIT = 1_000    # characters of each message kept for the stream
RECENT_EVENTS_KEPT = 40      # messages a newly opened dashboard gets right away

# Real spam and tweets contain explicit content. For presentations, the
# dashboard HIDES any streamed message containing these words (it shows
# "hidden in safe display mode" instead). The models still read the original
# text, so predictions are unchanged. Set to False to show everything.
SAFE_DISPLAY = True

# Matched as whole words only, because shorter forms appear inside normal
# words ("anal" in "analysis", "cum" in "cumulative").
EXPLICIT_WORDS = [
    "sex", "sexy", "sexual", "sexually", "xxx", "nude", "nudes", "naked", "horny",
    "slut", "sluts", "whore", "whores", "milf", "teen", "teens", "cock", "cocks",
    "dick", "dicks", "pussy", "penis", "erection", "erections", "boobs", "tits",
    "cum", "anal", "rape", "raped", "virgin", "virgins", "hardcore", "adult",
    "naughty", "fuck", "fucking", "fucked", "shit", "bitch", "bitches", "hooker",
    "hookers", "stripper", "strippers", "housewives", "panties", "lingerie",
]
# Matched as the start of a word ("porn" also catches "porno", "pornography").
EXPLICIT_WORD_STARTS = [
    "porn", "erotic", "masturbat", "orgasm", "nympho", "bestial", "hentai",
    "blowjob", "fetish", "incest",
]
