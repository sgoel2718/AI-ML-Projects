"""
TextPulse web server.

Start it from the project folder:
    uvicorn app.main:app --reload
then open http://127.0.0.1:8000

What it does:
  * serves the dashboard page (app/static/index.html)
  * replays unseen test messages as a live stream, classifies each one, and
    pushes the result to every open dashboard through a WebSocket
  * offers a small REST API, so you can analyse your own text

Coming from ASP.NET Core? FastAPI is the same idea: decorators like
@app.get("/api/status") play the role of [HttpGet("api/status")] on a
controller method, and Pydantic classes play the role of your request DTOs.
"""

import asyncio
import json
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from itertools import cycle

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.analyzer import TextAnalyzer
from src.config import (
    BENCHMARK_FILE, EXPLICIT_WORD_STARTS, EXPLICIT_WORDS, METRICS_FILE, PROJECT_DIR, RANDOM_SEED,
    RECENT_EVENTS_KEPT, SAFE_DISPLAY,
    SENTIMENT_MODEL_FILE, SENTIMENT_STREAM_FILE, SPAM_MODEL_FILE,
    SPAM_STREAM_FILE, STREAM_MAX_SPEED, STREAM_START_SPEED,
)

STATIC_DIR = PROJECT_DIR / "app" / "static"

# Each task = (name, saved model, stream of unseen test messages)
TASKS = [
    ("spam", SPAM_MODEL_FILE, SPAM_STREAM_FILE),
    ("sentiment", SENTIMENT_MODEL_FILE, SENTIMENT_STREAM_FILE),
]

# ---------------------------------------------------------------------------
# Shared state. One copy for the whole server, used by every request.
# ---------------------------------------------------------------------------
analyzers = {}       # task name -> TextAnalyzer
feeds = {}           # task name -> endless iterator of test messages
dashboards = set()   # every open WebSocket connection
recent_events = deque(maxlen=RECENT_EVENTS_KEPT)  # newest messages, for new visitors
stream = {"paused": False, "speed": STREAM_START_SPEED}
totals = {}          # running counters per task

# Chart history: one bucket per second for the last 60 seconds, so a dashboard
# that is refreshed (or opened later) can draw the charts straight away.
CHART_SECONDS = 60
chart_history = {}   # second -> {"spam": 0, "spamTotal": 0, "positive": 0, "sentimentTotal": 0}


def add_to_chart_history(task, is_flagged):
    """Count one result in the bucket for the current second."""
    second = int(time.time())
    bucket = chart_history.setdefault(
        second, {"spam": 0, "spamTotal": 0, "positive": 0, "sentimentTotal": 0}
    )
    if task == "spam":
        bucket["spamTotal"] += 1
        bucket["spam"] += is_flagged
    else:
        bucket["sentimentTotal"] += 1
        bucket["positive"] += is_flagged
    # Forget buckets older than the chart window
    for old_second in [s for s in chart_history if s < second - CHART_SECONDS]:
        del chart_history[old_second]


# \b means "word boundary": \b(sex)\b matches "sex" but not "Essex".
# \w* after a word start matches the rest of the word: "porn" -> "pornography".
EXPLICIT_PATTERN = re.compile(
    r"\b(?:" + "|".join(EXPLICIT_WORDS) + r")\b"
    + r"|\b(?:" + "|".join(EXPLICIT_WORD_STARTS) + r")\w*",
    re.IGNORECASE,
)
HIDDEN_TEXT = "Message hidden in safe display mode (explicit content)"

# Tweets mention real people's accounts. We show "@user" instead of the real
# handle, the same way clean_text() treats it for the model.
USERNAME = re.compile(r"@\w+")


def display_text(text):
    """What the dashboard shows for a streamed message."""
    if is_explicit(text):
        return HIDDEN_TEXT
    return USERNAME.sub("@user", text[:280])


def is_explicit(text):
    """True if safe display mode is on and the text contains explicit words."""
    return SAFE_DISPLAY and EXPLICIT_PATTERN.search(text) is not None


def endless_messages(stream_file):
    """Yield test messages forever, reshuffling each time we reach the end."""
    messages = pd.read_csv(stream_file)
    round_number = 0
    while True:
        shuffled = messages.sample(frac=1, random_state=RANDOM_SEED + round_number)
        for row in shuffled.itertuples(index=False):
            yield row.text, row.label, row.source
        round_number += 1


def load_models():
    """Load every model that has been trained. Missing ones are skipped."""
    for task, model_file, stream_file in TASKS:
        if model_file.exists() and stream_file.exists():
            analyzers[task] = TextAnalyzer(model_file)
            feeds[task] = endless_messages(stream_file)
            totals[task] = {"processed": 0, "flagged": 0, "correct": 0}
            print(f"Loaded {task} model")
        else:
            print(f"No {task} model yet. Run: python -m src.train")


async def send_to_all_dashboards(event):
    """Push one result to every open dashboard; forget closed connections."""
    for connection in list(dashboards):
        try:
            await connection.send_json(event)
        except Exception:
            dashboards.discard(connection)


async def run_live_stream():
    """
    The real-time engine. Forever: take the next unseen message, classify it,
    update the counters, and push the result to the dashboards.
    """
    task_order = cycle(list(analyzers))   # alternate: spam, sentiment, spam, ...
    while True:
        if stream["paused"] or not analyzers:
            await asyncio.sleep(0.2)
            continue

        task = next(task_order)
        text, true_label, source = next(feeds[task])
        started = time.perf_counter()
        result = analyzers[task].analyze(text)
        latency_ms = (time.perf_counter() - started) * 1000

        is_correct = result["label"] == true_label
        positive_label = analyzers[task].labels[1]       # "spam" or "positive"
        totals[task]["processed"] += 1
        totals[task]["flagged"] += result["label"] == positive_label
        totals[task]["correct"] += is_correct
        add_to_chart_history(task, result["label"] == positive_label)

        hidden = is_explicit(text)   # only changes what we SHOW, not the prediction
        event = {
            "type": "message",
            "task": task,
            "source": source,
            "text": display_text(text),
            "hidden": hidden,
            "prediction": result["label"],
            "confidence": result["confidence"],
            "top_words": [] if hidden else result["top_words"],
            "true_label": true_label,
            "correct": is_correct,
            "latency_ms": round(latency_ms, 2),
            "time": time.time(),
        }
        recent_events.append(event)
        await send_to_all_dashboards(event)
        await asyncio.sleep(1 / stream["speed"])


@asynccontextmanager
async def lifespan(app):
    """Code before 'yield' runs when the server starts; after it, when it stops."""
    load_models()
    engine = asyncio.create_task(run_live_stream())
    yield
    engine.cancel()


app = FastAPI(title="TextPulse", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Pages and API
# ---------------------------------------------------------------------------
@app.get("/")
def dashboard_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    """Everything a freshly opened dashboard needs to draw itself."""
    return {
        "models": list(analyzers),
        "paused": stream["paused"],
        "speed": stream["speed"],
        "max_speed": STREAM_MAX_SPEED,
        "totals": totals,
        "recent": list(recent_events),
        "chart_history": [{"second": second, **bucket} for second, bucket in chart_history.items()],
    }


@app.get("/api/metrics")
def metrics():
    """Test scores from training, plus the bulk benchmark if it has been run."""
    result = json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else {}
    if BENCHMARK_FILE.exists():
        result["benchmark"] = json.loads(BENCHMARK_FILE.read_text())
    return result


class AnalyzeRequest(BaseModel):
    """The body of POST /api/analyze. FastAPI checks it automatically."""
    text: str = Field(min_length=1, max_length=5000)


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    """Run every loaded model on your own text."""
    return {task: analyzer.analyze(request.text) for task, analyzer in analyzers.items()}


class SpeedRequest(BaseModel):
    messages_per_second: int = Field(ge=1, le=STREAM_MAX_SPEED)


@app.post("/api/stream/speed")
def set_speed(request: SpeedRequest):
    stream["speed"] = request.messages_per_second
    return {"speed": stream["speed"]}


@app.post("/api/stream/pause")
def pause():
    stream["paused"] = True
    return {"paused": True}


@app.post("/api/stream/resume")
def resume():
    stream["paused"] = False
    return {"paused": False}


@app.websocket("/ws/live")
async def live_updates(connection: WebSocket):
    """A dashboard connects here and then receives every new result."""
    await connection.accept()
    dashboards.add(connection)
    try:
        while True:
            await connection.receive_text()   # we only wait here to notice a disconnect
    except WebSocketDisconnect:
        dashboards.discard(connection)
