# How TextPulse Works: a Beginner's Guide

This guide walks through the project in the order data flows through it. If you understood the Day 3 spam filter (counting words per class), you already know the core idea. TextPulse adds three things on top: better numbers for words (TF-IDF), a second algorithm to compare against (Logistic Regression), and a web server that runs everything live.

## The big picture

```
Kaggle CSVs ─► load_data.py ─► train.py ─► models/ ─► app/main.py ─► dashboard in your browser
 (raw files)   (one table)     (learn)     (saved)    (live server)   (charts, feed, KPIs)
```

There are two separate phases:

1. **Training (offline, run once).** `python -m src.train` reads the data, learns, and saves the models to the `models/` folder.
2. **Serving (online, runs continuously).** `uvicorn app.main:app` loads the saved models and uses them. It never retrains.

This split is how most real ML systems work. Training is slow and happens occasionally; predicting is fast and happens all the time.

---

## 1. `src/config.py`: every setting in one place

```python
PROJECT_DIR = Path(__file__).resolve().parent.parent
```

`__file__` is the path of this file, `src/config.py`. `.parent` is the `src` folder, and `.parent` again is the project folder. Building every path from here means the code works whichever folder you run it from, on Windows or Linux.

```python
DATA_DIR = PROJECT_DIR / "data"
```

With `pathlib`, the `/` operator joins paths, so you never have to worry about `\` vs `/`.

The rest of the file is plain constants such as `TEST_SIZE = 0.2`. Other files import them, so a change here applies everywhere. It's the same idea as `appsettings.json` in ASP.NET.

---

## 2. `src/clean_text.py`: making text consistent

Without cleaning, "FREE", "Free" and "free!!!" would be three different words to the model.

```python
LINK = re.compile(r"https?://\S+|www\.\S+")
```

A regular expression that matches links. `https?` means "http" with an optional "s". `\S+` means "one or more characters that aren't spaces". The `|` means "or". It's compiled once at the top of the file, because the function runs 1.6 million times.

```python
text = LINK.sub(" url ", text)
```

Replaces every link with the word `url`. The exact link changes every time, but the fact that a message *has* a link is a strong spam signal, so we keep that fact as a word.

```python
text = NOT_A_LETTER.sub(" ", text)   # pattern: [^a-z' ]+
```

`[^...]` means "any character NOT in this list". So this removes everything except letters, apostrophes and spaces: digits, punctuation and emojis all go.

**Example:** `"FREE entry!!! Visit http://win.com now @john"` becomes `"free entry visit url now user"`.

Unlike the Day 3 notebook, we **keep** small words like "not". For sentiment, "good" and "not good" mean opposite things.

---

## 3. `src/load_data.py`: one standard table

Each Kaggle file has a different layout. These functions turn all of them into the same three columns: `text`, `label` and `source`.

```python
sms = pd.read_csv(SMS_SPAM_FILE, encoding="latin-1", usecols=[0, 1])
```

- `encoding="latin-1"`: this file wasn't saved as UTF-8. Without this you'd get a `UnicodeDecodeError`.
- `usecols=[0, 1]`: read only the first two columns. The other three are almost empty.

```python
enron["text"] = enron["Subject"].fillna("") + " " + enron["Message"].fillna("")
```

Joins the subject and body of each email. `fillna("")` turns missing values into empty text, because adding text to a missing value would cause an error.

```python
combined = combined.drop_duplicates(subset="text")
```

Spammers resend the same message. If one copy lands in training and another in testing, the model has effectively seen the test question before, and the score looks better than it really is. This is **data leakage**, and it removed 3,625 duplicates here.

```python
tweets = pd.read_csv(SENTIMENT_FILE, encoding="latin-1", header=None,
                     names=column_names, usecols=["target", "text"])
tweets["label"] = tweets["target"].map({0: "negative", 4: "positive"})
```

Sentiment140 has no header row, so we supply the column names. Its labels are numbers (0 for negative, 4 for positive), and `.map()` turns them into words.

---

## 4. `src/train.py`: learning

### Step 1: split

```python
train_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["label"])
```

- 80% of the data is for learning and 20% is hidden away for testing. Testing on data the model learned from would be like grading students on questions they've already seen.
- `random_state=42` makes the split the same every run, so results are reproducible.
- `stratify` keeps the spam/ham ratio the same in both parts.

### Step 2: TF-IDF, turning words into numbers

Day 3 used raw **counts**. TF-IDF improves on counts in two ways:

- **TF (term frequency):** a word used more often in a message gets a higher number.
- **IDF (inverse document frequency):** a word that appears in almost every message, like "the", gets a **lower** number, because it tells us nothing.

```python
TfidfVectorizer(
    preprocessor=clean_text,         # clean each message first
    token_pattern=r"[a-z][a-z']+",   # a word = 2+ letters (apostrophes allowed)
    ngram_range=(1, 2),              # single words AND pairs: "not", "good", "not good"
    min_df=3,                        # ignore words seen in fewer than 3 messages
    max_features=200_000,            # keep the 200,000 most common
    sublinear_tf=True,               # 10 x "free" counts less than 10 times as much
)
```

Each message becomes a list of 200,000 numbers, almost all zero. Python stores only the non-zero ones in a **sparse matrix**, so it fits in memory.

```python
vectorizer.fit(vocab_sample)     # learn the word list from 300,000 training messages
X_train = vectorizer.transform(train_data["text"])
```

`fit` learns *which* words exist, and `transform` converts text into numbers. We fit on a sample because collecting every word pair from 1.28 million tweets would need many gigabytes of memory, and the sample gives almost the same word list. We fit on **training** data only; looking at test data here would be leakage again.

### Step 3: two algorithms, compared fairly

| | Naive Bayes | Logistic Regression |
|---|---|---|
| Idea | How often does each word appear in each class? (Day 3) | Learn one weight per word that best separates the classes |
| Speed | Very fast | Slower |
| Usually | Good | Slightly more accurate |

```python
model.fit(X_train, train_data["label"])
scores = score_model(model, X_test, test_data["label"], positive_label)
```

Both are trained on exactly the same data and scored on exactly the same test set. The winner is the one with the higher **F1** score.

### Step 4: the scores

Using spam as the example:

- **Accuracy:** of all messages, how many did we get right?
- **Precision:** of the messages we called spam, how many really were spam? Low precision means real emails end up in the spam folder.
- **Recall:** of the real spam, how much did we catch? Low recall means spam gets through.
- **F1:** one number that balances precision and recall.

On the spam test set the confusion matrix is:

|  | predicted ham | predicted spam |
|---|---|---|
| **really ham** | 4,027 ✓ | 58 (good email sent to spam) |
| **really spam** | 88 (spam that got through) | 2,960 ✓ |

### Step 5: save

```python
pipeline = Pipeline([("tfidf", vectorizer), ("model", best_model)])
joblib.dump(pipeline, model_file, compress=3)
```

A `Pipeline` chains the vectorizer and the model together, so later we can pass in raw text and get a prediction out in one call. `joblib.dump` saves it to a file.

---

## 5. `src/analyzer.py`: predicting and explaining one message

```python
numbers = self.vectorizer.transform([text])
probabilities = self.model.predict_proba(numbers)[0]   # e.g. [0.02, 0.98]
winner = int(probabilities.argmax())                   # index of the bigger one: 1
```

`predict_proba` returns one probability per class. `argmax()` picks the position of the largest.

**Explaining the decision.** Both algorithms are *linear*: the final score is just the sum of (word value × word weight). So each word's share of the decision is easy to compute:

```python
word_ids = numbers.indices                       # which words this message contains
contributions = numbers.data * self.weights[word_ids]
```

`numbers.indices` and `numbers.data` are the sparse matrix's stored positions and values: just the handful of words in this message, out of 200,000. Multiplying by each word's weight gives how hard that word pushed. For Naive Bayes, the weight is exactly Day 3's log-likelihood ratio, log P(word|spam) − log P(word|ham).

---

## 6. `app/main.py`: the live server

FastAPI works much like ASP.NET Core Web API:

| ASP.NET Core | FastAPI |
|---|---|
| `[HttpGet("api/status")]` on a controller method | `@app.get("/api/status")` on a function |
| A request DTO class with `[Required]`, `[MaxLength]` | A Pydantic class with `Field(min_length=1, max_length=5000)` |
| Returning an object, serialised to JSON | Returning a dict, serialised to JSON |
| Swagger UI | Automatic, at `/docs` |

### The real-time engine

```python
async def run_live_stream():
    task_order = cycle(list(analyzers))     # spam, sentiment, spam, sentiment, ...
    while True:
        task = next(task_order)
        text, true_label, source = next(feeds[task])   # next unseen test message
        result = analyzers[task].analyze(text)          # classify it
        ...
        await send_to_all_dashboards(event)             # push to every browser
        await asyncio.sleep(1 / stream["speed"])        # 20 msg/s means wait 0.05 s
```

- `cycle` from `itertools` repeats a list forever, so the two models take turns.
- `feeds[task]` is a **generator** that yields test messages endlessly and reshuffles at the end.
- `async` / `await` let the server wait without blocking. While this loop sleeps, the same server keeps answering API requests. It's like `async Task` in C#.

### WebSockets vs normal requests

A normal request works like this: the browser asks, the server answers, and the connection is done. For live data the browser would have to keep asking ("anything new?") many times a second.

A **WebSocket** stays open, so the server can **push** each result the moment it exists:

```python
@app.websocket("/ws/live")
async def live_updates(connection: WebSocket):
    await connection.accept()
    dashboards.add(connection)        # remember this browser
```

`send_to_all_dashboards` loops over every remembered connection and sends the result. If a send fails, that browser has closed, so we forget it.

### Starting and stopping

```python
@asynccontextmanager
async def lifespan(app):
    load_models()                                    # runs at startup
    engine = asyncio.create_task(run_live_stream())  # start the stream in the background
    yield                                            # server runs here
    engine.cancel()                                  # runs at shutdown
```

---

## 7. `app/static/app.js`: the dashboard

```javascript
const socket = new WebSocket(`${protocol}://${location.host}/ws/live`);
socket.onmessage = (message) => {
  const event = JSON.parse(message.data);
  countEvent(event);         // update totals and chart buckets
  waitingEvents.push(event); // queue it for drawing
};
```

At 200 messages per second, redrawing the page for every message would make the browser stutter. So results wait in a queue, and `requestAnimationFrame` draws them together once per screen refresh, about 60 times a second.

```javascript
function escapeHtml(text) { ... }
```

Messages come from real spammers. If one contained `<script>`, inserting it straight into the page would run that code. Escaping turns it into harmless visible text. **Always escape text you didn't write yourself.**

```javascript
socket.onclose = () => { setStatus("offline", ...); setTimeout(connect, 2000); };
```

If the server restarts, the dashboard reconnects by itself after two seconds.

---

## Where to experiment

| Try this | Where | What you'll learn |
|---|---|---|
| Set `ngram_range=(1, 1)` and retrain | `src/train.py` | How much word pairs ("not good") help sentiment |
| Change `MultinomialNB(alpha=0.1)` to `alpha=1.0` | `src/train.py` | The effect of Laplace smoothing strength (Day 3) |
| Lower `MAX_WORDS` to 20,000 | `src/config.py` | The trade-off between model size and accuracy |
| Set `SAFE_DISPLAY = False` | `src/config.py` | See the raw data the model really learns from |
| Paste a modern phishing SMS into the dashboard | browser | Why training data that's 20 years old is a limitation |
