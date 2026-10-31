// ---------------------------------------------------------------------------
// TextPulse browser engine: the whole server, running inside the browser.
//
// The live site is a free Hugging Face *static* Space, which can't run Python.
// So this file does in JavaScript what app/main.py does in Python:
//   * loads the two trained models (exported by scripts/export_for_browser.py)
//   * replays unseen test messages and classifies each one
//   * answers the same questions app.js asks the server (status, analyse, ...)
//
// app.js uses this engine when it is loaded, and the FastAPI server otherwise.
// Every visitor gets their own stream, so pausing only affects your own screen.
// ---------------------------------------------------------------------------

(function () {
  const DATA = "data/";
  const START_SPEED = 20;       // messages per second, like STREAM_START_SPEED
  const TICK_MS = 50;           // the stream loop runs 20 times per second

  // ---- clean_text(): the same steps as src/clean_text.py -------------------
  const LINK = /https?:\/\/\S+|www\.\S+/g;
  const USERNAME = /@[\p{L}\p{N}_]+/gu;   // Python's \w also matches non-English letters
  const HTML_TAG = /<[^>]+>/g;
  const NOT_A_LETTER = /[^a-z' ]+/g;
  const EXTRA_SPACES = /\s+/g;
  const TOKEN = /[a-z][a-z']+/g;          // same token_pattern as training

  function cleanText(text) {
    return String(text).toLowerCase()
      .replace(LINK, " url ")
      .replace(USERNAME, " user ")
      .replace(HTML_TAG, " ")
      .replace(NOT_A_LETTER, " ")
      .replace(EXTRA_SPACES, " ")
      .trim();
  }

  // ---- One model: TF-IDF numbers -> sigmoid -> label ------------------------
  class Analyzer {
    constructor(model) {
      this.labels = model.labels;           // e.g. ["ham", "spam"]
      this.bias = model.bias;
      this.idf = model.idf;
      this.weights = model.weights;
      this.index = new Map(model.words.map((word, i) => [word, i]));  // word -> column
      this.words = model.words;
    }

    analyze(text, top = 4) {
      // 1. Words and word pairs, exactly like TfidfVectorizer(ngram_range=(1, 2))
      const tokens = cleanText(text).match(TOKEN) || [];
      const terms = tokens.slice();
      for (let i = 0; i + 1 < tokens.length; i++) terms.push(tokens[i] + " " + tokens[i + 1]);

      // 2. Count the ones the model knows
      const counts = new Map();
      for (const term of terms) {
        const id = this.index.get(term);
        if (id !== undefined) counts.set(id, (counts.get(id) || 0) + 1);
      }

      // 3. TF-IDF: (1 + log(count)) x idf, then scale so the lengths squared add up to 1
      const values = [];
      let squares = 0;
      for (const [id, count] of counts) {
        const value = (1 + Math.log(count)) * this.idf[id];
        values.push([id, value]);
        squares += value * value;
      }
      const length = Math.sqrt(squares) || 1;

      // 4. Score = sum(value x weight) + bias; probability = sigmoid(score)
      let score = this.bias;
      const contributions = values.map(([id, value]) => {
        const contribution = (value / length) * this.weights[id];
        score += contribution;
        return [id, contribution];
      });
      const probability = 1 / (1 + Math.exp(-score));     // chance of the second label
      const winner = probability >= 0.5 ? 1 : 0;

      // 5. The words that pushed toward the winning label, biggest first
      const direction = winner === 1 ? 1 : -1;
      const top_words = contributions
        .map(([id, c]) => ({ word: this.words[id], weight: Math.round(c * direction * 1000) / 1000 }))
        .filter((w) => w.weight > 0)
        .sort((a, b) => b.weight - a.weight)
        .slice(0, top);

      return {
        label: this.labels[winner],
        confidence: Math.round((winner === 1 ? probability : 1 - probability) * 10000) / 10000,
        top_words,
      };
    }
  }

  // ---- Safe display mode: same rules as display_text() in app/main.py ----
  let explicitPattern = null;
  const HIDDEN_TEXT = "Message hidden in safe display mode (explicit content)";
  const isExplicit = (text) => explicitPattern !== null && explicitPattern.test(text);
  const displayText = (text) => (isExplicit(text) ? HIDDEN_TEXT : text.slice(0, 280).replace(USERNAME, "@user"));

  // ---- Shared state, like the variables at the top of app/main.py ----------
  const analyzers = {};
  const feeds = {};            // task -> { messages, position }
  const stream = { paused: false, speed: START_SPEED };
  let metrics = {};
  let listener = null;
  let ready = null;

  const getJson = (name) => fetch(DATA + name).then((r) => {
    if (!r.ok) throw new Error(`Could not load ${name}`);
    return r.json();
  });

  // Shuffle a list in place (Fisher-Yates), so each visit sees a different order
  function shuffle(list) {
    for (let i = list.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [list[i], list[j]] = [list[j], list[i]];
    }
    return list;
  }

  function loadEverything() {
    return Promise.all([
      getJson("metrics.json"),
      getJson("spam_model.json"), getJson("stream_spam.json"),
      getJson("sentiment_model.json"), getJson("stream_sentiment.json"),
    ]).then(([loadedMetrics, spamModel, spamStream, sentimentModel, sentimentStream]) => {
      const { safe_display, ...scores } = loadedMetrics;
      metrics = scores;
      if (safe_display) {
        explicitPattern = new RegExp(
          "\\b(?:" + safe_display.words.join("|") + ")\\b|\\b(?:" + safe_display.word_starts.join("|") + ")\\w*", "i");
      }
      analyzers.spam = new Analyzer(spamModel);
      analyzers.sentiment = new Analyzer(sentimentModel);
      feeds.spam = { messages: shuffle(spamStream), position: 0 };
      feeds.sentiment = { messages: shuffle(sentimentStream), position: 0 };
    });
  }

  function nextMessage(task) {
    const feed = feeds[task];
    if (feed.position >= feed.messages.length) { shuffle(feed.messages); feed.position = 0; }
    return feed.messages[feed.position++];
  }

  // ---- The live stream: like run_live_stream() in app/main.py --------------
  let nextTask = 0;
  let owed = 0;               // messages due but not yet sent (speed x time)

  function classifyOne() {
    const task = nextTask++ % 2 === 0 ? "spam" : "sentiment";   // alternate the two
    const [trueLabel, text, source] = nextMessage(task);
    const started = performance.now();
    const result = analyzers[task].analyze(text);
    const latency = performance.now() - started;
    const hidden = isExplicit(text);   // only changes what we SHOW, not the prediction
    return {
      type: "message",
      task,
      source,
      text: displayText(text),
      hidden,
      prediction: result.label,
      confidence: result.confidence,
      top_words: hidden ? [] : result.top_words,
      true_label: trueLabel,
      correct: result.label === trueLabel,
      latency_ms: Math.round(latency * 100) / 100,
      time: Date.now() / 1000,
    };
  }

  function tick() {
    if (!stream.paused && listener) {
      owed += (stream.speed * TICK_MS) / 1000;     // e.g. 20 msg/s x 0.05 s = 1 message
      while (owed >= 1) { listener(classifyOne()); owed -= 1; }
    }
    setTimeout(tick, TICK_MS);
  }

  // ---- The same questions app.js asks the server ---------------------------
  window.TextPulseEngine = {
    name: "browser",
    async status() {
      ready = ready || loadEverything();
      await ready;
      return {
        models: Object.keys(analyzers), paused: stream.paused, speed: stream.speed,
        totals: {}, recent: [], chart_history: [],
      };
    },
    async metrics() { await ready; return metrics; },
    async analyze(text) {
      await ready;
      const results = {};
      for (const task in analyzers) results[task] = analyzers[task].analyze(text);
      return results;
    },
    async setSpeed(messagesPerSecond) { stream.speed = messagesPerSecond; },
    async pause() { stream.paused = true; },
    async resume() { stream.paused = false; },
    listen(onEvent, onOpen) {
      listener = onEvent;
      onOpen();
      tick();
    },
  };
})();
