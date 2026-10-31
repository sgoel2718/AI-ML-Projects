// ---------------------------------------------------------------------------
// TextPulse dashboard logic, in plain JavaScript (no framework).
//
// 1. Ask the server for the current totals and model scores   (fetch)
// 2. Open a WebSocket and receive every new result as it happens
// 3. Update the numbers, charts and feed a few times per second
// 4. Let the user analyse their own text
// ---------------------------------------------------------------------------

const FEED_ROWS = 60;          // rows kept in the live feed table
const CHART_SECONDS = 60;      // how much history the charts show

// Running totals for this session, one entry per model
const totals = {
  spam: { processed: 0, flagged: 0, correct: 0 },
  sentiment: { processed: 0, flagged: 0, correct: 0 },
};
let loadedModels = [];
let feedFilter = "all";
let waitingEvents = [];        // results received but not yet drawn

// One "bucket" per second for the charts: { second: {spam, spamTotal, positive, sentimentTotal} }
const perSecond = new Map();

const $ = (id) => document.getElementById(id);
const percent = (part, whole) => (whole ? (100 * part / whole).toFixed(1) + "%" : "–");
const formatNumber = (n) => n.toLocaleString("en-IN");

// Below 65% the model is close to guessing, so we say so instead of hiding it.
const LOW_CONFIDENCE = 0.65;
const sureText = (confidence) =>
  `${Math.round(confidence * 100)}% sure${confidence < LOW_CONFIDENCE ? " · low" : ""}`;

// Escape text before putting it into HTML, so a message containing <script>
// is shown as text instead of being run. Always do this with untrusted text.
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------
function cssColor(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function makeChart(canvasId, colorName, label) {
  const color = cssColor(colorName);
  return new Chart($(canvasId), {
    type: "line",
    data: {
      labels: [],
      datasets: [{ label, data: [], borderColor: color, backgroundColor: color + "22", fill: true, tension: 0.35, pointRadius: 0, borderWidth: 2 }],
    },
    options: {
      animation: false,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` ${c.parsed.y.toFixed(1)}%` } } },
      scales: {
        x: { ticks: { color: cssColor("--muted"), maxTicksLimit: 7 }, grid: { display: false } },
        y: { min: 0, max: 100, ticks: { color: cssColor("--muted"), callback: (v) => v + "%" }, grid: { color: cssColor("--line") } },
      },
    },
  });
}

const spamChart = makeChart("spam-chart", "--spam", "Spam rate");
const sentimentChart = makeChart("sentiment-chart", "--positive", "Positive tweets");

function updateCharts() {
  const now = Math.floor(Date.now() / 1000);
  const labels = [], spamValues = [], positiveValues = [];
  // Plot completed seconds only: the current second is still filling up, and a
  // half-full bucket with one or two messages would swing to 0% or 100%.
  for (let second = now - CHART_SECONDS; second < now; second++) {
    const bucket = perSecond.get(second);
    labels.push(second === now - 1 ? "now" : `-${now - 1 - second}s`);
    spamValues.push(bucket && bucket.spamTotal ? (100 * bucket.spam) / bucket.spamTotal : null);
    positiveValues.push(bucket && bucket.sentimentTotal ? (100 * bucket.positive) / bucket.sentimentTotal : null);
  }
  // Forget buckets older than the chart window
  for (const second of perSecond.keys()) if (second < now - CHART_SECONDS - 1) perSecond.delete(second);

  spamChart.data.labels = labels;
  spamChart.data.datasets[0].data = spamValues;
  spamChart.options.spanGaps = true;
  spamChart.update();
  sentimentChart.data.labels = labels;
  sentimentChart.data.datasets[0].data = positiveValues;
  sentimentChart.options.spanGaps = true;
  sentimentChart.update();
}

// ---------------------------------------------------------------------------
// Handling one new result
// ---------------------------------------------------------------------------
function countEvent(event) {
  const t = totals[event.task];
  t.processed += 1;
  t.correct += event.correct ? 1 : 0;
  const isFlagged = event.prediction === "spam" || event.prediction === "positive";
  t.flagged += isFlagged ? 1 : 0;

  const second = Math.floor(Date.now() / 1000);
  const bucket = perSecond.get(second) || { spam: 0, spamTotal: 0, positive: 0, sentimentTotal: 0 };
  if (event.task === "spam") { bucket.spamTotal += 1; bucket.spam += isFlagged ? 1 : 0; }
  else { bucket.sentimentTotal += 1; bucket.positive += isFlagged ? 1 : 0; }
  perSecond.set(second, bucket);
}

function feedRow(event) {
  const time = new Date(event.time * 1000).toLocaleTimeString("en-GB");
  const words = event.top_words.map((w) => `<span class="word">${escapeHtml(w.word)}</span>`).join("");
  const confidence = Math.round(event.confidence * 100);
  const row = document.createElement("tr");
  row.className = "fresh";
  row.dataset.task = event.task;
  row.hidden = feedFilter !== "all" && feedFilter !== event.task;
  row.innerHTML = `
    <td class="time">${time}</td>
    <td class="source">${escapeHtml(event.source)}</td>
    <td class="message"><span class="message-text ${event.hidden ? "hidden-text" : ""}">${escapeHtml(event.text)}</span><div class="words">${words}</div></td>
    <td><span class="badge ${event.prediction}">${event.prediction}</span>
        <span class="confidence"><span style="width:${confidence}%"></span></span>
        <span class="confidence-text ${event.confidence < LOW_CONFIDENCE ? "low" : ""}">${sureText(event.confidence)}</span></td>
    <td class="check ${event.correct ? "right" : "wrong"}" title="Real label: ${event.true_label}">${event.correct ? "✓" : "✗"}</td>`;
  return row;
}

// ---------------------------------------------------------------------------
// Drawing. At 200 messages per second we can't redraw for every message,
// so results wait in a queue and are drawn together on each screen refresh.
// ---------------------------------------------------------------------------
function drawWaitingEvents() {
  if (waitingEvents.length) {
    const feed = $("feed");
    const newest = waitingEvents.slice(-FEED_ROWS);
    for (const event of newest) feed.prepend(feedRow(event));
    while (feed.rows.length > FEED_ROWS) feed.deleteRow(-1);
    waitingEvents = [];
    drawTotals();
  }
  requestAnimationFrame(drawWaitingEvents);
}

function drawTotals() {
  const all = totals.spam.processed + totals.sentiment.processed;
  $("kpi-total").textContent = formatNumber(all);
  $("kpi-spam").textContent = loadedModels.includes("spam") ? percent(totals.spam.flagged, totals.spam.processed) : "–";
  $("kpi-positive").textContent = loadedModels.includes("sentiment") ? percent(totals.sentiment.flagged, totals.sentiment.processed) : "–";
  if (!loadedModels.includes("sentiment")) $("kpi-positive-note").textContent = "sentiment model not trained";
  const spamAccuracy = percent(totals.spam.correct, totals.spam.processed);
  const sentimentAccuracy = percent(totals.sentiment.correct, totals.sentiment.processed);
  $("kpi-accuracy").textContent = loadedModels.length > 1 ? `${spamAccuracy} · ${sentimentAccuracy}` : spamAccuracy;
  $("kpi-accuracy").classList.toggle("two-values", loadedModels.length > 1);
  $("kpi-accuracy-note").textContent = loadedModels.length > 1 ? "spam · sentiment, on unseen data" : "spam model, on unseen data";
}

// Messages per second, averaged over the last 5 complete seconds. It uses the
// same per-second buckets as the charts, so it's right even just after a refresh.
function drawThroughput() {
  const now = Math.floor(Date.now() / 1000);
  let count = 0;
  for (let second = now - 5; second < now; second++) {
    const bucket = perSecond.get(second);
    if (bucket) count += bucket.spamTotal + bucket.sentimentTotal;
  }
  $("kpi-rate").textContent = (count / 5).toFixed(1);
}

// ---------------------------------------------------------------------------
// Live connection (WebSocket). Reconnects automatically if the server restarts.
// ---------------------------------------------------------------------------
function setStatus(kind, text) {
  $("status").className = `status ${kind}`;
  $("status-text").textContent = text;
}

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws/live`);
  socket.onopen = () => setStatus($("pause").dataset.paused === "true" ? "paused" : "live", $("pause").dataset.paused === "true" ? "Paused" : "Live");
  socket.onmessage = (message) => {
    const event = JSON.parse(message.data);
    countEvent(event);
    waitingEvents.push(event);
  };
  socket.onclose = () => {
    setStatus("offline", "Offline, retrying");
    setTimeout(connect, 2000);
  };
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
async function post(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  return response.json();
}

$("pause").addEventListener("click", async () => {
  const isPaused = $("pause").dataset.paused === "true";
  await post(isPaused ? "/api/stream/resume" : "/api/stream/pause");
  setPausedLook(!isPaused);
});

function setPausedLook(paused) {
  $("pause").dataset.paused = String(paused);
  $("pause").textContent = paused ? "Resume stream" : "Pause stream";
  setStatus(paused ? "paused" : "live", paused ? "Paused" : "Live");
}

$("speed").addEventListener("change", () => post("/api/stream/speed", { messages_per_second: Number($("speed").value) }));

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    feedFilter = tab.dataset.filter;
    document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-pressed", String(t === tab)));
    for (const row of $("feed").rows) row.hidden = feedFilter !== "all" && feedFilter !== row.dataset.task;
  });
});

// ---------------------------------------------------------------------------
// Analyse your own text
// ---------------------------------------------------------------------------
document.querySelectorAll("[data-sample]").forEach((chip) => {
  chip.addEventListener("click", () => {
    $("analyze-text").value = chip.dataset.sample;
    $("analyze-form").requestSubmit();
  });
});

$("analyze-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = $("analyze-text").value.trim();
  if (!text) return;
  const results = await post("/api/analyze", { text });
  const names = { spam: "Spam model", sentiment: "Sentiment model" };
  $("analyze-result").innerHTML = Object.entries(results).map(([task, r]) => `
    <div class="result">
      <div class="result-top">
        <span class="result-name">${names[task]}</span>
        <span><span class="badge ${r.label}">${r.label}</span> <span class="confidence-text ${r.confidence < LOW_CONFIDENCE ? "low" : ""}">${sureText(r.confidence)}</span></span>
      </div>
      <div class="words">${r.top_words.map((w) => `<span class="word">${escapeHtml(w.word)} +${w.weight}</span>`).join("") || '<span class="confidence-text">no strong words</span>'}</div>
    </div>`).join("");
});

// ---------------------------------------------------------------------------
// Model card: scores from training
// ---------------------------------------------------------------------------
function modelSection(title, m) {
  const best = m.best;
  const comparison = m.comparison.map((c) => `
    <tr><td>${c.algorithm}${c.algorithm === best.algorithm ? " ✓" : ""}</td>
        <td class="num">${(c.accuracy * 100).toFixed(2)}%</td><td class="num">${(c.f1 * 100).toFixed(2)}%</td><td class="num">${c.train_seconds}s</td></tr>`).join("");
  const bySource = Object.entries(best.accuracy_by_source || {}).map(([s, a]) => `${s} ${(a * 100).toFixed(1)}%`).join(" · ");
  return `
    <section class="model">
      <h3>${title}</h3>
      <p class="model-meta">${formatNumber(m.rows_total)} messages · ${formatNumber(m.rows_test)} held out for testing · ${best.algorithm}${bySource ? "<br>Accuracy by source: " + bySource : ""}</p>
      <div class="stats">
        <div class="stat"><b>${(best.accuracy * 100).toFixed(1)}%</b><span>accuracy</span></div>
        <div class="stat"><b>${(best.precision * 100).toFixed(1)}%</b><span>precision</span></div>
        <div class="stat"><b>${(best.recall * 100).toFixed(1)}%</b><span>recall</span></div>
        <div class="stat"><b>${(best.f1 * 100).toFixed(1)}%</b><span>F1</span></div>
      </div>
      <table class="compare"><tr><th>Algorithm</th><th>Accuracy</th><th>F1</th><th>Train</th></tr>${comparison}</table>
    </section>`;
}

async function loadModelCard() {
  const metrics = await (await fetch("/api/metrics")).json();
  let html = "";
  if (metrics.spam) html += modelSection("Spam detector", metrics.spam);
  if (metrics.sentiment) html += modelSection("Sentiment analyser", metrics.sentiment);
  if (metrics.benchmark) {
    const b = metrics.benchmark;
    html += `<div class="benchmark"><b>Bulk scoring:</b> ${formatNumber(b.records)} records in ${b.seconds}s (${formatNumber(b.records_per_second)} per second)</div>`;
  }
  $("model-card").innerHTML = html || '<p class="empty">No trained models yet. Run <code>python -m src.train</code>.</p>';
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------
async function start() {
  const status = await (await fetch("/api/status")).json();
  loadedModels = status.models;
  for (const task of loadedModels) Object.assign(totals[task], status.totals[task]);
  $("speed").value = String(status.speed);
  setPausedLook(status.paused);
  $("spam-chart-empty").hidden = loadedModels.includes("spam");
  $("sentiment-chart-empty").hidden = loadedModels.includes("sentiment");
  waitingEvents = status.recent;          // show the latest messages straight away
  // Refill the charts with the last 60 seconds the server remembers
  for (const bucket of status.chart_history || []) {
    const { second, ...counts } = bucket;
    perSecond.set(second, counts);
  }
  updateCharts();
  drawThroughput();
  drawTotals();
  loadModelCard();
  connect();
  requestAnimationFrame(drawWaitingEvents);
  setInterval(updateCharts, 1000);
  setInterval(drawThroughput, 1000);
}

start();
