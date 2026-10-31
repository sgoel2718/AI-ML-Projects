# Deploying TextPulse live (Hugging Face static Space + GitHub Actions)

**Live site: [shubham2019-textpulse.hf.space](https://shubham2019-textpulse.hf.space)** · Space page: [huggingface.co/spaces/Shubham2019/textpulse](https://huggingface.co/spaces/Shubham2019/textpulse)

## Why a static Space

Hugging Face Spaces come in two kinds:

| Kind | What it runs | Free? |
| --- | --- | --- |
| **Docker / Gradio** | A server (here, FastAPI) | Needs a Hugging Face PRO subscription |
| **Static** | Only files: HTML, JavaScript, JSON | Free for everyone, and **never sleeps** |

TextPulse uses a static Space. The trick is that the server isn't needed on the live site, because the models can run in the visitor's browser:

1. Both models are Logistic Regression on TF-IDF numbers. A prediction only needs each word's **IDF value**, each word's **weight**, and one **bias**:
   `probability = sigmoid(sum(tfidf value × weight) + bias)`.
2. `scripts/export_for_browser.py` saves those numbers as JSON in `deploy/browser/data/`, together with the unseen test messages for the stream. Before saving, it checks that its copy of the maths agrees with scikit-learn. It does: same label on 100% of 6,000 test messages, with probabilities within 0.0001.
3. `app/static/engine.js` repeats the same steps in JavaScript (clean the text, split into words and word pairs, TF-IDF, sigmoid) and plays the role of the server: the live stream, pause, speed, and *Analyse your own text*.
4. `app/static/app.js` asks the same questions either way. When `engine.js` is loaded, the answers come from the browser; otherwise they come from the FastAPI server.

**One difference:** on the live site every visitor gets their own stream, so pausing affects only your screen. On the FastAPI server, everyone shares one stream.

## How it deploys

| Workflow | When it runs | What it does |
| --- | --- | --- |
| `deploy-textpulse.yml` | Every push that changes `textpulse/app/static/` or `textpulse/deploy/browser/`, or when you click **Run workflow** | Builds the site folder and uploads it to the Space. It's live within a minute. |
| `keepalive-textpulse.yml` ("Check TextPulse is live") | Once a day | Opens the site and its model files. If any don't answer, the run fails and GitHub emails you. |

This is **CI/CD** (continuous integration and deployment): GitHub is the single source of truth, and the live site updates itself whenever the code changes.

What the deploy uploads:

| From | Goes to the Space as |
| --- | --- |
| `app/static/index.html` | `index.html`, with `engine.js` added before `app.js` |
| `app/static/` (CSS, JS, Chart.js) | `static/` |
| `deploy/browser/data/` | `data/`: the two models, the stream messages and the scores |
| `deploy/browser/SPACE_README.md` | `README.md`: the Space settings (`sdk: static`) and description |

## One-time setup (already done for this repository)

1. **Hugging Face token:** [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **Create new token** → type **Write**, name `github-deploy`. Copy it; it's shown only once. Treat it like a password and never put it in code, chat or a commit.
2. **GitHub settings:** this repository → **Settings → Secrets and variables → Actions**
   - **Secrets** → **New repository secret**: name `HF_TOKEN`, value = the token.
   - **Variables** → **New repository variable**: name `HF_SPACE`, value `Shubham2019/textpulse`.
3. **First deploy:** **Actions** → **Deploy TextPulse to Hugging Face** → **Run workflow**. The workflow creates the Space if it doesn't exist.

## After you retrain the models

The live site uses the JSON export, not the `.joblib` files, so export again after training:

```
python -m src.train
python -m scripts.export_for_browser
git add textpulse/deploy/browser && git commit -m "Retrained models" && git push
```

The push redeploys the site.

## The Docker option (Hugging Face PRO)

`deploy/docker/` contains everything needed to run the real FastAPI server as a Docker Space instead (Dockerfile, pinned library versions, the `.joblib` models). Hugging Face only hosts Docker Spaces on a PRO subscription. To use it, change the deploy workflow to assemble the Space from `deploy/docker/`, `app/` and `src/`, and set `space_sdk="docker"`.

## Limits to know

- **Download size:** a visitor's browser downloads about 17 MB once (the two models and the test messages), which takes a few seconds on a normal connection. The status badge shows "Loading models" until it's ready.
- **Stream sample:** the live site replays all 7,133 spam test messages and a random 20,000 of the 320,000 test tweets, to keep the download small. Accuracy on the sample matches the full test set.
- **Scheduled workflows** in a repository with no new commits for 60 days are paused by GitHub. That only stops the daily check, not the site.
