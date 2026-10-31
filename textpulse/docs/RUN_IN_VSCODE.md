# Run TextPulse in VS Code (Windows)

These steps take about 10 minutes the first time. Afterwards, starting the dashboard takes one command.

You need **VS Code** with the **Python** extension (by Microsoft), and Python 3.10 or newer. Anaconda's Python works.

## 1. Open the project

1. Open VS Code.
2. **File → Open Folder…** and choose `D:\AI ML\Claude\textpulse` (the folder that contains `README.md`, `app` and `src`).
3. If VS Code asks *"Do you trust the authors of the files in this folder?"*, click **Yes, I trust the authors**.

## 2. Create a Python environment and install the libraries

An environment keeps this project's libraries separate, so they can't break your other Anaconda projects.

1. Press **Ctrl+Shift+P**, type **Python: Create Environment**, and press Enter.
2. Choose **Venv**.
3. Choose your Python interpreter, for example the Anaconda one (`anaconda3\python.exe`), version 3.10 or newer.
4. When it asks about dependencies, **tick `requirements.txt`** and click **OK**.

VS Code creates a `.venv` folder and installs everything. This takes 2 to 4 minutes; watch the progress in the bottom-right corner.

## 3. Open a terminal

**Terminal → New Terminal** (or Ctrl+`). The prompt should start with `(.venv)`. If it doesn't, close the terminal and open a new one.

Check that everything is installed:

```
python -m pytest
```

You should see `7 passed`.

## 4. Train the models (once, about 3 to 5 minutes)

```
python -m src.train
```

You'll see each step print: the 1.6 million tweets load, are converted to numbers, and the two algorithms are compared. When it finishes, the `models` folder contains `spam_model.joblib` and `sentiment_model.joblib`.

Optional: score every record and measure the speed (about 1 minute). The result appears in the dashboard's model card.

```
python -m scripts.benchmark
```

## 5. Start the dashboard

```
python -m uvicorn app.main:app
```

When you see `Uvicorn running on http://127.0.0.1:8000`, **Ctrl+click** that link, or open http://127.0.0.1:8000 in your browser.

- The API documentation is at http://127.0.0.1:8000/docs
- To stop the server, click in the terminal and press **Ctrl+C**.

**Next time:** open the folder in VS Code, open a terminal, and run only step 5.

## If something goes wrong

| Problem | Fix |
| --- | --- |
| `running scripts is disabled on this system` when the terminal opens | Click the **⌄** next to **+** in the terminal panel, choose **Command Prompt**, and run the commands there. |
| `No module named ...` | The terminal isn't using `.venv`. Press Ctrl+Shift+P → **Python: Select Interpreter** → choose the one marked `.venv`, then open a new terminal. |
| `Missing dataset: ...` | A file is missing from the `data` folder. See `data/README.md`. |
| `address already in use` / port 8000 busy | Run `python -m uvicorn app.main:app --port 8001` and open http://127.0.0.1:8001 |
| `MemoryError` during training | Close other programs, or lower `VOCAB_SAMPLE_SIZE` in `src/config.py` to `150_000`. |
| Dashboard says *model not trained yet* | Run step 4, then restart the server (Ctrl+C, then step 5 again). |

## Presenting tips

- Start the server a minute before you present, so the charts have history.
- Use the **Speed** menu: 20 msg/s is readable, while 200 msg/s shows off throughput.
- Click **Pause stream** to talk through one message and its highlighted words.
- Use the sample buttons (**Phishing**, **Angry tweet**…) or type a message the panel suggests.
- Safe display mode is on by default (`SAFE_DISPLAY` in `src/config.py`), so explicit spam and real @usernames are hidden on screen.
