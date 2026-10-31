# Deploying TextPulse live (Hugging Face Spaces + GitHub Actions)

TextPulse runs live on a free [Hugging Face Space](https://huggingface.co/spaces). Two GitHub Actions in this repository manage it:

| Workflow | When it runs | What it does |
| --- | --- | --- |
| `deploy-textpulse.yml` | Every push that changes `textpulse/`, or when you click **Run workflow** | Builds the Space folder and uploads it to Hugging Face, which rebuilds the app (3–5 minutes) |
| `keepalive-textpulse.yml` | Every 6 hours | Visits the app so the free Space never reaches its 48-hour sleep limit, and restarts it if it has stopped or crashed |

This is **CI/CD** (continuous integration and deployment): GitHub is the single source of truth, and the live app updates itself whenever the code changes.

## One-time setup (about 10 minutes)

### 1. Create a Hugging Face account
Sign up at [huggingface.co/join](https://huggingface.co/join) and verify your email. You don't need to create the Space; the deploy workflow creates it.

### 2. Create a Hugging Face access token
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **Create new token**.
2. Choose **Write** as the token type, and name it `github-deploy`.
3. Click **Create token** and **copy it**. You'll only see it once.

Treat the token like a password. Never paste it into code, a chat or a commit. It only goes into GitHub's encrypted secrets in the next step.

### 3. Add the token and Space name to GitHub
In this repository on GitHub, open **Settings → Secrets and variables → Actions**.

1. **Secrets** tab → **New repository secret**
   - Name: `HF_TOKEN`
   - Secret: paste the token → **Add secret**
2. **Variables** tab → **New repository variable**
   - Name: `HF_SPACE`
   - Value: `your-huggingface-username/textpulse` → **Add variable**

### 4. Run the first deployment
**Actions** tab → **Deploy TextPulse to Hugging Face** → **Run workflow** → **Run workflow**.

When it finishes (about 1 minute), its log prints the Space page and the live link. Hugging Face then builds the app for 3–5 minutes. Your live link is:

```
https://<your-huggingface-username>-textpulse.hf.space
```

### 5. Start the keep-alive
**Actions** tab → **Keep TextPulse awake** → **Run workflow** once. After that it runs automatically every 6 hours.

## Everyday use

- **Change the app:** push to `main`. The Space redeploys itself.
- **Check the app:** open the **Logs** tab on the Space page.
- **Pause the app on purpose:** use **Settings → Pause** on the Space. The keep-alive won't restart a paused Space.

## What gets deployed

The deploy workflow combines the current code with pinned settings from `textpulse/deploy/`:

| From | Goes to the Space as | Why |
| --- | --- | --- |
| `textpulse/app/`, `textpulse/src/` | `app/`, `src/` | The application code |
| `textpulse/deploy/Dockerfile` | `Dockerfile` | How Hugging Face builds and starts the app (port 7860) |
| `textpulse/deploy/requirements.txt` | `requirements.txt` | **Exact** library versions the models were trained with |
| `textpulse/deploy/SPACE_README.md` | `README.md` | The Space's settings (`sdk: docker`, `app_port: 7860`) and description |
| `textpulse/deploy/models/` | `models/` | The trained models and the test messages for the live stream |

The trained models in `textpulse/deploy/models/` were created with scikit-learn 1.9.1 on Python 3.13, which is why `deploy/requirements.txt` pins those exact versions. A model saved by one scikit-learn version may not load in another. If you retrain, copy the new files from `textpulse/models/` into `textpulse/deploy/models/` **and** update the versions in `deploy/requirements.txt` to the ones you trained with (`pip freeze` shows them).

## Limits to know

- **Free hardware:** 2 CPUs and 16 GB of RAM, which is plenty for TextPulse.
- **One shared stream:** everyone who opens the link sees the same stream. Pausing it pauses it for everyone.
- **Scheduled workflows** in a repository with no new commits for 60 days are paused by GitHub. GitHub emails you first; re-enable it in the **Actions** tab, or push any change.
- **Hugging Face maintenance** can restart the Space. It comes back on its own, and the keep-alive restarts it if needed.
