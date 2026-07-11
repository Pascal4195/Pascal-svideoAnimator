# Video → Anime Converter

Self-hosted, free, no credits, no watermark. Converts a video clip into
anime style using AnimeGANv2, running entirely on Render's free CPU tier.

## File structure

```
anime-converter/
├── app.py            Flask app: upload, job queue, status, download
├── model.py           AnimeGANv2 generator architecture (PyTorch)
├── requirements.txt    Python dependencies
├── Dockerfile          Build instructions (installs ffmpeg + deps)
├── .gitignore
└── README.md           This file
```

You do **not** need to add the model weights file yourself — the app
downloads it automatically (~8MB) the first time it runs.

---

## Step 1: Upload to GitHub (web UI, no terminal needed)

1. Go to [github.com/new](https://github.com/new)
2. Repo name: `anime-converter` (or whatever you like)
3. Set to **Public** or **Private** — either works on Render
4. Do NOT check "Add a README" (we already have one) — click **Create repository**
5. On the empty repo page, click **"uploading an existing file"**
6. Upload all 6 files from this project: `app.py`, `model.py`,
   `requirements.txt`, `Dockerfile`, `.gitignore`, `README.md`
7. Scroll down, click **Commit changes**

---

## Step 2: Deploy on Render

1. Go to [render.com](https://render.com) → log in
2. Click **New +** → **Web Service**
3. Connect your GitHub account if not already connected, select your
   `anime-converter` repo
4. Render should auto-detect the **Dockerfile** — if it asks for a
   runtime/environment, choose **Docker**
5. Settings:
   - **Name**: anything, e.g. `anime-converter`
   - **Instance type**: **Free**
   - Leave build/start commands blank — the Dockerfile handles both
6. Click **Create Web Service**

First deploy will take a few minutes (installing PyTorch is the slow
part). Watch the **Logs** tab — when you see something like
`Listening at: http://0.0.0.0:10000`, it's live.

---

## Step 3: Use it

1. Open the `.onrender.com` URL Render gives you, in your phone browser
2. Upload a short clip (**start with 5–10 sec** to confirm it works)
3. Pick fps (8 or 12 recommended) and resolution (480px recommended —
   faster, and free-tier RAM is limited)
4. Hit Convert, wait — it'll show live frame progress
5. Download the anime version when done

### For your full 1–2 min video
Split it into ~15-20 sec chunks first (CapCut/VN), run each chunk
through the tool separately, then stitch the anime chunks back
together in CapCut/VN in order.

---

## Known limits (free tier)

- **Sleeps after 15 min idle** — first request after sleeping takes
  30-60 sec to wake up. Normal, not a bug.
- **512MB RAM** — stick to 480px resolution and short clips to stay
  safely inside this.
- **Speed** — CPU only, no GPU. Expect roughly 1-3 seconds per frame.
  At 12fps, a 15-sec chunk (~180 frames) could take several minutes.
- If a job errors out, the status panel will show the error message —
  screenshot it if you want help debugging.
