# Video → Anime Converter

Self-hosted, free, no credits, no watermark. Converts a video clip into
anime style using AnimeGANv2 (face-portrait weights), running on
Render's free CPU tier via ONNX Runtime — a much lighter engine than
full PyTorch, needed to fit inside the 512MB free-tier RAM budget.

## File structure

```
anime-converter/
├── app.py                          Flask app: upload, job queue, status, download
├── weights/
│   └── face_paint_512_v2.onnx      Model weights (~8.7MB) — must be committed as-is
├── requirements.txt                 Python dependencies
├── Dockerfile                       Build instructions (installs ffmpeg + deps)
├── .gitignore
└── README.md                        This file
```

Unlike earlier versions of this project, the weights file now ships
**inside the repo** rather than being downloaded at runtime — GitHub's
uploader handles the ~8.7MB file fine, just make sure it lands inside
a `weights/` folder, not the repo root.

---

## Step 1: Upload to GitHub (web UI, no terminal needed)

1. Go to [github.com/new](https://github.com/new) (or reuse your
   existing `anime-converter` repo — just replace the files below)
2. If starting fresh: repo name `anime-converter`, **Public** or
   **Private** either works, don't check "Add a README" — **Create repository**
3. Click **"uploading an existing file"**
4. Upload `app.py`, `requirements.txt`, `Dockerfile`, `.gitignore`, `README.md`
   directly into the repo root
5. For the weights file: on the repo page, click **Add file → Create new file**,
   type `weights/face_paint_512_v2.onnx` as the filename (the `weights/`
   prefix automatically creates the folder), then look for the
   **"upload files"** link that appears instead of typing content —
   use that to upload the actual `.onnx` file into that folder.
6. Commit changes

---

## Step 2: Deploy on Render

1. Go to [render.com](https://render.com) → log in
2. **New +** → **Web Service** → select your `anime-converter` repo
3. Render should auto-detect the **Dockerfile** — choose **Docker** if asked
4. Settings: any name, **Free** instance type, leave build/start
   commands blank
5. **Create Web Service**

Build is faster now than before (no PyTorch to install) — a couple
minutes typically. Watch **Logs** for `Listening at: http://0.0.0.0:...`

---

## Step 3: Use it

1. Open the `.onrender.com` URL in your phone browser
2. Upload a short clip (**start with 5–10 sec** to confirm it works)
3. Pick fps (8 recommended) and resolution (480px is the max — this
   matches the model's own training resolution, so it's also the
   *best* quality setting, not just the biggest allowed one)
4. Convert, watch progress, download when done

### For your full 1–2 min video
Split it into ~15-20 sec chunks first (CapCut/VN), run each chunk
through the tool separately, then stitch the anime chunks back
together in CapCut/VN in order.

---

## Known limits (free tier)

- **Sleeps after 15 min idle** — first request after sleeping takes
  30-60 sec to wake up. Normal, not a bug.
- **512MB RAM total.** Measured peak memory with ONNX Runtime:
  ~279MB at 360px, ~466MB at 480px (leaves headroom for
  Flask/ffmpeg). 480px is the hard ceiling for a reason — going
  higher (e.g. 720px) would need well over 1GB, which isn't
  achievable on the free tier no matter how the code is tuned.
- **Frame cap**: max ~250 frames per job (~20-25 sec clip at 10fps),
  enforced server-side regardless of what you select.
- **Speed**: CPU only. Expect roughly 1-2 seconds per frame — faster
  than the old PyTorch version.
- If a job errors out, the status panel shows the error message.
- If you refresh the page mid-job, it automatically reconnects to
  the in-progress job (saved in your browser's local storage) rather
  than losing track of it.
