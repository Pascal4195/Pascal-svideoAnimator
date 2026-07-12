# Video → Anime Converter

Self-hosted, free, no credits, no watermark. Converts a video clip into
anime style using AnimeGANv2 (paprika weights — full-scene style),
running on Render's free CPU tier via ONNX Runtime, then upscaled to
720p as a final step.

## File structure

```
anime-converter/
├── app.py                Flask app: upload, job queue, status, download
├── weights/
│   └── paprika.onnx      Model weights (~8.7MB) — must be committed as-is
├── requirements.txt       Python dependencies
├── Dockerfile             Build instructions (installs ffmpeg + deps)
├── .gitignore
└── README.md              This file
```

The weights file ships **inside the repo** — GitHub's uploader handles
the ~8.7MB file fine, just make sure it lands inside a `weights/`
folder, not the repo root.

---

## Step 1: Upload to GitHub (web UI, no terminal needed)

1. Go to your existing `anime-converter` repo (or create one at
   [github.com/new](https://github.com/new) if starting fresh)
2. Upload `app.py`, `requirements.txt`, `Dockerfile`, `.gitignore`,
   `README.md` into the repo root (replace existing versions)
3. For the weights file: click **Add file → Create new file**, type
   `weights/paprika.onnx` as the filename (the `weights/` prefix
   creates the folder), then use the **"upload files"** link that
   appears instead of typing content — upload the actual `.onnx`
   file there
4. If `model.py` or `weights/face_paint_512_v2.onnx` still exist in
   your repo from an earlier version, you can delete them — nothing
   uses them anymore
5. Commit changes

---

## Step 2: Deploy on Render

Same as before — if already connected to this repo, Render
auto-rebuilds on commit. Otherwise: **New + → Web Service** → select
repo → **Free** instance → leave build/start commands blank (Docker
handles it) → **Create Web Service**.

---

## Step 3: Use it

1. Open the `.onrender.com` URL in your phone browser
2. Upload a short clip (**start with 5–10 sec** to confirm it works)
3. Pick fps (8 recommended) and resolution (480px max — this is the
   AI conversion resolution; the final output is then upscaled to
   720p automatically)
4. Convert, watch progress, download when done — the downloaded file
   will be a real 720p video

**Note on the 720p upscale**: it's a resize of the finished video, not
extra AI detail. The actual content sharpness reflects the 480px
conversion resolution; the upscale just outputs a proper 720p file
for compatibility/sizing purposes, and adds only a few seconds to
total processing time.

### For your full 1–2 min video
Split it into ~15-20 sec chunks first (CapCut/VN), run each chunk
through the tool separately, then stitch the anime chunks back
together in CapCut/VN in order.

---

## Known limits (free tier)

- **Sleeps after 15 min idle** — first request after sleeping takes
  30-60 sec to wake up. Normal, not a bug.
- **512MB RAM total.** The AI conversion step is capped at 480px for
  this reason (measured peak ~466MB with ONNX Runtime at that
  resolution). The 720p upscale happens after conversion and doesn't
  add meaningful memory load.
- **Frame cap**: max ~250 frames per job (~20-25 sec clip at 10fps),
  enforced server-side regardless of what you select.
- If a job errors out, the status panel shows the error message.
- If you refresh the page mid-job, it automatically reconnects to
  the in-progress job (saved in your browser's local storage).
