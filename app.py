import os
import gc
import uuid
import shutil
import subprocess
import threading
import traceback

import numpy as np
import onnxruntime as ort
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, render_template_string

APP_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(APP_DIR, "jobs")
# .onnx weights ship inside the repo itself — no runtime download needed
WEIGHTS_PATH = os.path.join(APP_DIR, "weights", "paprika.onnx")

os.makedirs(JOBS_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60MB upload cap (free tier RAM is tight)

# Hard server-side ceilings — enforced no matter what the client sends.
# 480px is close to this model's own 512px training resolution — going
# higher doesn't improve quality and risks running out of the 512MB
# free-tier RAM budget (verified: 480px frame ≈ 466MB peak with ONNX
# Runtime; 720px would need well over 1GB, not achievable free).
HARD_MAX_SIDE = 480
HARD_MAX_FPS = 10

# in-memory job tracker: {job_id: {"status": ..., "progress": ..., "error": ...}}
JOBS = {}

# Only one job may run frame conversion at a time — free tier RAM can't
# safely handle two jobs' models/frames in memory simultaneously.
# Extra uploads wait their turn instead of running concurrently.
_processing_lock = threading.Lock()

_session = None
_session_lock = threading.Lock()


def get_session():
    """Lazy-load the ONNX session once, on first use (keeps app boot fast)."""
    global _session
    with _session_lock:
        if _session is None:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1  # keep memory/CPU overhead minimal on free tier
            _session = ort.InferenceSession(WEIGHTS_PATH, sess_options=so,
                                             providers=["CPUExecutionProvider"])
        return _session


def to_input(img: Image.Image):
    x = np.asarray(img).astype("float32") / 127.5 - 1.0
    x = x.transpose(2, 0, 1)[None, ...]  # HWC -> NCHW
    return x


def to_image(out: np.ndarray):
    x = out[0].transpose(1, 2, 0)
    x = (x + 1.0) * 127.5
    x = x.clip(0, 255).astype("uint8")
    return Image.fromarray(x)


def run(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stdout.decode(errors='ignore')}")


def process_job(job_id, in_path, fps, max_side):
    job = JOBS[job_id]
    work = os.path.join(JOBS_DIR, job_id)
    frames_in = os.path.join(work, "frames_in")
    frames_out = os.path.join(work, "frames_out")
    os.makedirs(frames_in, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)

    # never trust client-sent values alone — free tier has only 512MB total
    fps = min(fps, HARD_MAX_FPS)
    max_side = min(max_side, HARD_MAX_SIDE) if max_side else HARD_MAX_SIDE

    job["status"] = "queued"
    with _processing_lock:  # wait here if another job is currently processing
        try:
            job["status"] = "extracting_frames"
            run(["ffmpeg", "-y", "-threads", "1", "-i", in_path, "-vf", f"fps={fps}",
                 os.path.join(frames_in, "f_%06d.png")])

            frame_files = sorted(os.listdir(frames_in))
            total = len(frame_files)
            if total == 0:
                raise RuntimeError("No frames extracted — check the video file.")
            if total > 250:
                raise RuntimeError(
                    f"Clip produced {total} frames — too many for free-tier RAM. "
                    f"Use a shorter clip (under ~20 sec at {fps}fps)."
                )

            model = get_session()
            job["status"] = "converting"
            job["total_frames"] = total

            for i, fname in enumerate(frame_files):
                img = Image.open(os.path.join(frames_in, fname)).convert("RGB")
                if max_side:
                    w, h = img.size
                    scale = max_side / max(w, h)
                    if scale < 1:
                        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                    # keep dimensions divisible by 8 (model requirement)
                    w2, h2 = img.size
                    w2 -= w2 % 8
                    h2 -= h2 % 8
                    img = img.resize((max(w2, 8), max(h2, 8)), Image.LANCZOS)

                inp = to_input(img)
                out = model.run(None, {"input": inp})[0]
                out_img = to_image(out)
                out_img.save(os.path.join(frames_out, fname))
                job["progress"] = i + 1

                # explicit cleanup — arrays/images can otherwise accumulate
                # across hundreds of frames on a 512MB instance
                del img, inp, out, out_img
                if (i + 1) % 20 == 0:
                    gc.collect()

            # release any lingering frame/tensor memory before spawning
            # ffmpeg subprocesses, which add their own memory on top
            gc.collect()

            job["status"] = "reassembling"
            out_video = os.path.join(work, "anime_no_audio.mp4")
            run(["ffmpeg", "-y", "-threads", "1", "-framerate", str(fps), "-i",
                 os.path.join(frames_out, "f_%06d.png"),
                 "-c:v", "libx264", "-preset", "ultrafast", "-bf", "0",
                 "-pix_fmt", "yuv420p", out_video])

            # Upscale to 720p — this is a plain resize on the already-generated
            # short video, not extra AI work, so it's fast (seconds, not
            # per-frame). It does NOT add real extra detail beyond what the
            # source resolution had; it just outputs a proper 720p file.
            upscaled_video = os.path.join(work, "anime_720p.mp4")
            run(["ffmpeg", "-y", "-threads", "1", "-i", out_video, "-vf", "scale=-2:720:flags=bilinear",
                 "-c:v", "libx264", "-preset", "ultrafast", "-bf", "0",
                 "-pix_fmt", "yuv420p", upscaled_video])

            final_video = os.path.join(work, "anime_final.mp4")
            # try to mux original audio back in; fall back to silent video if source has no audio
            try:
                run(["ffmpeg", "-y", "-threads", "1", "-i", upscaled_video, "-i", in_path,
                     "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0?",
                     "-shortest", final_video])
            except Exception:
                shutil.copy(upscaled_video, final_video)

            job["status"] = "done"
            job["result"] = os.path.basename(final_video)
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            traceback.print_exc()
        finally:
            shutil.rmtree(frames_in, ignore_errors=True)
            shutil.rmtree(frames_out, ignore_errors=True)


INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Anime Video Converter</title>
  <style>
    :root {
      --bg: #0b0f0e;
      --panel: #121816;
      --accent: #39ff8f;
      --accent-dim: #1f8f55;
      --text: #e6f2ec;
      --text-dim: #8fa89c;
      --border: #22302a;
    }
    * { box-sizing: border-box; }
    body {
      font-family: 'Share Tech Mono', 'Courier New', monospace;
      background: var(--bg);
      color: var(--text);
      max-width: 480px;
      margin: 40px auto;
      padding: 0 16px;
    }
    h1 { font-size: 1.3rem; color: var(--accent); letter-spacing: 0.5px; }
    p { color: var(--text-dim); }
    label { color: var(--text-dim); font-size: 0.85rem; display: block; margin-top: 12px; }
    input, select, button {
      width: 100%;
      padding: 10px;
      margin: 6px 0;
      font-size: 1rem;
      font-family: inherit;
      background: var(--panel);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
    }
    input[type="file"] { padding: 8px; }
    button {
      background: var(--accent);
      color: #06120c;
      font-weight: bold;
      border: none;
      cursor: pointer;
      margin-top: 16px;
    }
    button:hover { background: var(--accent-dim); }
    #status {
      margin-top: 16px;
      padding: 12px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 0.9rem;
      white-space: pre-wrap;
      color: var(--accent);
    }
    a.download {
      display: block;
      margin-top: 12px;
      text-align: center;
      background: var(--accent);
      color: #06120c;
      font-weight: bold;
      padding: 10px;
      border-radius: 6px;
      text-decoration: none;
    }
  </style>
</head>
<body>
  <h1>▲ VIDEO → ANIME CONVERTER</h1>
  <p>Upload a short clip (10–20s recommended per run on free CPU hosting).</p>
  <form id="f">
    <input type="file" name="video" accept="video/*" required>
    <label>Output frame rate</label>
    <select name="fps">
      <option value="6">6 fps (fastest, safest for free tier)</option>
      <option value="8" selected>8 fps</option>
      <option value="10">10 fps (max — higher will be capped)</option>
    </select>
    <label>Max resolution (longest side)</label>
    <select name="max_side">
      <option value="240">240px (fastest, safest for free tier)</option>
      <option value="360">360px</option>
      <option value="480" selected>480px (max — matches model's native training resolution)</option>
    </select>
    <p style="font-size:0.8rem;color:var(--text-dim);">Free-tier RAM is limited to 512MB — clips over ~20 sec or higher settings than these may fail. Keep clips short.</p>
    <button type="submit">Convert to Anime</button>
  </form>
  <div id="status"></div>

<script>
const form = document.getElementById('f');
const statusEl = document.getElementById('status');
const STORAGE_KEY = 'anime_converter_job_id';

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  statusEl.textContent = 'Uploading...';
  const fd = new FormData(form);
  const res = await fetch('/upload', { method: 'POST', body: fd });
  const data = await res.json();
  if (!res.ok) { statusEl.textContent = 'Error: ' + (data.error || 'upload failed'); return; }
  const jobId = data.job_id;
  localStorage.setItem(STORAGE_KEY, jobId);
  poll(jobId);
});

async function poll(jobId) {
  let res, data;
  try {
    res = await fetch('/status/' + jobId);
    data = await res.json();
  } catch (err) {
    // network hiccup — don't give up, just retry
    statusEl.textContent = 'Connection lost, retrying...';
    setTimeout(() => poll(jobId), 3000);
    return;
  }
  if (!res.ok) {
    // job unknown to this server (e.g. it redeployed/restarted) — clear stale ID
    localStorage.removeItem(STORAGE_KEY);
    statusEl.textContent = 'This job is no longer available (server may have restarted). Please start a new conversion.';
    return;
  }
  let msg = 'Status: ' + data.status;
  if (data.status === 'queued') {
    msg = 'Waiting — another conversion is currently running. Yours will start automatically.';
  }
  if (data.total_frames) {
    msg += '\\nFrames: ' + (data.progress || 0) + ' / ' + data.total_frames;
  }
  statusEl.textContent = msg;
  if (data.status === 'done') {
    localStorage.removeItem(STORAGE_KEY);
    statusEl.innerHTML = msg + '<br><a class="download" href="/download/' + jobId + '">Download anime video</a>';
    return;
  }
  if (data.status === 'error') {
    localStorage.removeItem(STORAGE_KEY);
    statusEl.textContent = msg + '\\n' + (data.error || '');
    return;
  }
  setTimeout(() => poll(jobId), 3000);
}

// on page load, reconnect to an in-progress job if one exists (e.g. after a refresh)
window.addEventListener('DOMContentLoaded', () => {
  const savedJobId = localStorage.getItem(STORAGE_KEY);
  if (savedJobId) {
    statusEl.textContent = 'Reconnecting to previous job...';
    poll(savedJobId);
  }
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400
    f = request.files["video"]
    fps = int(request.form.get("fps", 12))
    max_side = int(request.form.get("max_side", 480))

    job_id = uuid.uuid4().hex[:12]
    work = os.path.join(JOBS_DIR, job_id)
    os.makedirs(work, exist_ok=True)
    in_path = os.path.join(work, "input" + os.path.splitext(f.filename)[1])
    f.save(in_path)

    JOBS[job_id] = {"status": "queued", "progress": 0}
    t = threading.Thread(target=process_job, args=(job_id, in_path, fps, max_side), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(job)


@app.route("/download/<job_id>")
def download(job_id):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "not ready"}), 400
    work = os.path.join(JOBS_DIR, job_id)
    return send_from_directory(work, job["result"], as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
