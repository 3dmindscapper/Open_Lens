"""
Open Lens — Web Server
======================
Hosts the document translator as a local web service.
Any device on the same network can upload a document and download the
translated version — no desktop app needed.

Run:
    python web_app.py

Then open the printed URL on any device (phone, tablet, other PC) connected
to the same Wi-Fi / LAN.
"""

from __future__ import annotations

import glob as glob_module
import io
import os
import re
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is importable (same pattern as translator_ui.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, abort, jsonify, render_template_string, request, send_file
from werkzeug.utils import secure_filename

from translator_tool.pipeline import process_document
from translator_tool.file_handler import SUPPORTED_IMAGE_EXTS

# ---------------------------------------------------------------------------
# Storage folders — use a sub-directory of the system temp folder
# ---------------------------------------------------------------------------
_BASE_TMP = Path(os.environ.get("TEMP", os.environ.get("TMP", "/tmp")))
UPLOAD_FOLDER = _BASE_TMP / "openlens_uploads"
OUTPUT_FOLDER = _BASE_TMP / "openlens_outputs"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 100 * 1024 * 1024          # 100 MB
ALLOWED_EXTENSIONS = {".pdf"} | SUPPORTED_IMAGE_EXTS
FILE_TTL_SECONDS = 600                         # clean up files after 10 min

_LANG_CODE_RE = re.compile(r"^[a-z]{2,5}(-[A-Z]{2})?$|^auto$")

TARGET_LANGUAGES = [
    ("English",               "en"),
    ("French",                "fr"),
    ("German",                "de"),
    ("Spanish",               "es"),
    ("Italian",               "it"),
    ("Portuguese",            "pt"),
    ("Dutch",                 "nl"),
    ("Polish",                "pl"),
    ("Russian",               "ru"),
    ("Ukrainian",             "uk"),
    ("Turkish",               "tr"),
    ("Romanian",              "ro"),
    ("Czech",                 "cs"),
    ("Hungarian",             "hu"),
    ("Swedish",               "sv"),
    ("Norwegian",             "no"),
    ("Danish",                "da"),
    ("Finnish",               "fi"),
    ("Greek",                 "el"),
    ("Bulgarian",             "bg"),
    ("Croatian",              "hr"),
    ("Slovak",                "sk"),
    ("Lithuanian",            "lt"),
    ("Latvian",               "lv"),
    ("Estonian",              "et"),
    ("Japanese",              "ja"),
    ("Korean",                "ko"),
    ("Chinese (Simplified)",  "zh"),
    ("Chinese (Traditional)", "zt"),
    ("Arabic",                "ar"),
    ("Hebrew",                "he"),
    ("Hindi",                 "hi"),
    ("Thai",                  "th"),
    ("Vietnamese",            "vi"),
    ("Indonesian",            "id"),
    ("Malay",                 "ms"),
]

# ---------------------------------------------------------------------------
# Job registry  {job_id: dict}
# ---------------------------------------------------------------------------
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _job_output_files(output_path: str) -> list[str]:
    """Return the actual file(s) written by save_output for a given output_path."""
    base = Path(output_path)
    if base.exists():
        return [str(base)]
    # Multi-page image case: _page1, _page2, …
    pattern = str(base.parent / f"{base.stem}_page*{base.suffix}")
    found = sorted(glob_module.glob(pattern))
    return found


def _schedule_cleanup(job_id: str, delay: int = FILE_TTL_SECONDS) -> None:
    """Delete input/output files and the job record after *delay* seconds."""
    def _worker():
        time.sleep(delay)
        with jobs_lock:
            job = jobs.pop(job_id, None)
        if not job:
            return
        for key in ("input_path", "output_path", "zip_path"):
            p = job.get(key)
            if p:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
        # Also remove numbered page files
        if job.get("output_path"):
            for p in _job_output_files(job["output_path"]):
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass

    threading.Thread(target=_worker, daemon=True).start()


def _run_translation(job_id: str, input_path: str,
                     target_lang: str, source_lang: str) -> None:
    """Background worker — runs the full translation pipeline."""
    with jobs_lock:
        jobs[job_id]["status"] = "running"

    log_lines: list[str] = []

    def _log(msg: str) -> None:
        log_lines.append(msg)
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["log"] = log_lines[:]

    try:
        output_path = process_document(
            input_path=input_path,
            target_lang=target_lang,
            source_lang=source_lang,
            verbose=False,
            log_callback=_log,
        )

        # Resolve the actual written file(s)
        written = _job_output_files(output_path)

        if len(written) == 0:
            raise RuntimeError("Translation finished but no output file was found.")

        # Package multiple files as a ZIP so there is always one download link
        if len(written) > 1:
            zip_path = str(OUTPUT_FOLDER / f"{job_id}_result.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in written:
                    zf.write(f, Path(f).name)
            final_path = zip_path
            download_ext = ".zip"
        else:
            final_path = written[0]
            download_ext = Path(written[0]).suffix

        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["output_path"] = output_path   # original guess path
            jobs[job_id]["final_path"] = final_path
            jobs[job_id]["download_ext"] = download_ext
            jobs[job_id]["log"] = log_lines

    except Exception as exc:
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(exc)
                jobs[job_id]["log"] = log_lines

    finally:
        _schedule_cleanup(job_id)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(_HTML, languages=TARGET_LANGUAGES)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    if not _allowed(file.filename):
        exts = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({"error": f"Unsupported file type. Allowed: {exts}"}), 400

    target_lang = request.form.get("target_lang", "en").strip()
    source_lang = request.form.get("source_lang", "auto").strip()

    if not _LANG_CODE_RE.match(target_lang):
        return jsonify({"error": "Invalid target language code."}), 400
    if not _LANG_CODE_RE.match(source_lang):
        return jsonify({"error": "Invalid source language code."}), 400

    job_id = uuid.uuid4().hex
    safe_name = secure_filename(file.filename)
    suffix = Path(safe_name).suffix.lower()
    input_path = str(UPLOAD_FOLDER / f"{job_id}_input{suffix}")
    file.save(input_path)

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "log": [],
            "input_path": input_path,
            "output_path": None,
            "final_path": None,
            "download_ext": None,
            "original_name": safe_name,
            "error": "",
        }

    t = threading.Thread(
        target=_run_translation,
        args=(job_id, input_path, target_lang, source_lang),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    if not re.match(r"^[0-9a-f]{32}$", job_id):
        abort(400)
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
    return jsonify({
        "status": job["status"],
        "log": job.get("log", []),
        "error": job.get("error", ""),
    })


@app.route("/download/<job_id>")
def download(job_id: str):
    if not re.match(r"^[0-9a-f]{32}$", job_id):
        abort(400)
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        abort(404)
    if job.get("status") != "done":
        abort(400)

    final_path = job.get("final_path")
    if not final_path or not Path(final_path).exists():
        abort(404)

    original_name = job.get("original_name", "document")
    stem = Path(original_name).stem
    ext = job.get("download_ext", Path(final_path).suffix)
    download_name = f"{stem}_translated{ext}"

    return send_file(
        final_path,
        as_attachment=True,
        download_name=download_name,
    )


# ---------------------------------------------------------------------------
# Inline HTML + CSS + JS  (no external dependencies)
# ---------------------------------------------------------------------------
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Open Lens — Document Translator</title>
<style>
:root {
  --bg:       #1e1e2e;
  --panel:    #2a2a3e;
  --accent:   #7c6af7;
  --acchov:   #9d8ffa;
  --text:     #cdd6f4;
  --sub:      #a6adc8;
  --success:  #a6e3a1;
  --error:    #f38ba8;
  --warning:  #fab387;
  --entry:    #313244;
  --border:   #45475a;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:"Segoe UI",system-ui,sans-serif;
     min-height:100vh;display:flex;flex-direction:column;align-items:center}

/* ── header ── */
header{width:100%;background:var(--accent);padding:20px;text-align:center}
header h1{font-size:1.55rem;color:#fff;letter-spacing:.3px}
header p{font-size:.85rem;color:#e0d9ff;margin-top:5px}

/* ── main layout ── */
main{width:100%;max-width:700px;padding:28px 16px;display:flex;
     flex-direction:column;gap:22px}

/* ── card ── */
.card{background:var(--panel);border-radius:12px;padding:22px}
.card h2{font-size:.95rem;color:var(--sub);margin-bottom:14px;font-weight:600;
          text-transform:uppercase;letter-spacing:.6px}

/* ── drop zone ── */
.drop-zone{border:2px dashed var(--border);border-radius:10px;padding:40px 20px;
            text-align:center;cursor:pointer;
            transition:border-color .2s,background .2s}
.drop-zone:hover,.drop-zone.over{border-color:var(--accent);
            background:rgba(124,106,247,.08)}
.drop-zone .icon{font-size:2.8rem;line-height:1;margin-bottom:10px}
.drop-zone .hint{color:var(--sub);font-size:.9rem}
.drop-zone .hint strong{color:var(--accent)}
.drop-zone .types{color:var(--border);font-size:.78rem;margin-top:6px}
#file-name{margin-top:10px;color:var(--success);font-size:.85rem;min-height:1.1em;
            word-break:break-all}
input[type=file]{display:none}

/* ── language row ── */
.lang-row{display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:end}
.lang-sep{color:var(--sub);font-size:1.2rem;text-align:center;padding-bottom:8px}
label{display:block;color:var(--sub);font-size:.83rem;margin-bottom:5px}
select{width:100%;background:var(--entry);color:var(--text);
        border:1px solid var(--border);border-radius:7px;
        padding:9px 10px;font-size:.9rem;outline:none;cursor:pointer}
select:focus{border-color:var(--accent)}

/* ── translate button ── */
#go-btn{width:100%;background:var(--accent);color:#fff;border:none;
         border-radius:9px;padding:14px;font-size:1.05rem;font-weight:700;
         cursor:pointer;transition:background .2s,opacity .2s}
#go-btn:hover:not(:disabled){background:var(--acchov)}
#go-btn:disabled{opacity:.45;cursor:not-allowed}

/* ── progress card ── */
#progress-card{display:none}
.badge{display:inline-block;padding:4px 12px;border-radius:20px;
        font-size:.78rem;font-weight:700;margin-bottom:12px}
.badge-run {background:rgba(250,179,135,.15);color:var(--warning)}
.badge-done{background:rgba(166,227,161,.15);color:var(--success)}
.badge-err {background:rgba(243,139,168,.15);color:var(--error)}

#log-box{background:#11111b;border-radius:8px;padding:13px 15px;
          font-family:"Cascadia Code","Consolas",monospace;font-size:.76rem;
          color:#cdd6f4;max-height:260px;overflow-y:auto;
          white-space:pre-wrap;word-break:break-all;line-height:1.5}

/* ── download button ── */
#dl-btn{display:none;width:100%;background:#40a02b;color:#fff;border:none;
         border-radius:9px;padding:13px;font-size:1rem;font-weight:700;
         cursor:pointer;margin-top:14px;transition:background .2s;
         text-decoration:none;text-align:center}
#dl-btn:hover{background:#54b330}

/* ── spinner ── */
.spin{display:inline-block;width:16px;height:16px;
       border:3px solid rgba(255,255,255,.3);border-top-color:#fff;
       border-radius:50%;animation:rot .75s linear infinite;
       vertical-align:middle;margin-right:8px}
@keyframes rot{to{transform:rotate(360deg)}}

/* ── footer ── */
footer{margin-top:auto;padding:18px;color:var(--border);font-size:.76rem;text-align:center}
</style>
</head>
<body>

<header>
  <h1>&#128270; Open Lens — Document Translator</h1>
  <p>Fully offline &bull; PDF, JPG, PNG &bull; Powered by Tesseract + Argos Translate</p>
</header>

<main>

  <!-- ── File picker ── -->
  <div class="card">
    <h2>1 &mdash; Choose a file</h2>
    <div class="drop-zone" id="drop">
      <div class="icon">&#128196;</div>
      <p class="hint">Drag &amp; drop your file here, or <strong>click to browse</strong></p>
      <p class="types">PDF &bull; JPG &bull; PNG &bull; TIFF &bull; BMP &bull; WebP &mdash; max 100 MB</p>
      <div id="file-name"></div>
    </div>
    <input type="file" id="file-input"
           accept=".pdf,.jpg,.jpeg,.png,.tiff,.tif,.bmp,.webp">
  </div>

  <!-- ── Language selection ── -->
  <div class="card">
    <h2>2 &mdash; Choose languages</h2>
    <div class="lang-row">
      <div>
        <label for="src-lang">Source language</label>
        <select id="src-lang">
          <option value="auto">Auto-detect</option>
          {% for name, code in languages %}
          <option value="{{ code }}">{{ name }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="lang-sep">&#8594;</div>
      <div>
        <label for="tgt-lang">Translate to</label>
        <select id="tgt-lang">
          {% for name, code in languages %}
          <option value="{{ code }}" {% if code == 'en' %}selected{% endif %}>{{ name }}</option>
          {% endfor %}
        </select>
      </div>
    </div>
  </div>

  <!-- ── Action ── -->
  <button id="go-btn" disabled>Translate</button>

  <!-- ── Progress / result ── -->
  <div class="card" id="progress-card">
    <span class="badge badge-run" id="badge">Processing&hellip;</span>
    <div id="log-box">Waiting&hellip;</div>
    <a id="dl-btn" href="#" download>&#11015; Download translated file</a>
  </div>

</main>

<footer>Open Lens &mdash; all processing runs locally on this server &bull; files are deleted after 10 minutes</footer>

<script>
const drop    = document.getElementById('drop');
const fileIn  = document.getElementById('file-input');
const nameEl  = document.getElementById('file-name');
const goBtn   = document.getElementById('go-btn');
const progCard= document.getElementById('progress-card');
const logBox  = document.getElementById('log-box');
const badge   = document.getElementById('badge');
const dlBtn   = document.getElementById('dl-btn');

let selFile = null;
let poll    = null;

/* ── drag & drop ── */
drop.addEventListener('click',    () => fileIn.click());
drop.addEventListener('dragover',  e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('over');
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
fileIn.addEventListener('change', () => { if (fileIn.files[0]) setFile(fileIn.files[0]); });

function setFile(f) {
  selFile = f;
  nameEl.textContent = `\u2713 ${f.name}  (${(f.size/1024/1024).toFixed(2)} MB)`;
  goBtn.disabled = false;
  progCard.style.display = 'none';
  dlBtn.style.display = 'none';
  logBox.textContent = '';
}

/* ── translate ── */
goBtn.addEventListener('click', () => { if (selFile) startJob(); });

async function startJob() {
  goBtn.disabled = true;
  goBtn.innerHTML = '<span class="spin"></span>Uploading\u2026';
  progCard.style.display = 'block';
  dlBtn.style.display = 'none';
  logBox.textContent = 'Uploading file\u2026\n';
  badge.className = 'badge badge-run';
  badge.textContent = 'Uploading\u2026';

  const fd = new FormData();
  fd.append('file',        selFile);
  fd.append('source_lang', document.getElementById('src-lang').value);
  fd.append('target_lang', document.getElementById('tgt-lang').value);

  let jobId;
  try {
    const r = await fetch('/upload', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok || d.error) { showErr(d.error || 'Upload failed'); return; }
    jobId = d.job_id;
  } catch(e) { showErr('Network error: ' + e.message); return; }

  goBtn.innerHTML = '<span class="spin"></span>Translating\u2026';
  badge.textContent = 'Translating\u2026';
  logBox.textContent = 'Translation started\u2026\n';

  let seen = 0;
  poll = setInterval(async () => {
    try {
      const r = await fetch('/status/' + jobId);
      const d = await r.json();
      if (d.log && d.log.length > seen) {
        logBox.textContent += d.log.slice(seen).join('\n') + '\n';
        seen = d.log.length;
        logBox.scrollTop = logBox.scrollHeight;
      }
      if (d.status === 'done') {
        clearInterval(poll);
        badge.className = 'badge badge-done';
        badge.textContent = '\u2713 Done';
        goBtn.innerHTML  = 'Translate';
        goBtn.disabled   = false;
        dlBtn.href       = '/download/' + jobId;
        dlBtn.style.display = 'block';
      } else if (d.status === 'error') {
        clearInterval(poll);
        showErr(d.error || 'Translation failed');
      }
    } catch(_) { /* network hiccup – keep polling */ }
  }, 1500);
}

function showErr(msg) {
  clearInterval(poll);
  badge.className = 'badge badge-err';
  badge.textContent = '\u2717 Error';
  logBox.textContent += '\n\u274C ' + msg;
  goBtn.innerHTML = 'Translate';
  goBtn.disabled  = !selFile;
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import socket

    try:
        # Get the LAN IP (connect to a public address without sending traffic)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = socket.gethostbyname(socket.gethostname())

    port = int(os.environ.get("PORT", 5000))

    print()
    print("  ╔══════════════════════════════════════════╗")
    print("  ║     Open Lens — Web Translator           ║")
    print("  ╠══════════════════════════════════════════╣")
    print(f"  ║  Local:    http://localhost:{port:<15}║")
    print(f"  ║  Network:  http://{lan_ip}:{port:<6}        ║")
    print("  ╠══════════════════════════════════════════╣")
    print("  ║  Share the Network URL with any device   ║")
    print("  ║  on the same Wi-Fi or LAN network.       ║")
    print("  ║  Press Ctrl+C to stop the server.        ║")
    print("  ╚══════════════════════════════════════════╝")
    print()

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
