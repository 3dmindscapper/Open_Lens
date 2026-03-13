# Open Lens — Bulk Document Translator

Open Lens is an offline document translation tool that translates text directly on documents while preserving their original layout and readability. It processes **PNG**, **JPG**, and **PDF** files (including multi-page PDFs) by detecting the source language, translating the text, inpainting the document to remove the original text, and overlaying formatted translated text that matches the original document's appearance.

## How It Works

The translation pipeline runs five steps on every page:

1. **OCR** — Tesseract extracts text blocks with bounding boxes at paragraph level.
2. **Language Detection** — `langdetect` identifies the source language automatically.
3. **Translation** — Argos Translate performs fully offline neural machine translation. Models (~100 MB each) are downloaded once and cached locally.
4. **Inpainting** — OpenCV erases the original text regions and reconstructs the background using the TELEA algorithm, smoothing uniform backgrounds for a clean result.
5. **Rendering** — Translated text is drawn back into each bounding box with auto-fitted font size, word wrapping, sampled ink colour from the original, and RTL support for Arabic/Hebrew.

## Supported Formats

| Format | Input | Output |
|--------|-------|--------|
| PDF (multi-page) | Yes | Yes (single multi-page PDF) |
| PNG | Yes | Yes |
| JPG / JPEG | Yes | Yes |
| TIFF | Yes | Yes |
| BMP | Yes | Yes |
| WebP | Yes | Yes |

Multi-page PDFs are converted page-by-page at 200 DPI, processed individually, and saved back as a single multi-page PDF.

## Requirements

- **Python 3.10+**
- **Tesseract OCR** — [Windows installer](https://github.com/UB-Mannheim/tesseract/wiki) · `brew install tesseract` (macOS) · `sudo apt install tesseract-ocr` (Linux)
- **Poppler** — Included in this repository under `poppler-25.12.0/`. Required for PDF support.

Python dependencies (installed via pip):

```
pip install -r Open_Lens_Core/requirements_translator.txt
```

Key packages: `Pillow`, `numpy`, `opencv-python`, `pytesseract`, `langdetect`, `argostranslate`, `pdf2image`.

Optional for RTL languages: `arabic-reshaper`, `python-bidi`.

## Quick Start

### GUI

Double-click **Launch Translator.bat** or run:

```bash
cd Open_Lens_Core
python translator_ui.py
```

The GUI defaults to translating to **English** with automatic source language detection. Select a different target language from the dropdown if needed.

### Command Line

```bash
cd Open_Lens_Core

# Translate a PDF to English (auto-detect source language)
python translate.py document.pdf -t en

# Translate an image to French, specifying German as source
python translate.py scan.jpg -t fr -s de -o output.jpg

# Specify custom Tesseract path
python translate.py invoice.png -t es --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### CLI Options

| Flag | Description |
|------|-------------|
| `-t`, `--target` | Target language code (e.g. `en`, `fr`, `de`, `es`) — **required** |
| `-s`, `--source` | Source language code (default: `auto` — auto-detect) |
| `-o`, `--output` | Output file path (auto-generated if omitted) |
| `--tesseract` | Path to Tesseract binary |
| `--font` | Path to a custom `.ttf` font file |
| `-q`, `--quiet` | Suppress progress output |

## Project Structure

```
Open_Lens/
├── Launch Translator.bat          # Windows launcher for the GUI
├── README.md                      # This file
├── Open_Lens_Core/
│   ├── translate.py               # CLI entry point
│   ├── translator_ui.py           # Tkinter GUI
│   ├── requirements_translator.txt
│   └── translator_tool/           # Core library
│       ├── __init__.py
│       ├── main.py                # Argument parser & CLI logic
│       ├── pipeline.py            # Orchestrates the full pipeline
│       ├── ocr_extractor.py       # Tesseract OCR wrapper
│       ├── language_utils.py      # Language detection & Argos translation
│       ├── inpainter.py           # OpenCV text removal
│       ├── renderer.py            # Translated text rendering
│       └── file_handler.py        # File I/O (PDF, images)
└── poppler-25.12.0/               # Bundled Poppler binaries for PDF support
```

## Supported Languages

The tool supports 35+ languages including English, French, German, Spanish, Italian, Portuguese, Dutch, Polish, Russian, Ukrainian, Turkish, Romanian, Czech, Hungarian, Swedish, Norwegian, Danish, Finnish, Greek, Bulgarian, Croatian, Slovak, Lithuanian, Latvian, Estonian, Japanese, Korean, Chinese (Simplified & Traditional), Arabic, Hebrew, Hindi, Thai, Vietnamese, Indonesian, and Malay.

Translation between any pair is handled directly if a model exists, or via English as a pivot language.



## Web Server / Frontend Integration

The translation pipeline exposes a single callable that is easy to wrap in an HTTP server, making it straightforward to connect to a frontend or an agent workflow that uploads documents for translation.

### FastAPI example

Install the server dependency: `pip install fastapi uvicorn python-multipart`

Create `server.py` alongside `translator_ui.py` inside your folder:

```python
import os, uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
import sys
sys.path.insert(0, str(Path(__file__).parent))

from translator_tool.pipeline import process_document

app = FastAPI()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/translate")
async def translate(file: UploadFile, target_lang: str = Form(default="en")):
    suffix = Path(file.filename).suffix
    job_id = uuid.uuid4().hex
    in_path  = UPLOAD_DIR / f"{job_id}_in{suffix}"
    out_path = UPLOAD_DIR / f"{job_id}_out{suffix}"

    in_path.write_bytes(await file.read())

    process_document(
        input_path=str(in_path),
        target_lang=target_lang,
        output_path=str(out_path),
    )

    return FileResponse(str(out_path), filename=f"translated_{file.filename}")
```

Start the server:

```bash
cd <your-folder>
uvicorn server:app --host 0.0.0.0 --port 8000
```

Your frontend or agent POSTs a `multipart/form-data` upload to `http://localhost:8000/translate` and receives the translated file directly in the response.

### Production considerations

| Topic | Notes |
|---|---|
| **Long processing times** | A multi-page PDF can take 30–120 seconds. Use a task queue (Celery + Redis) so the endpoint returns a job ID immediately and the client polls for the result instead of holding a long connection open. |
| **Concurrency** | The pipeline is CPU-heavy. Run jobs in a thread pool (`asyncio.run_in_executor`) and set `--workers N` in uvicorn to match your CPU count. |
| **File cleanup** | Delete temporary upload/output files after download, or sweep them on a schedule. |
| **First-run model download** | Argos Translate downloads ~100 MB language models on first use per language pair. Pre-warm them at server startup or bake them into your deployment environment so the first request does not time out. |
| **Tesseract & Poppler** | Both must be available in the server environment. Poppler is auto-detected from the bundled `poppler-*/Library/bin` folder relative to the project — no PATH configuration needed. |

## License

See [Open_Lens_Core/LICENSE](Open_Lens_Core/LICENSE).
