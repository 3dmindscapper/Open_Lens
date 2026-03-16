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
├── Launch Translator.bat          # Windows launcher for the desktop GUI
├── Launch Web Server.bat          # Windows launcher for the web server
├── README.md                      # This file
├── Open_Lens_Core/
│   ├── translate.py               # CLI entry point
│   ├── translator_ui.py           # Tkinter GUI
│   ├── web_app.py                 # Flask web server (LAN-accessible)
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

## Web Server — Share Over Your Network

`web_app.py` is a fully self-contained Flask web server. Once running, **any phone, tablet, or PC on the same Wi-Fi or LAN** can open a browser, upload a document, and download the translated version — no installation required on the remote device.

### Starting the server

Double-click **Launch Web Server.bat**, or run manually:

```bash
cd Open_Lens_Core
python web_app.py
```

The console prints two URLs:

```
  Local:    http://localhost:5000
  Network:  http://192.168.x.x:5000   ← share this with other devices
```

Open the **Network** URL on any device connected to the same network. The web UI lets you:

1. Drag-and-drop (or browse) a file — PDF, JPG, PNG, TIFF, BMP, or WebP (up to 100 MB)
2. Pick source and target languages (auto-detect available)
3. Click **Translate** and watch live progress logs stream in the browser
4. Click **Download** when done — multi-page results are automatically bundled into a ZIP

Uploaded and output files are **automatically deleted after 10 minutes**.

### Making it reachable from outside your LAN

By default the server only works on your local network. To expose it over the internet:

- **Port forward** port `5000` on your router to the host machine, **or**
- Use a tunnel tool such as [ngrok](https://ngrok.com/): `ngrok http 5000` — this gives you a public HTTPS URL instantly with no router changes.

### Changing the port

Set the `PORT` environment variable before launching:

```bash
set PORT=8080
python web_app.py
```

## License

See [Open_Lens_Core/LICENSE](Open_Lens_Core/LICENSE).
