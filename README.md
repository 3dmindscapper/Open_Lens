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
pip install -r layout-parser/requirements_translator.txt
```

Key packages: `Pillow`, `numpy`, `opencv-python`, `pytesseract`, `langdetect`, `argostranslate`, `pdf2image`.

Optional for RTL languages: `arabic-reshaper`, `python-bidi`.

## Quick Start

### GUI

Double-click **Launch Translator.bat** or run:

```bash
cd layout-parser
python translator_ui.py
```

The GUI defaults to translating to **English** with automatic source language detection. Select a different target language from the dropdown if needed.

### Command Line

```bash
cd layout-parser

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
├── layout-parser/
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

## License

See [layout-parser/LICENSE](layout-parser/LICENSE).
