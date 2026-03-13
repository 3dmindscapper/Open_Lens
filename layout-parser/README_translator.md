# Document Translation Tool

Detects text in images and PDFs, translates it, removes the original, and renders the translation back in the same position — like Google Lens, but for whole documents and **fully offline** (no internet connection, no API keys, no external services).

## Supported input formats
| Format | Notes |
|--------|-------|
| JPG / JPEG | Any resolution |
| PNG | Any resolution |
| TIFF / BMP / WebP | Any resolution |
| PDF | Each page processed independently |

Output is saved in the **same format** as the input (multi-page PDFs stay as PDF).

---

## Installation

### 1 — Python dependencies

```bash
pip install -r requirements_translator.txt
```

> **Translation models** — Argos Translate downloads neural MT models (~100 MB each) on
> **first use** of a given language pair, then caches them permanently on disk.
> After that first download, the tool runs with **no internet connection required**.

### 2 — Tesseract OCR binary

Tesseract must be installed **separately** from the Python package.

| OS | Command / Link |
|----|----------------|
| **Windows** | Download installer from https://github.com/UB-Mannheim/tesseract/wiki — add to PATH or use `--tesseract` flag |
| **macOS** | `brew install tesseract` |
| **Linux** | `sudo apt-get install tesseract-ocr` |

Install extra language packs for non-English documents:

```bash
# Examples
sudo apt-get install tesseract-ocr-fra   # French
sudo apt-get install tesseract-ocr-deu   # German
sudo apt-get install tesseract-ocr-chi-sim  # Chinese (Simplified)
# Windows: re-run installer and select additional languages
```

### 3 — Poppler (PDF only)

| OS | Command / Link |
|----|----------------|
| **Windows** | Download from https://github.com/oschwartz10612/poppler-windows/releases — add `bin/` to PATH |
| **macOS** | `brew install poppler` |
| **Linux** | `sudo apt-get install poppler-utils` |

### 4 — RTL language support (optional)

For correct rendering of Arabic, Hebrew, Persian, and Urdu:

```bash
pip install arabic-reshaper python-bidi
```

---

## Usage

### Command line

```bash
# Translate a PDF to English
python translate.py report.pdf -t en

# Translate a scanned image from German to Spanish
python translate.py scan.jpg -t es -s de

# Specify output path
python translate.py invoice.png -t fr -o invoice_french.png

# Windows: specify Tesseract path explicitly
python translate.py document.pdf -t en --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"

# Use a custom font
python translate.py scan.jpg -t ja --font "C:\Windows\Fonts\msgothic.ttc"
```

### Full argument reference

```
usage: translate_doc [-h] -t LANG [-s LANG] [-o PATH] [--tesseract PATH] [--font PATH] [-q] INPUT

positional arguments:
  INPUT                 Path to input file (PDF, JPG, PNG, TIFF, …)

options:
  -t, --target LANG     Target language code (required). Examples: en, fr, de, es, it, pt, ru, ja, zh-CN, ar, ko …
  -s, --source LANG     Source language code (default: auto-detect)
  -o, --output PATH     Output file path (default: <input>_translated_<lang>.<ext>)
  --tesseract PATH      Path to Tesseract binary (if not on PATH)
  --font PATH           Path to a .ttf font file (system font auto-detected if omitted)
  -q, --quiet           Suppress progress output
```

### Python API

```python
from translator_tool import process_document

# Simplest usage – auto-detect source language, translate to English
output_path = process_document("scan.pdf", target_lang="en")

# Full control
output_path = process_document(
    input_path="document.jpg",
    target_lang="fr",
    source_lang="de",          # skip auto-detection
    output_path="result.jpg",
    font_path=r"C:\Windows\Fonts\arial.ttf",
    verbose=True,
)
```

---

## Language codes

Use standard ISO 639-1 codes for `--target` and `--source`:

| Code | Language | Code | Language |
|------|----------|------|----------|
| `en` | English  | `ja` | Japanese |
| `fr` | French   | `ko` | Korean   |
| `de` | German   | `zh-CN` | Chinese (Simplified) |
| `es` | Spanish  | `zh-TW` | Chinese (Traditional) |
| `it` | Italian  | `ar` | Arabic   |
| `pt` | Portuguese | `ru` | Russian |
| `nl` | Dutch    | `tr` | Turkish  |
| `pl` | Polish   | `vi` | Vietnamese |
| `hi` | Hindi    | `th` | Thai     |

---

## Pipeline overview

```
Input file (PDF / image)
        │
        ▼
  Load pages as RGB images
        │
        ▼
  OCR pass 1 (English) → detect language (langdetect)
        │
        ▼
  OCR pass 2 (correct language pack) → text blocks + bounding boxes
        │
        ▼
  Translate each block  (Argos Translate — local neural MT, fully offline)
  models downloaded once on first use, then cached to disk
        │
        ▼
  Inpaint original text regions (OpenCV TELEA algorithm)
        │
        ▼
  Render translated text into bounding boxes
  (auto font-size fitting, word wrap, RTL support)
        │
        ▼
  Save output (same format as input)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `TesseractNotFoundError` | Install Tesseract and add to PATH, or use `--tesseract` flag |
| Translation fails / empty output | Language pair model not yet available—run once with internet to download it; afterwards it works offline |
| Garbled Arabic/Hebrew text | `pip install arabic-reshaper python-bidi` |
| Bad OCR on non-English scan | Install the Tesseract language pack and set `--source` |
| `PDFInfoNotInstalledError` | Install Poppler and add to PATH |
| Font doesn't support target script | Use `--font` with a Unicode font like Arial Unicode MS |
