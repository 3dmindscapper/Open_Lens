# Document Translation Tool

Detects text in images and PDFs, translates it, removes the original, and renders the translation back in the same position — like Google Lens, but for whole documents and **fully offline** (no internet connection, no API keys, no external services).

The pipeline is **pluggable**: it auto-detects the best available backends (OCR, layout detection, inpainting, translation) and falls back gracefully to baseline engines when optional dependencies are not installed.

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

### 1 — Python dependencies (core)

```bash
pip install -r requirements_translator.txt
```

> **Translation models** — Argos Translate downloads neural MT models (~100 MB each) on
> **first use** of a given language pair, then caches them permanently on disk.
> After that first download, the tool runs with **no internet connection required**.

### 1b — Optional enhanced backends

Install any combination — the pipeline auto-detects what is available:

```bash
# Better OCR (recommended)
pip install paddlepaddle paddleocr

# Semantic layout detection
pip install layoutparser
# + Detectron2: https://detectron2.readthedocs.io/en/latest/tutorials/install.html

# Deep-learning inpainting (recommended for complex backgrounds)
pip install simple-lama-inpainting torch

# Local LLM translation via Ollama (optional)
# Install Ollama from https://ollama.com/ then:
#   ollama pull qwen2.5:7b
```

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
usage: translate_doc [-h] [-t LANG] [-s LANG] [-o PATH] [--tesseract PATH]
                     [--font PATH] [-q]
                     [--layout-engine {auto,layoutparser,paddleocr,none}]
                     [--ocr-engine {auto,paddleocr,tesseract}]
                     [--inpaint-engine {auto,lama,telea}]
                     [--translator {auto,argos,ollama}]
                     [--device {auto,cuda,cpu}] [--ollama-url URL]
                     [--ollama-model MODEL] [--glossary PATH]
                     INPUT

positional arguments:
  INPUT                 Path to input file (PDF, JPG, PNG, TIFF, …)

options:
  -t, --target LANG     Target language code (default: en). Examples: en, fr, de, es, ja, zh-CN, ar …
  -s, --source LANG     Source language code (default: auto-detect)
  -o, --output PATH     Output file path (default: <input>_translated_<lang>.<ext>)
  --tesseract PATH      Path to Tesseract binary (if not on PATH)
  --font PATH           Path to a .ttf font file (system font auto-detected if omitted)
  -q, --quiet           Suppress progress output

Engine selection (auto-detected by default):
  --layout-engine       Layout detection: auto | layoutparser | paddleocr | none
  --ocr-engine          OCR: auto | paddleocr | tesseract
  --inpaint-engine      Inpainting: auto | lama | telea
  --translator          Translation: auto | argos | ollama
  --device              Compute device: auto | cuda | cpu
  --ollama-url URL      Ollama server URL (default: http://localhost:11434)
  --ollama-model MODEL  Ollama model name (default: qwen2.5:7b)
  --glossary PATH       TSV glossary file (source_term<TAB>target_term)
```

### Python API

```python
from translator_tool import process_document

# Simplest usage – auto-detect everything, translate to English
output_path = process_document("scan.pdf", target_lang="en")

# Full control via TranslationConfig
from translator_tool.config import TranslationConfig

config = TranslationConfig(
    target_lang="fr",
    source_lang="de",
    layout_engine="layoutparser",   # or "paddleocr", "none", "auto"
    ocr_engine="paddleocr",         # or "tesseract", "auto"
    inpaint_engine="lama",          # or "telea", "auto"
    translator_engine="argos",      # or "ollama", "auto"
    device="auto",                  # or "cuda", "cpu"
    font_path=r"C:\Windows\Fonts\arial.ttf",
    glossary_path="glossary.tsv",   # optional domain-specific terms
)

output_path = process_document(
    input_path="document.jpg",
    target_lang="fr",
    output_path="result.jpg",
    config=config,
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
  Layout detection (Layout Parser / PaddleOCR / none)      ← NEW
  Identifies text, title, table, figure regions
        │
        ▼
  Language detection (langdetect on a quick OCR sample)
        │
        ▼
  OCR (PaddleOCR / Tesseract) — per layout region or full page
  Produces text blocks with bounding boxes
        │
        ▼
  Translate blocks (Argos Translate / Ollama LLM)          ← NEW
  Optional TSV glossary post-processing
        │
        ▼
  Inpaint original text (LaMa deep inpainting / OpenCV TELEA)
        │                                                    ← IMPROVED
        ▼
  Font analysis → colour (K-means), weight (distance transform),
  size (binary-search calibration), alignment detection      ← NEW
        │
        ▼
  Render translated text into bounding boxes
  (calibrated font size, detected alignment, measured line spacing,
   sampled ink colour, RTL support)
        │
        ▼
  Save output (same format as input)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `TesseractNotFoundError` | Install Tesseract and add to PATH, or use `--tesseract` flag |
| Translation fails / empty output | Language pair model not yet available — run once with internet to download it; afterwards it works offline |
| Garbled Arabic/Hebrew text | `pip install arabic-reshaper python-bidi` |
| Bad OCR on non-English scan | Install PaddleOCR (`pip install paddlepaddle paddleocr`) or the Tesseract language pack, and set `--source` |
| `PDFInfoNotInstalledError` | Install Poppler and add to PATH |
| Font doesn't support target script | Use `--font` with a Unicode font like Arial Unicode MS |
| Inpainting leaves artefacts on stamps / watermarks | Install LaMa: `pip install simple-lama-inpainting torch` |
| Want better layout-aware OCR | Install Layout Parser + Detectron2 or PaddleOCR |
| Ollama translation not working | Ensure the Ollama server is running (`ollama serve`) and the model is pulled (`ollama pull qwen2.5:7b`) |
