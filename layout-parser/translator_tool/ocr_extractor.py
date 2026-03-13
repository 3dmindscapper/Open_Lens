"""
OCR extraction module.

Uses Tesseract (via pytesseract) to extract text blocks with their bounding boxes.
Works at paragraph level so each block is a coherent unit for translation.
"""

from __future__ import annotations

import io
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# Tesseract helpers
# ---------------------------------------------------------------------------

def _require_pytesseract():
    try:
        import pytesseract
        return pytesseract
    except ImportError:
        raise ImportError(
            "pytesseract is required for OCR.\n"
            "Install it with:  pip install pytesseract\n"
            "You also need Tesseract installed on your system:\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  macOS:   brew install tesseract\n"
            "  Linux:   sudo apt-get install tesseract-ocr"
        )


def configure_tesseract_path(tesseract_cmd: Optional[str] = None):
    """Optionally override tesseract binary path (useful on Windows)."""
    pytesseract = _require_pytesseract()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        return

    # Auto-detect common Windows installation locations
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for path in candidates:
            if Path(path).exists():
                pytesseract.pytesseract.tesseract_cmd = path
                return


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------

TextBlock = Dict[str, Any]
"""
Each extracted block is a dict with keys:
  x, y, w, h   – bounding box (top-left origin, pixels)
  x2, y2       – bottom-right corner
  text         – concatenated OCR text for the block
  mean_conf    – average Tesseract confidence (0–100)
"""


def _raw_ocr_dataframe(image_np: np.ndarray, ocr_lang: str) -> pd.DataFrame:
    pytesseract = _require_pytesseract()
    raw = pytesseract.image_to_data(
        image_np,
        lang=ocr_lang,
        output_type=pytesseract.Output.STRING,
    )
    df = pd.read_csv(
        io.StringIO(raw),
        sep="\t",
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        converters={"text": str},
    )
    return df


def extract_text_blocks(
    image: Image.Image,
    ocr_lang: str = "eng",
    min_confidence: float = 20.0,
    min_text_length: int = 2,
) -> List[TextBlock]:
    """Extract paragraph-level text blocks with bounding boxes via Tesseract.

    Args:
        image:          Input PIL Image (RGB).
        ocr_lang:       Tesseract language string, e.g. ``"eng"``, ``"fra"``,
                        ``"eng+fra"``.  Use ``"eng+osd"`` for orientation
                        detection.
        min_confidence: Minimum per-word confidence to include (0–100).
        min_text_length: Minimum non-space character length for a block.

    Returns:
        List of block dicts ``{x, y, w, h, x2, y2, text, mean_conf}``.
    """
    configure_tesseract_path()

    img_np = np.array(image)
    df = _raw_ocr_dataframe(img_np, ocr_lang)

    # Keep only rows with a real word
    df = df[df["text"].notna()].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df[df["conf"].apply(_safe_float) >= min_confidence]

    if df.empty:
        return []

    # Group by (block_num, par_num) → one dict per paragraph
    group_keys = ["block_num", "par_num"]
    blocks: List[TextBlock] = []

    for _, group in df.groupby(group_keys, sort=True):
        text = " ".join(group["text"].tolist()).strip()
        if len(text.replace(" ", "")) < min_text_length:
            continue

        x1 = int(group["left"].min())
        y1 = int(group["top"].min())
        x2 = int((group["left"] + group["width"]).max())
        y2 = int((group["top"] + group["height"]).max())
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            continue

        mean_conf = float(group["conf"].apply(_safe_float).mean())

        blocks.append(
            {
                "x": x1,
                "y": y1,
                "w": w,
                "h": h,
                "x2": x2,
                "y2": y2,
                "text": text,
                "mean_conf": mean_conf,
            }
        )

    return blocks


def _safe_float(val) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Full-page text (for language detection)
# ---------------------------------------------------------------------------

def extract_full_text(image: Image.Image, ocr_lang: str = "eng") -> str:
    """Return all OCR'd text from a page as a single string."""
    configure_tesseract_path()
    pytesseract = _require_pytesseract()
    img_np = np.array(image)
    return pytesseract.image_to_string(img_np, lang=ocr_lang)
