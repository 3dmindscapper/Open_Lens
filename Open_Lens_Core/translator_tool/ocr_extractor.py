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

    # Group by (block_num, par_num, line_num) to get individual lines first,
    # then merge lines within the same Tesseract paragraph only when they
    # have consistent left-alignment and tight vertical spacing.
    line_group_keys = ["block_num", "par_num", "line_num"]

    # Build per-line blocks keyed by (block_num, par_num)
    line_blocks: Dict[tuple, List[TextBlock]] = {}
    for keys, group in df.groupby(line_group_keys, sort=True):
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
        para_key = (int(keys[0]), int(keys[1]))  # (block_num, par_num)

        # Collect word-level bounding boxes for precise inpainting masks
        word_boxes = []
        for _, row in group.iterrows():
            wx = int(row["left"])
            wy = int(row["top"])
            ww = int(row["width"])
            wh = int(row["height"])
            word_boxes.append({"x": wx, "y": wy, "w": ww, "h": wh,
                               "x2": wx + ww, "y2": wy + wh,
                               "text": str(row["text"]).strip()})

        blk: TextBlock = {
            "x": x1, "y": y1, "w": w, "h": h,
            "x2": x2, "y2": y2,
            "text": text, "mean_conf": mean_conf,
            "word_boxes": word_boxes,
        }
        line_blocks.setdefault(para_key, []).append(blk)

    # Within each Tesseract paragraph, merge consecutive lines only if they
    # share similar left alignment and are tightly spaced vertically.
    blocks: List[TextBlock] = []
    img_w = image.size[0] if hasattr(image, 'size') else 1000
    left_tolerance = max(20, int(img_w * 0.025))

    for _para_key, lines in line_blocks.items():
        lines.sort(key=lambda b: b["y"])
        if not lines:
            continue

        current = dict(lines[0])
        current["word_boxes"] = list(current.get("word_boxes", []))
        # Use the height of the *first* line as gap reference to prevent
        # snowball merging (merged block height grows, loosening tolerance).
        first_line_h = lines[0]["h"]
        for nxt in lines[1:]:
            gap = nxt["y"] - current["y2"]
            left_diff = abs(nxt["x"] - current["x"])

            # Merge only if: tight vertical gap AND similar left margin
            if 0 <= gap <= first_line_h * 0.3 and left_diff <= left_tolerance:
                current = {
                    "x":  min(current["x"],  nxt["x"]),
                    "y":  min(current["y"],  nxt["y"]),
                    "x2": max(current["x2"], nxt["x2"]),
                    "y2": max(current["y2"], nxt["y2"]),
                    "w":  max(current["x2"], nxt["x2"]) - min(current["x"], nxt["x"]),
                    "h":  max(current["y2"], nxt["y2"]) - min(current["y"], nxt["y"]),
                    "text": current["text"] + " " + nxt["text"],
                    "mean_conf": (current["mean_conf"] + nxt["mean_conf"]) / 2,
                    "word_boxes": current["word_boxes"] + nxt.get("word_boxes", []),
                }
            else:
                blocks.append(current)
                current = dict(nxt)
        blocks.append(current)

    blocks = _split_wide_blocks(blocks, image.size[0])
    return blocks


def _split_wide_blocks(blocks: List[TextBlock], img_width: int) -> List[TextBlock]:
    """Split blocks that span large horizontal gaps (e.g., merged table columns).

    Only splits when words sit on the same line with a large gap between them.
    Multi-line blocks wrapped within one column are NOT split, to avoid
    fragmenting paragraphs that have lines of different widths.
    """
    result = []
    for block in blocks:
        word_boxes = block.get("word_boxes", [])
        if len(word_boxes) < 3:
            result.append(block)
            continue

        # Group words into lines by similar y-coordinate
        word_heights = [wb["h"] for wb in word_boxes if wb.get("h", 0) > 0]
        if not word_heights:
            result.append(block)
            continue
        median_wh = sorted(word_heights)[len(word_heights) // 2]
        y_tolerance = max(median_wh * 0.5, 8)

        lines: List[List[dict]] = []
        for wb in sorted(word_boxes, key=lambda w: (w["y"], w["x"])):
            placed = False
            for line in lines:
                if abs(wb["y"] - line[0]["y"]) <= y_tolerance:
                    line.append(wb)
                    placed = True
                    break
            if not placed:
                lines.append([wb])

        # For each line, find large horizontal gaps that indicate column breaks
        # A split is valid only if it appears in ALL (or most) lines of the block
        split_x_candidates: List[float] = []
        for line in lines:
            if len(line) < 2:
                continue
            line_sorted = sorted(line, key=lambda w: w["x"])
            gap_values = []
            for i in range(1, len(line_sorted)):
                prev_right = line_sorted[i - 1]["x"] + line_sorted[i - 1]["w"]
                curr_left = line_sorted[i]["x"]
                gap_values.append(curr_left - prev_right)

            positive_gaps = [g for g in gap_values if g > 0]
            if not positive_gaps:
                continue
            median_gap = sorted(positive_gaps)[len(positive_gaps) // 2]
            threshold = max(40, int(median_gap * 3.5))

            for i in range(1, len(line_sorted)):
                prev_right = line_sorted[i - 1]["x"] + line_sorted[i - 1]["w"]
                curr_left = line_sorted[i]["x"]
                gap = curr_left - prev_right
                if gap >= threshold:
                    split_x_candidates.append((prev_right + curr_left) / 2)

        if not split_x_candidates:
            result.append(block)
            continue

        # For single-line blocks: any large gap is a valid split
        # For multi-line blocks: require the split to appear in most lines
        if len(lines) > 1:
            # Cluster the split candidates and keep only consistent ones
            split_x_candidates.sort()
            clusters: List[List[float]] = []
            for sx in split_x_candidates:
                placed = False
                for cl in clusters:
                    if abs(sx - cl[0]) < 60:
                        cl.append(sx)
                        placed = True
                        break
                if not placed:
                    clusters.append([sx])

            # Keep splits that appear in at least half the lines
            min_count = max(1, len(lines) // 2)
            valid_splits = [sum(cl) / len(cl) for cl in clusters if len(cl) >= min_count]

            if not valid_splits:
                result.append(block)
                continue
        else:
            valid_splits = split_x_candidates

        valid_splits.sort()

        # Assign each word to a column based on the split points
        columns: List[List[dict]] = [[] for _ in range(len(valid_splits) + 1)]
        for wb in word_boxes:
            word_center = wb["x"] + wb["w"] / 2
            col_idx = 0
            for sx in valid_splits:
                if word_center > sx:
                    col_idx += 1
                else:
                    break
            columns[col_idx].append(wb)

        # Create sub-blocks for each non-empty column
        for col_words in columns:
            if not col_words:
                continue

            col_words_ordered = sorted(col_words, key=lambda w: (w["y"], w["x"]))
            x1 = min(wb["x"] for wb in col_words)
            y1 = min(wb["y"] for wb in col_words)
            x2 = max(wb["x2"] for wb in col_words)
            y2 = max(wb["y2"] for wb in col_words)

            text = " ".join(wb.get("text", "") for wb in col_words_ordered).strip()
            if not text:
                continue

            sub_block: TextBlock = {
                "x": x1, "y": y1,
                "x2": x2, "y2": y2,
                "w": x2 - x1, "h": y2 - y1,
                "text": text,
                "mean_conf": block.get("mean_conf", 0),
                "word_boxes": col_words,
            }
            result.append(sub_block)

    return result


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
