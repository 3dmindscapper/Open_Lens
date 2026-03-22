"""
OCR extraction module.

Supports three engines (auto-selected in priority order):
  1. **EasyOCR** (preferred): PyTorch-based, 80+ languages, tight bounding
     boxes, robust on complex backgrounds.
  2. **PaddleOCR**: PaddlePaddle-based, 80+ languages.
  3. **Tesseract** (fallback): via pytesseract, requires Tesseract binary.

The public API (``extract_text_blocks``, ``extract_full_text``) auto-selects
the best available engine unless the caller specifies one explicitly.
"""

from __future__ import annotations

import io
import csv
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-OCR image enhancement
# ---------------------------------------------------------------------------

def _enhance_for_ocr(img_np: np.ndarray) -> np.ndarray:
    """Enhance an image for better Tesseract accuracy.

    Suppresses light watermarks / stamps and enhances text contrast by:
      1. Converting to grayscale.
      2. Applying CLAHE to boost local contrast.
      3. Using adaptive thresholding to produce a clean binary image.

    Returns a grayscale numpy array suitable for Tesseract.
    """
    import cv2

    if img_np.ndim == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()

    # CLAHE: boost local contrast without blowing out bright areas
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Adaptive threshold: suppresses light watermarks and stamps while
    # preserving dark body text.  Block size must be large enough to
    # capture local background variation.
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )

    return binary


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
    min_confidence: float = 30.0,
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
        if len(word_boxes) < 2:
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
            threshold = max(30, int(median_gap * 2.0))

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

            # Keep splits that appear in at least a third of the lines
            min_count = max(1, len(lines) // 3)
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

def extract_full_text(image: Image.Image, ocr_lang: str = "eng",
                      engine: str = "auto") -> str:
    """Return all OCR'd text from a page as a single string."""
    if engine == "auto":
        engine = _pick_engine()

    if engine == "easyocr":
        try:
            blocks = _extract_easyocr(image, ocr_lang)
            return " ".join(b["text"] for b in blocks if b.get("text"))
        except Exception:
            pass  # fall through to Tesseract

    if engine == "paddleocr":
        blocks = _extract_paddle(image, ocr_lang)
        return " ".join(b["text"] for b in blocks if b.get("text"))

    configure_tesseract_path()
    pytesseract = _require_pytesseract()
    img_np = np.array(image)
    return pytesseract.image_to_string(img_np, lang=ocr_lang)


# ---------------------------------------------------------------------------
# PaddleOCR engine
# ---------------------------------------------------------------------------

_PADDLE_OCR_CACHE: Dict[str, Any] = {}


def _get_paddle_ocr(lang: str = "en", device: str = "cpu"):
    """Return a cached PaddleOCR instance for the given language."""
    key = f"{lang}_{device}"
    if key in _PADDLE_OCR_CACHE:
        return _PADDLE_OCR_CACHE[key]

    from paddleocr import PaddleOCR  # type: ignore

    use_gpu = device != "cpu"
    # PaddleOCR uses its own language codes (e.g. "en", "french", "ch", "japan")
    paddle_lang = _tesseract_to_paddle_lang(lang)
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang=paddle_lang,
        use_gpu=use_gpu,
        show_log=False,
    )
    _PADDLE_OCR_CACHE[key] = ocr
    log.info("PaddleOCR loaded: lang=%s device=%s", paddle_lang, device)
    return ocr


# Tesseract lang code → PaddleOCR lang code
_TESS_TO_PADDLE = {
    "eng": "en", "fra": "french", "deu": "german", "spa": "es",
    "ita": "it", "por": "pt", "nld": "nl", "rus": "ru",
    "chi_sim": "ch", "chi_tra": "chinese_cht", "jpn": "japan",
    "kor": "korean", "ara": "ar", "hin": "hi", "tur": "tr",
    "pol": "pl", "ukr": "uk", "vie": "vi", "tha": "th",
    "tam": "ta", "tel": "te", "kan": "ka", "mar": "mr",
    "ben": "bn", "nep": "ne", "urd": "ur", "fas": "fa",
    "heb": "he", "swe": "sv", "nor": "no", "dan": "da",
    "fin": "fi", "ces": "cs", "ron": "ro", "hun": "hu",
    "bul": "bg", "hrv": "hr", "slk": "sk", "slv": "sl",
    "srp": "rs_latin", "cat": "ca", "ell": "el",
}


def _tesseract_to_paddle_lang(tess_lang: str) -> str:
    """Map Tesseract language code to PaddleOCR language code."""
    # Handle compound codes like "eng+fra"
    primary = tess_lang.split("+")[0].strip()
    return _TESS_TO_PADDLE.get(primary, "en")


def _extract_paddle(
    image: Image.Image,
    ocr_lang: str = "eng",
    min_confidence: float = 0.3,
    device: str = "cpu",
) -> List[TextBlock]:
    """Extract text blocks using PaddleOCR."""
    ocr = _get_paddle_ocr(ocr_lang, device)
    img_np = np.array(image)
    result = ocr.ocr(img_np, cls=True)

    if not result or not result[0]:
        return []

    # PaddleOCR returns: [[[box, (text, conf)], ...]]
    # where box = [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    raw_lines: List[Dict[str, Any]] = []
    for line_data in result[0]:
        box_pts = line_data[0]
        text, conf = line_data[1]
        text = text.strip()
        if not text or conf < min_confidence:
            continue

        # Convert polygon to axis-aligned bounding box
        xs = [pt[0] for pt in box_pts]
        ys = [pt[1] for pt in box_pts]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))

        raw_lines.append({
            "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
            "x2": x2, "y2": y2,
            "text": text,
            "mean_conf": float(conf * 100),
            "word_boxes": [{"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                            "x2": x2, "y2": y2, "text": text}],
        })

    if not raw_lines:
        return []

    # Group nearby lines into paragraph blocks (same logic as Tesseract path)
    blocks = _merge_lines_into_blocks(raw_lines, image.size[0])
    blocks = _split_wide_blocks(blocks, image.size[0])
    return blocks


# ---------------------------------------------------------------------------
# EasyOCR engine
# ---------------------------------------------------------------------------

_EASYOCR_CACHE: Dict[str, Any] = {}

# Tesseract lang code → EasyOCR lang code(s)
_TESS_TO_EASYOCR: Dict[str, List[str]] = {
    "eng": ["en"], "fra": ["fr"], "deu": ["de"], "spa": ["es"],
    "ita": ["it"], "por": ["pt"], "nld": ["nl"], "rus": ["ru"],
    "chi_sim": ["ch_sim"], "chi_tra": ["ch_tra"], "jpn": ["ja"],
    "kor": ["ko"], "ara": ["ar"], "hin": ["hi"], "tur": ["tr"],
    "pol": ["pl"], "ukr": ["uk"], "vie": ["vi"], "tha": ["th"],
    "tam": ["ta"], "tel": ["te"], "kan": ["kn"], "mar": ["mr"],
    "ben": ["bn"], "nep": ["ne"], "urd": ["ur"], "fas": ["fa"],
    "heb": ["he"], "swe": ["sv"], "nor": ["no"], "dan": ["da"],
    "fin": ["fi"], "ces": ["cs"], "ron": ["ro"], "hun": ["hu"],
    "bul": ["bg"], "hrv": ["hr"], "slk": ["sk"], "slv": ["sl"],
    "srp": ["rs_latin"], "cat": ["ca"], "ell": ["el"],
    "lat": ["la"], "ind": ["id"], "msa": ["ms"],
}


def _tesseract_to_easyocr_langs(tess_lang: str) -> List[str]:
    """Map Tesseract language code to EasyOCR language code(s)."""
    parts = tess_lang.split("+")
    result = []
    for p in parts:
        p = p.strip()
        mapped = _TESS_TO_EASYOCR.get(p, ["en"])
        for lang in mapped:
            if lang not in result:
                result.append(lang)
    return result or ["en"]


def _get_easyocr_reader(langs: List[str], device: str = "cpu"):
    """Return a cached EasyOCR Reader instance."""
    key = "_".join(sorted(langs)) + f"_{device}"
    if key in _EASYOCR_CACHE:
        return _EASYOCR_CACHE[key]

    import easyocr  # type: ignore

    gpu = device != "cpu"
    reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
    _EASYOCR_CACHE[key] = reader
    log.info("EasyOCR reader loaded: langs=%s device=%s", langs, device)
    return reader


def _extract_easyocr(
    image: Image.Image,
    ocr_lang: str = "eng",
    min_confidence: float = 0.3,
    device: str = "cpu",
) -> List[TextBlock]:
    """Extract text blocks using EasyOCR."""
    langs = _tesseract_to_easyocr_langs(ocr_lang)
    reader = _get_easyocr_reader(langs, device)
    img_np = np.array(image)

    # EasyOCR returns: [[box, text, conf], ...]
    # where box = [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    results = reader.readtext(img_np, paragraph=False)

    if not results:
        return []

    raw_lines: List[TextBlock] = []
    for detection in results:
        box_pts, text, conf = detection
        text = text.strip()
        if not text or conf < min_confidence:
            continue

        xs = [pt[0] for pt in box_pts]
        ys = [pt[1] for pt in box_pts]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))

        if x2 <= x1 or y2 <= y1:
            continue

        raw_lines.append({
            "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
            "x2": x2, "y2": y2,
            "text": text,
            "mean_conf": float(conf * 100),
            "word_boxes": [{"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                            "x2": x2, "y2": y2, "text": text}],
        })

    if not raw_lines:
        return []

    blocks = _merge_lines_into_blocks(raw_lines, image.size[0])
    blocks = _split_wide_blocks(blocks, image.size[0])
    return blocks


def _merge_lines_into_blocks(
    lines: List[TextBlock],
    img_width: int,
) -> List[TextBlock]:
    """Merge adjacent text lines into paragraph-level blocks.

    Used by both Tesseract and PaddleOCR paths for consistency.
    """
    if not lines:
        return []

    lines_sorted = sorted(lines, key=lambda b: (b["y"], b["x"]))
    left_tolerance = max(20, int(img_width * 0.025))

    blocks: List[TextBlock] = []
    current = dict(lines_sorted[0])
    current["word_boxes"] = list(current.get("word_boxes", []))
    first_line_h = lines_sorted[0]["h"]

    for nxt in lines_sorted[1:]:
        gap = nxt["y"] - current["y2"]
        left_diff = abs(nxt["x"] - current["x"])

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
            current["word_boxes"] = list(current.get("word_boxes", []))
            first_line_h = nxt["h"]

    blocks.append(current)
    return blocks


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def _pick_engine() -> str:
    """Auto-select OCR engine based on availability."""
    try:
        import easyocr  # noqa: F401
        return "easyocr"
    except Exception:
        pass
    try:
        from paddleocr import PaddleOCR  # noqa: F401
        return "paddleocr"
    except Exception:
        return "tesseract"


def extract_text_blocks_unified(
    image: Image.Image,
    ocr_lang: str = "eng",
    engine: str = "auto",
    device: str = "cpu",
    min_confidence: float = 20.0,
    min_text_length: int = 2,
    region_offset: Optional[Tuple[int, int]] = None,
) -> List[TextBlock]:
    """Extract text blocks using the best available engine.

    Args:
        image:          Input PIL Image (RGB).
        ocr_lang:       Tesseract-style language string (mapped for PaddleOCR).
        engine:         ``"paddleocr"``, ``"tesseract"``, or ``"auto"``.
        device:         ``"cpu"`` or ``"cuda"`` (for PaddleOCR).
        min_confidence: Minimum confidence threshold.
        min_text_length: Minimum text length per block.
        region_offset:  (x_offset, y_offset) to add to all coordinates when
                        extracting from a cropped region of a larger image.

    Returns:
        List of TextBlock dicts.
    """
    if engine == "auto":
        engine = _pick_engine()

    if engine == "easyocr":
        try:
            easyocr_conf = min_confidence / 100.0 if min_confidence > 1 else min_confidence
            blocks = _extract_easyocr(image, ocr_lang, min_confidence=easyocr_conf, device=device)
        except Exception as exc:
            log.warning("EasyOCR failed (%s), falling back to Tesseract", exc)
            blocks = extract_text_blocks(image, ocr_lang, min_confidence, min_text_length)
    elif engine == "paddleocr":
        try:
            paddle_conf = min_confidence / 100.0 if min_confidence > 1 else min_confidence
            blocks = _extract_paddle(image, ocr_lang, min_confidence=paddle_conf, device=device)
        except Exception as exc:
            log.warning("PaddleOCR failed (%s), falling back to Tesseract", exc)
            blocks = extract_text_blocks(image, ocr_lang, min_confidence, min_text_length)
    else:
        blocks = extract_text_blocks(image, ocr_lang, min_confidence, min_text_length)

    # Filter by text length
    blocks = [b for b in blocks if len(b.get("text", "").replace(" ", "")) >= min_text_length]

    # Apply region offset if extracting from a cropped sub-region
    if region_offset:
        ox, oy = region_offset
        for b in blocks:
            b["x"] += ox
            b["y"] += oy
            b["x2"] += ox
            b["y2"] += oy
            for wb in b.get("word_boxes", []):
                wb["x"] += ox
                wb["y"] += oy
                wb["x2"] += ox
                wb["y2"] += oy

    return blocks
