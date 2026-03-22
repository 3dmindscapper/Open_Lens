"""
Post-OCR text classification.

Classifies each OCR block as one of:
  - **translatable**: Regular document text that should be translated.
  - **watermark**: Light/transparent text overlaid on the document (kept as-is).
  - **stamp**: Text from official stamps or seals (kept as-is).
  - **noise**: Very low-confidence OCR fragments (dropped entirely).

Classification signals:
  1. Ink contrast — watermarks are significantly lighter than body text.
  2. Size anomaly — watermarks are typically much larger than body text.
  3. OCR confidence — garbled stamp/watermark text has low confidence.
  4. Colour — stamps often use red/blue ink distinct from black body text.

This module does NOT require Layout Parser or any ML model — it uses
purely image-based heuristics on the original document.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# Block classification labels
TRANSLATABLE = "translatable"
WATERMARK = "watermark"
STAMP = "stamp"
NOISE = "noise"


# ---------------------------------------------------------------------------
# Per-block feature extraction
# ---------------------------------------------------------------------------

def _ink_contrast(img_np: np.ndarray, block: Dict[str, Any]) -> float:
    """Measure how dark the text ink is relative to the local background.

    Returns a value in [0, 255] where higher = darker ink = more visible.
    Watermark text returns low values (< 40–60).
    """
    h_img, w_img = img_np.shape[:2]
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np

    # Sample background from a border ring around the block
    pad = 20
    bx1 = max(0, block["x"] - pad)
    by1 = max(0, block["y"] - pad)
    bx2 = min(w_img, block["x2"] + pad)
    by2 = min(h_img, block["y2"] + pad)
    border_ring = np.ones((by2 - by1, bx2 - bx1), dtype=bool)
    ix1 = block["x"] - bx1
    iy1 = block["y"] - by1
    ix2 = block["x2"] - bx1
    iy2 = block["y2"] - by1
    border_ring[max(0, iy1):iy2, max(0, ix1):ix2] = False

    border_pixels = gray[by1:by2, bx1:bx2][border_ring]
    if len(border_pixels) == 0:
        return 255.0  # assume high contrast

    bg_brightness = float(np.median(border_pixels))

    # Sample ink pixels from word bounding boxes
    word_boxes = block.get("word_boxes", [block])
    ink_values = []
    for wb in word_boxes:
        wx1 = max(0, wb["x"])
        wy1 = max(0, wb["y"])
        wx2 = min(w_img, wb.get("x2", wb["x"] + wb["w"]))
        wy2 = min(h_img, wb.get("y2", wb["y"] + wb["h"]))
        if wx2 <= wx1 or wy2 <= wy1:
            continue
        roi = gray[wy1:wy2, wx1:wx2].ravel()
        if len(roi) == 0:
            continue
        # Take the darkest 20% of pixels as "ink"
        n_ink = max(1, len(roi) // 5)
        darkest = np.partition(roi, n_ink)[:n_ink]
        ink_values.extend(darkest.tolist())

    if not ink_values:
        return 255.0

    ink_brightness = float(np.median(ink_values))
    contrast = bg_brightness - ink_brightness
    return max(0.0, contrast)


def _block_font_height(block: Dict[str, Any]) -> float:
    """Estimate the font height from word-level bounding boxes."""
    word_boxes = block.get("word_boxes", [])
    if word_boxes:
        heights = [wb.get("h", 0) for wb in word_boxes if wb.get("h", 0) > 0]
        if heights:
            return float(np.median(heights))
    return float(block.get("h", 0))


def _ink_color_hue(img_np: np.ndarray, block: Dict[str, Any]) -> Tuple[float, float]:
    """Return (dominant_hue, saturation) of the text ink.

    Stamps typically have high saturation and hue in red/blue range.
    Body text has low saturation (near-black).
    """
    h_img, w_img = img_np.shape[:2]
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV) if img_np.ndim == 3 else None
    if hsv is None:
        return 0.0, 0.0

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np

    word_boxes = block.get("word_boxes", [block])
    hues = []
    sats = []
    for wb in word_boxes:
        wx1 = max(0, wb["x"])
        wy1 = max(0, wb["y"])
        wx2 = min(w_img, wb.get("x2", wb["x"] + wb["w"]))
        wy2 = min(h_img, wb.get("y2", wb["y"] + wb["h"]))
        if wx2 <= wx1 or wy2 <= wy1:
            continue
        roi_gray = gray[wy1:wy2, wx1:wx2].ravel()
        roi_hsv = hsv[wy1:wy2, wx1:wx2].reshape(-1, 3)
        if len(roi_gray) == 0:
            continue
        # Find dark pixels (ink) using threshold
        thresh = np.percentile(roi_gray, 25)
        ink_mask = roi_gray <= thresh
        ink_hsv = roi_hsv[ink_mask]
        if len(ink_hsv) > 0:
            hues.extend(ink_hsv[:, 0].tolist())
            sats.extend(ink_hsv[:, 1].tolist())

    if not hues:
        return 0.0, 0.0

    return float(np.median(hues)), float(np.median(sats))


# ---------------------------------------------------------------------------
# Document-level statistics
# ---------------------------------------------------------------------------

def _compute_doc_stats(
    blocks: List[Dict[str, Any]],
    img_np: np.ndarray,
) -> Dict[str, float]:
    """Compute document-wide statistics used for anomaly detection."""
    font_heights = [_block_font_height(b) for b in blocks]
    contrasts = [_ink_contrast(img_np, b) for b in blocks]

    # Filter out zero / extreme values for robust median
    valid_heights = [h for h in font_heights if 5 < h < 200]
    valid_contrasts = [c for c in contrasts if c > 10]

    return {
        "median_height": float(np.median(valid_heights)) if valid_heights else 12.0,
        "p75_height": float(np.percentile(valid_heights, 75)) if valid_heights else 16.0,
        "median_contrast": float(np.median(valid_contrasts)) if valid_contrasts else 100.0,
        "p25_contrast": float(np.percentile(valid_contrasts, 25)) if valid_contrasts else 50.0,
    }


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_blocks(
    blocks: List[Dict[str, Any]],
    image: Image.Image,
    min_confidence: float = 35.0,
    watermark_contrast_ratio: float = 0.82,
    watermark_size_ratio: float = 1.8,
    stamp_saturation: float = 60.0,
) -> List[Dict[str, Any]]:
    """Classify each block and set ``block["_classification"]``.

    Blocks flagged as watermark/stamp/noise also get ``block["_skip"] = True``
    so the pipeline can exclude them from translation and inpainting.

    Args:
        blocks:                   OCR text blocks.
        image:                    Original document image (RGB PIL).
        min_confidence:           Blocks below this confidence → noise.
        watermark_contrast_ratio: If contrast < median * this ratio → watermark.
        watermark_size_ratio:     If font height > median * this ratio → watermark candidate.
        stamp_saturation:         Ink saturation above this → stamp candidate.

    Returns:
        The same blocks list, with ``_classification`` and ``_skip`` added.
    """
    if not blocks:
        return blocks

    img_np = np.array(image, dtype=np.uint8)
    stats = _compute_doc_stats(blocks, img_np)

    median_h = stats["median_height"]
    median_contrast = stats["median_contrast"]

    classified = {"translatable": 0, "watermark": 0, "stamp": 0, "noise": 0}

    for block in blocks:
        conf = block.get("mean_conf", 100.0)
        contrast = _ink_contrast(img_np, block)
        font_h = _block_font_height(block)
        hue, sat = _ink_color_hue(img_np, block)

        # Store features for debugging
        block["_contrast"] = round(contrast, 1)
        block["_font_h"] = round(font_h, 1)
        block["_ink_sat"] = round(sat, 1)

        # --- Rule 1: Very low confidence → noise ---
        if conf < min_confidence:
            block["_classification"] = NOISE
            block["_skip"] = True
            classified["noise"] += 1
            continue

        # --- Rule 2: Watermark detection ---
        # Watermarks have LOW contrast AND often LARGE size.
        # Primary: both conditions met.
        is_low_contrast = contrast < median_contrast * watermark_contrast_ratio
        is_oversized = font_h > median_h * watermark_size_ratio

        if is_low_contrast and is_oversized:
            block["_classification"] = WATERMARK
            block["_skip"] = True
            classified["watermark"] += 1
            continue

        # Secondary: extremely oversized (> 3.5x median) regardless of contrast
        if font_h > median_h * 3.5:
            block["_classification"] = WATERMARK
            block["_skip"] = True
            classified["watermark"] += 1
            continue

        # Tertiary: very low contrast (< 55% of median) regardless of size
        if contrast < median_contrast * 0.55:
            block["_classification"] = WATERMARK
            block["_skip"] = True
            classified["watermark"] += 1
            continue

        # --- Rule 3: Stamp detection ---
        # High saturation (coloured ink) + moderate/low confidence + oversized
        is_coloured = sat > stamp_saturation
        is_moderate_conf = conf < 60.0
        if is_coloured and is_moderate_conf and is_oversized:
            block["_classification"] = STAMP
            block["_skip"] = True
            classified["stamp"] += 1
            continue

        # --- Default: translatable ---
        block["_classification"] = TRANSLATABLE
        block["_skip"] = False
        classified["translatable"] += 1

    log.info(
        "Classification: %d translatable, %d watermark, %d stamp, %d noise",
        classified["translatable"], classified["watermark"],
        classified["stamp"], classified["noise"],
    )

    return blocks
