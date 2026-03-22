"""
Text removal via OpenCV inpainting.

Strategy
--------
1. Build a binary mask covering all text bounding boxes (with a small
   padding so no ink bleeds through).
2. Run cv2.inpaint (TELEA algorithm) to reconstruct the background.
3. For predominantly uniform (white/light) backgrounds also fill regions
   with the sampled background colour as a post-process to smooth results.
"""

from __future__ import annotations

from typing import List, Dict, Any

import cv2
import numpy as np
from PIL import Image


def remove_text_blocks(
    image: Image.Image,
    blocks: List[Dict[str, Any]],
    padding: int = 4,
    inpaint_radius: int = 5,
) -> Image.Image:
    """Erase text regions from *image* using inpainting.

    Uses **word-level** bounding boxes when available (``block["word_boxes"]``)
    so that table borders, lines, and other non-text elements inside a block's
    overall bounding box are preserved.  The mask is refined with ink-detection
    and morphological dilation to cleanly catch anti-aliased text edges.

    Args:
        image:         Input PIL Image (RGB).
        blocks:        Text block dicts from :func:`ocr_extractor.extract_text_blocks`.
        padding:       Extra pixels to expand each word box to cover ink edges.
        inpaint_radius: Neighbourhood radius used by cv2.inpaint.

    Returns:
        A new PIL Image with the text regions reconstructed/erased.
    """
    if not blocks:
        return image.copy()

    img_np = np.array(image, dtype=np.uint8)
    h_img, w_img = img_np.shape[:2]
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Build mask from word-level boxes, refined with ink detection
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for block in blocks:
        word_boxes = block.get("word_boxes")
        boxes = word_boxes if word_boxes else [block]
        for wb in boxes:
            x1 = max(0, wb["x"] - padding)
            y1 = max(0, wb["y"] - padding)
            x2 = min(w_img, wb.get("x2", wb["x"] + wb["w"]) + padding)
            y2 = min(h_img, wb.get("y2", wb["y"] + wb["h"]) + padding)

            # Within the padded word box, detect actual ink pixels:
            # find the local background brightness and threshold below it.
            roi_gray = gray[y1:y2, x1:x2]
            if roi_gray.size == 0:
                continue

            # Adaptive threshold: pixels definitively darker than the local
            # background are ink.  Use a wide margin to avoid catching
            # watermarks, stamps, and other semi-transparent overlays.
            bg_bright = np.percentile(roi_gray, 90)
            ink_thresh = bg_bright - 40
            ink_mask = (roi_gray < ink_thresh).astype(np.uint8) * 255

            # Also detect colored (saturated) text: red, blue, etc.
            # Use a high bar to avoid catching gray watermark tints.
            roi_sat = hsv[y1:y2, x1:x2, 1]
            bg_sat = np.percentile(roi_sat, 15)
            sat_thresh = max(bg_sat + 60, 70)
            color_ink = (roi_sat > sat_thresh).astype(np.uint8) * 255
            ink_mask = np.maximum(ink_mask, color_ink)

            if ink_mask.any():
                # Light dilation to cover anti-aliased edges only
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                ink_mask = cv2.dilate(ink_mask, kernel, iterations=1)
                mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], ink_mask)
            else:
                # No ink detected — skip this word box entirely rather
                # than blindly filling (avoids destroying watermarks)
                pass

    # OpenCV inpainting reconstructs the masked region from surroundings
    inpainted = cv2.inpaint(img_np, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)

    # Post-process: where background is nearly uniform, clean up with
    # the sampled background colour for a pristine result.
    inpainted = _smooth_uniform_backgrounds(inpainted, blocks, mask, padding)

    return Image.fromarray(inpainted)


def _median_bg_color(img_np: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                     w_img: int, h_img: int, border_width: int = 28) -> np.ndarray:
    """Robust background colour estimate using a wide border ring."""
    bx1, by1 = max(0, x1 - border_width), max(0, y1 - border_width)
    bx2, by2 = min(w_img, x2 + border_width), min(h_img, y2 + border_width)
    region = img_np[by1:by2, bx1:bx2]
    inner = np.zeros(region.shape[:2], dtype=bool)
    iy1, iy2 = y1 - by1, y2 - by1
    ix1_r, ix2_r = x1 - bx1, x2 - bx1
    inner[max(0, iy1):iy2, max(0, ix1_r):ix2_r] = True
    border_px = region[~inner].reshape(-1, region.shape[2])
    if len(border_px) == 0:
        return np.array([255, 255, 255], dtype=np.uint8)
    return np.median(border_px, axis=0).astype(np.uint8)


def _smooth_uniform_backgrounds(
    img_np: np.ndarray,
    blocks: List[Dict[str, Any]],
    mask: np.ndarray,
    padding: int,
    uniformity_threshold: float = 45.0,
) -> np.ndarray:
    """For each text block whose surrounding background is truly uniform,
    replace the word-box areas with the median background colour.

    For uniform backgrounds (white paper, solid cells) this produces a
    pristine result with no TELEA smudging.  Complex backgrounds keep
    the TELEA inpainting result untouched.
    """
    h_img, w_img = img_np.shape[:2]
    out = img_np.copy()
    # Extra margin beyond word-box padding to cover dilation zone
    fill_extra = 4

    for block in blocks:
        # Use the overall block bbox for border sampling
        bx1 = max(0, block["x"] - padding)
        by1 = max(0, block["y"] - padding)
        bx2 = min(w_img, block["x2"] + padding)
        by2 = min(h_img, block["y2"] + padding)

        border_pixels = _sample_border_pixels(img_np, bx1, by1, bx2, by2, w_img, h_img)
        if len(border_pixels) == 0:
            continue

        std = float(np.std(border_pixels))
        if std < uniformity_threshold:
            bg_colour = np.median(border_pixels, axis=0).astype(np.uint8)
            # For uniform backgrounds, fill ONLY where the mask indicates
            # actual ink/text.  This preserves watermarks, stamps, and
            # other non-text content that overlaps with word-box areas.
            word_boxes = block.get("word_boxes")
            if word_boxes:
                for wb in word_boxes:
                    x1 = max(0, wb["x"] - padding - fill_extra)
                    y1 = max(0, wb["y"] - padding - fill_extra)
                    x2 = min(w_img, wb.get("x2", wb["x"] + wb["w"]) + padding + fill_extra)
                    y2 = min(h_img, wb.get("y2", wb["y"] + wb["h"]) + padding + fill_extra)
                    sub_mask = mask[y1:y2, x1:x2]
                    fill_where = sub_mask > 0
                    roi = out[y1:y2, x1:x2]
                    roi[fill_where] = bg_colour
            else:
                fx1 = max(0, bx1 - fill_extra)
                fy1 = max(0, by1 - fill_extra)
                fx2 = min(w_img, bx2 + fill_extra)
                fy2 = min(h_img, by2 + fill_extra)
                sub_mask = mask[fy1:fy2, fx1:fx2]
                fill_where = sub_mask > 0
                roi = out[fy1:fy2, fx1:fx2]
                roi[fill_where] = bg_colour
        # else: complex background — keep the TELEA inpainting result as-is

    return out


def _sample_border_pixels(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    w_img: int, h_img: int,
    border_width: int = 28,
) -> np.ndarray:
    """Return pixel values from a border ring outside the given bbox."""
    bx1 = max(0, x1 - border_width)
    by1 = max(0, y1 - border_width)
    bx2 = min(w_img, x2 + border_width)
    by2 = min(h_img, y2 + border_width)

    region = img[by1:by2, bx1:bx2]

    # Create a mask that covers only the originally masked area
    inner_mask = np.zeros(region.shape[:2], dtype=bool)
    ix1 = x1 - bx1
    iy1 = y1 - by1
    ix2 = ix1 + (x2 - x1)
    iy2 = iy1 + (y2 - y1)
    inner_mask[max(0, iy1):iy2, max(0, ix1):ix2] = True

    border_mask = ~inner_mask
    pixels = region[border_mask].reshape(-1, region.shape[2])
    return pixels
