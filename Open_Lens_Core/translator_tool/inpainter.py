"""
Text removal via inpainting.

Supports two engines:
  - **LaMa** (default when available): deep-learning inpainting that handles
    complex textures, stamps, watermarks, and coloured backgrounds.
  - **OpenCV TELEA** (fallback): classical algorithm, fast, works on CPU.

Both paths use word-level bounding boxes for precise mask generation.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def remove_text_blocks(
    image: Image.Image,
    blocks: List[Dict[str, Any]],
    padding: int = 8,
    inpaint_radius: int = 5,
) -> Image.Image:
    """Erase text regions from *image* using inpainting.

    Uses **word-level** bounding boxes when available (``block["word_boxes"]``)
    so that table borders, lines, and other non-text elements inside a block's
    overall bounding box are preserved.  All OCR-confirmed word boxes are
    masked unconditionally, then the mask is dilated to cleanly catch
    anti-aliased text edges and descenders.

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

    # Build mask from word-level boxes.
    # OCR confirmed text in every block that reaches this point, so we
    # always mask the padded word box.
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for block in blocks:
        pad = _dynamic_padding(block, padding)
        word_boxes = block.get("word_boxes")
        boxes = word_boxes if word_boxes else [block]
        for wb in boxes:
            x1 = max(0, wb["x"] - pad)
            y1 = max(0, wb["y"] - pad)
            x2 = min(w_img, wb.get("x2", wb["x"] + wb["w"]) + pad)
            y2 = min(h_img, wb.get("y2", wb["y"] + wb["h"]) + pad)
            mask[y1:y2, x1:x2] = 255

    # Light dilation to catch anti-aliased text edges — small kernel,
    # single iteration to avoid over-expansion that smudges backgrounds.
    if mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)

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


# ---------------------------------------------------------------------------
# LaMa inpainting engine
# ---------------------------------------------------------------------------

_LAMA_MODEL = None


def _get_lama_model():
    """Return a cached SimpleLama instance."""
    global _LAMA_MODEL
    if _LAMA_MODEL is not None:
        return _LAMA_MODEL

    from simple_lama_inpainting import SimpleLama  # type: ignore

    _LAMA_MODEL = SimpleLama()
    log.info("LaMa inpainting model loaded")
    return _LAMA_MODEL


def _dynamic_padding(block: Dict[str, Any], base_padding: int = 8) -> int:
    """Compute padding proportional to the text height in a block.

    Larger text needs more padding to cover anti-aliased edges and
    descenders, but the padding is capped to prevent over-masking that
    causes smudging on complex backgrounds.
    """
    word_boxes = block.get("word_boxes")
    heights = []
    if word_boxes:
        heights = [wb.get("h", 0) for wb in word_boxes if wb.get("h", 0) > 0]
    if not heights:
        heights = [block.get("h", 0)]
    avg_h = sum(heights) / len(heights) if heights else 0
    # Scale padding: 15% of average text height, capped to prevent over-masking.
    # Smaller padding + light dilation is better than huge padding.
    return max(base_padding, min(int(avg_h * 0.15), 14))


def _build_mask(
    image: Image.Image,
    blocks: List[Dict[str, Any]],
    padding: int = 8,
) -> np.ndarray:
    """Build a binary mask covering all text regions using word-level boxes,
    then dilate to catch anti-aliased edges and descenders."""
    img_np = np.array(image, dtype=np.uint8)
    h_img, w_img = img_np.shape[:2]

    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for block in blocks:
        pad = _dynamic_padding(block, padding)
        word_boxes = block.get("word_boxes")
        boxes = word_boxes if word_boxes else [block]
        for wb in boxes:
            x1 = max(0, wb["x"] - pad)
            y1 = max(0, wb["y"] - pad)
            x2 = min(w_img, wb.get("x2", wb["x"] + wb["w"]) + pad)
            y2 = min(h_img, wb.get("y2", wb["y"] + wb["h"]) + pad)
            mask[y1:y2, x1:x2] = 255

    # Light dilation — small kernel, single pass to avoid smudging.
    if mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def remove_text_lama(
    image: Image.Image,
    blocks: List[Dict[str, Any]],
    padding: int = 8,
) -> Image.Image:
    """Remove text using LaMa deep-learning inpainting."""
    if not blocks:
        return image.copy()

    mask_np = _build_mask(image, blocks, padding)

    if not mask_np.any():
        return image.copy()

    lama = _get_lama_model()

    # SimpleLama expects PIL Images
    mask_pil = Image.fromarray(mask_np).convert("L")
    result = lama(image.convert("RGB"), mask_pil)

    # Post-process uniform backgrounds (same as TELEA path)
    result_np = np.array(result, dtype=np.uint8)
    result_np = _smooth_uniform_backgrounds(result_np, blocks, mask_np, padding)

    return Image.fromarray(result_np)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def remove_text(
    image: Image.Image,
    blocks: List[Dict[str, Any]],
    engine: str = "auto",
    padding: int = 8,
    inpaint_radius: int = 5,
) -> Image.Image:
    """Remove text regions using the best available engine.

    Args:
        image:          RGB PIL Image.
        blocks:         Text block dicts with bounding boxes.
        engine:         ``"lama"``, ``"telea"``, or ``"auto"``.
        padding:        Pixel padding around word boxes.
        inpaint_radius: Radius for TELEA inpainting (ignored by LaMa).

    Returns:
        A new PIL Image with text regions erased.
    """
    if not blocks:
        return image.copy()

    if engine == "auto":
        try:
            from simple_lama_inpainting import SimpleLama  # noqa: F401
            engine = "lama"
        except Exception:
            engine = "telea"

    if engine == "lama":
        try:
            return remove_text_lama(image, blocks, padding)
        except Exception as exc:
            log.warning("LaMa inpainting failed (%s), falling back to TELEA", exc)

    return remove_text_blocks(image, blocks, padding, inpaint_radius)
