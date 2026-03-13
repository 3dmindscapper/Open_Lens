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
    padding: int = 6,
    inpaint_radius: int = 14,
) -> Image.Image:
    """Erase text regions from *image* using inpainting.

    Args:
        image:         Input PIL Image (RGB).
        blocks:        Text block dicts from :func:`ocr_extractor.extract_text_blocks`.
        padding:       Extra pixels to expand each bounding box to cover ink edges.
        inpaint_radius: Neighbourhood radius used by cv2.inpaint.

    Returns:
        A new PIL Image with the text regions reconstructed/erased.
    """
    if not blocks:
        return image.copy()

    img_np = np.array(image, dtype=np.uint8)
    h_img, w_img = img_np.shape[:2]

    # Build mask: 255 where text exists, 0 elsewhere
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for block in blocks:
        x1 = max(0, block["x"] - padding)
        y1 = max(0, block["y"] - padding)
        x2 = min(w_img, block["x2"] + padding)
        y2 = min(h_img, block["y2"] + padding)
        mask[y1:y2, x1:x2] = 255

    # OpenCV inpainting reconstructs the masked region from surroundings
    inpainted = cv2.inpaint(img_np, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)

    # Post-process: where background is nearly uniform (std < threshold),
    # flood-fill with the local background colour for a cleaner result.
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
    uniformity_threshold: float = 50.0,
) -> np.ndarray:
    """For each text block whose surrounding background is nearly uniform,
    replace the block area with the median background colour."""
    h_img, w_img = img_np.shape[:2]
    out = img_np.copy()

    for block in blocks:
        x1 = max(0, block["x"] - padding)
        y1 = max(0, block["y"] - padding)
        x2 = min(w_img, block["x2"] + padding)
        y2 = min(h_img, block["y2"] + padding)

        # Sample a border ring around the block to estimate background
        border_pixels = _sample_border_pixels(img_np, x1, y1, x2, y2, w_img, h_img)
        if len(border_pixels) == 0:
            continue

        std = float(np.std(border_pixels))
        if std < uniformity_threshold:
            bg_colour = np.median(border_pixels, axis=0).astype(np.uint8)
            out[y1:y2, x1:x2] = bg_colour
        else:
            # Background is complex; use wide-ring median for better accuracy
            h_out, w_out = out.shape[:2]
            bg_colour = _median_bg_color(img_np, x1, y1, x2, y2, w_out, h_out)
            out[y1:y2, x1:x2] = bg_colour

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
