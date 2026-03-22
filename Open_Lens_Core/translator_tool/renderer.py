"""
Translated text rendering.

For each block:
  1. Sample the original text colour from the pre-inpainted image.
  2. Auto-fit a TrueType font so the translated string fills the bbox.
  3. Wrap the text across multiple lines if needed.
  4. Handle RTL languages via python-bidi + arabic-reshaper (optional).
"""

from __future__ import annotations

import os
import sys
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .language_utils import is_rtl


# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

# Ordered list of fonts with good Unicode / multilingual coverage
_WINDOWS_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialuni.ttf",   # Arial Unicode MS (best coverage)
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
    r"C:\Windows\Fonts\times.ttf",
]

_WINDOWS_BOLD_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\tahomabd.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\verdanab.ttf",
    r"C:\Windows\Fonts\timesbd.ttf",
]

_WINDOWS_ITALIC_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\ariali.ttf",
    r"C:\Windows\Fonts\calibrii.ttf",
    r"C:\Windows\Fonts\segoeuii.ttf",
    r"C:\Windows\Fonts\verdanai.ttf",
    r"C:\Windows\Fonts\timesi.ttf",
]

_LINUX_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]

_MACOS_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/SFNS.ttf",
]


def find_system_font() -> Optional[str]:
    """Return the path to the best available system font, or None."""
    if sys.platform == "win32":
        candidates = _WINDOWS_FONT_CANDIDATES
    elif sys.platform == "darwin":
        candidates = _MACOS_FONT_CANDIDATES
    else:
        candidates = _LINUX_FONT_CANDIDATES

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _find_bold_font() -> Optional[str]:
    """Return path to a bold system font, or None."""
    if sys.platform == "win32":
        candidates = _WINDOWS_BOLD_FONT_CANDIDATES
    else:
        return None  # TODO: add Linux/macOS bold candidates
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _find_italic_font() -> Optional[str]:
    """Return path to an italic system font, or None."""
    if sys.platform == "win32":
        candidates = _WINDOWS_ITALIC_FONT_CANDIDATES
    else:
        return None
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            pass
    # Fallback: Pillow's built-in bitmap font (fixed size, limited charset)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Text metrics
# ---------------------------------------------------------------------------

def _text_bbox(text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    """Return (width, height) of *text* rendered in *font*."""
    try:
        bbox = font.getbbox(text)   # (left, top, right, bottom)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Older Pillow
        w, h = font.getsize(text)
        return w, h


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Word-wrap *text* so each line fits within *max_width* pixels."""
    words = text.split()
    if not words:
        return []

    lines: List[str] = []
    current: List[str] = []

    for word in words:
        candidate = " ".join(current + [word])
        w, _ = _text_bbox(candidate, font)
        if w <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]

    if current:
        lines.append(" ".join(current))

    return lines or [text]


def _fit_font(
    text: str,
    max_width: int,
    max_height: int,
    font_path: Optional[str],
    min_size: int = 6,
    max_size: Optional[int] = None,
) -> Tuple[ImageFont.FreeTypeFont, List[str], int]:
    """Find the largest font size for which *text* fits in the bounding box.

    Returns (font, wrapped_lines, font_size).
    """
    if max_size is None:
        # Start from the height of the box as an upper bound
        max_size = max(max_height, 12)

    # Use a step-based search for speed: try larger steps first, then refine
    best = None
    for size in range(max_size, min_size - 1, -1):
        font = _load_font(font_path, size)
        lines = _wrap_text(text, font, max_width)
        _, line_h = _text_bbox("Ag", font)
        line_spacing = line_h + 2
        total_h = line_spacing * len(lines)
        max_line_w = max((_text_bbox(ln, font)[0] for ln in lines), default=0)

        if max_line_w <= max_width and total_h <= max_height:
            return font, lines, size

    # Minimum size fallback
    font = _load_font(font_path, min_size)
    lines = _wrap_text(text, font, max_width)
    return font, lines, min_size


def _estimate_original_font_size(block: Dict[str, Any]) -> Optional[int]:
    """Estimate original font size from word-level OCR bounding boxes.

    Uses the 25th-percentile word height (conservative) scaled down to
    account for OCR bbox padding, ascenders/descenders, and the tendency
    of Tesseract boxes to be taller than the visual font size.
    """
    word_boxes = block.get("word_boxes", [])
    if not word_boxes:
        return None
    heights = sorted(wb["h"] for wb in word_boxes if wb.get("h", 0) > 0)
    if not heights:
        return None
    # Use 25th percentile (robust against outlier tall boxes from
    # watermarks, stamps, or ascender-heavy characters).
    p25_idx = max(0, len(heights) // 4)
    ref_h = heights[p25_idx]
    return max(6, int(ref_h * 0.60))


# ---------------------------------------------------------------------------
# Colour sampling
# ---------------------------------------------------------------------------

def _sample_text_color(
    original_image: Image.Image,
    block: Dict[str, Any],
) -> Tuple[int, int, int]:
    """Estimate the ink colour in a block by finding the darkest pixel cluster."""
    x, y, x2, y2 = block["x"], block["y"], block["x2"], block["y2"]
    region = np.array(original_image.crop((x, y, x2, y2)))

    if region.size == 0:
        return (0, 0, 0)

    # Grayscale brightness
    gray = 0.299 * region[:, :, 0] + 0.587 * region[:, :, 1] + 0.114 * region[:, :, 2]
    threshold = np.percentile(gray, 20)  # bottom 20 % = ink
    dark_mask = gray < threshold

    if dark_mask.sum() < 5:
        return (0, 0, 0)

    dark_pixels = region[dark_mask]
    colour = tuple(np.median(dark_pixels, axis=0).astype(int)[:3])
    return colour  # type: ignore[return-value]


def _detect_font_weight(
    original_image: Image.Image,
    block: Dict[str, Any],
) -> str:
    """Detect whether the original text appears bold or italic.

    Returns ``"bold"``, ``"italic"``, or ``"regular"``.

    Heuristic: bold text has thicker strokes, so the proportion of dark
    pixels in the bounding box is higher.  ALL-CAPS text with high ink
    density is also a strong bold signal.
    """
    x, y, x2, y2 = block["x"], block["y"], block["x2"], block["y2"]
    region = np.array(original_image.crop((x, y, x2, y2)))
    if region.size == 0:
        return "regular"

    gray = 0.299 * region[:, :, 0] + 0.587 * region[:, :, 1] + 0.114 * region[:, :, 2]
    bg_bright = np.percentile(gray, 85)
    ink_mask = gray < (bg_bright - 30)
    ink_ratio = ink_mask.sum() / max(gray.size, 1)

    text = block.get("text", "")
    is_upper = text == text.upper() and len(text) > 3

    # Bold: high ink density or ALL-CAPS header
    if ink_ratio > 0.25 or (is_upper and ink_ratio > 0.15):
        return "bold"

    # Italic: very low ink density with narrow strokes (heuristic)
    if ink_ratio < 0.08 and not is_upper:
        return "italic"

    return "regular"


# ---------------------------------------------------------------------------
# RTL support (optional)
# ---------------------------------------------------------------------------

def _prepare_rtl_text(text: str, lang_code: str) -> str:
    """Reshape and reorder Arabic/Hebrew text for correct rendering."""
    if not is_rtl(lang_code):
        return text
    try:
        import arabic_reshaper  # type: ignore
        from bidi.algorithm import get_display  # type: ignore
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except ImportError:
        # Libraries not installed – render as-is (may look reversed)
        return text


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_translated_blocks(
    image: Image.Image,
    blocks: List[Dict[str, Any]],
    original_image: Optional[Image.Image] = None,
    font_path: Optional[str] = None,
    target_lang: str = "en",
    default_text_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Draw translated text onto *image* inside each block's bounding box.

    Args:
        image:              The inpainted (background-only) PIL Image to draw on.
        blocks:             Block dicts that have ``translated_text`` populated.
        original_image:     Original image used to sample ink colour (optional).
        font_path:          Path to a .ttf font file. Auto-detected if None.
        target_lang:        ISO language code of the translated text (for RTL).
        default_text_color: Fallback text colour as RGB tuple.

    Returns:
        A new PIL Image with translated text rendered in place.
    """
    if font_path is None:
        font_path = find_system_font()

    bold_font_path = _find_bold_font()
    italic_font_path = _find_italic_font()

    result = image.copy()
    draw = ImageDraw.Draw(result)

    rtl = is_rtl(target_lang)

    for block in blocks:
        translated = block.get("translated_text", block.get("text", ""))
        if not translated or not translated.strip():
            continue

        x, y = block["x"], block["y"]
        w, h = block["w"], block["h"]

        # Determine text colour
        if original_image is not None:
            text_color = _sample_text_color(original_image, block)
        else:
            text_color = default_text_color

        # Detect font weight from original image
        weight = "regular"
        if original_image is not None:
            weight = _detect_font_weight(original_image, block)

        # Pick font variant
        if weight == "bold" and bold_font_path:
            active_font_path = bold_font_path
        elif weight == "italic" and italic_font_path:
            active_font_path = italic_font_path
        else:
            active_font_path = font_path

        # Prepare text (RTL reshaping if needed)
        display_text = _prepare_rtl_text(translated.strip(), target_lang)

        # Estimate original font size from word bounding box heights (most
        # reliable) to avoid blowing up text larger than the original.
        estimated_size = _estimate_original_font_size(block)
        if estimated_size is None:
            original_text = block.get("text", "")
            if original_text and original_text.strip():
                _, _, estimated_size = _fit_font(original_text.strip(), w, h, active_font_path)

        # Fit translated text, capped at the original visual font size
        font, lines, font_size = _fit_font(display_text, w, h, active_font_path, max_size=estimated_size)

        # Compute line height with tight spacing to match document formatting
        _, line_h = _text_bbox("Ag", font)
        line_spacing = line_h + 2

        # Top-align text within the bounding box (matches document formatting)
        current_y = y

        for line in lines:
            if current_y + line_h > y + h:
                break  # strict overflow – don't draw past bounding box

            if rtl:
                line_w, _ = _text_bbox(line, font)
                draw_x = x + w - line_w  # right-align for RTL
            else:
                draw_x = x

            draw.text((draw_x, current_y), line, font=font, fill=text_color)
            current_y += line_spacing

    return result
